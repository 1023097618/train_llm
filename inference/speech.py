"""Speak: the thinker writes the reply while the talker produces Mimi codes in lockstep, Mimi decodes them to audio."""
import torch

from model.speech import NUM_CODEBOOKS


def sample_top_k(logits, temperature, top_k):
    if temperature <= 0:
        return logits.argmax(-1)
    logits = logits / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    return torch.multinomial(logits.softmax(-1), 1).squeeze(-1)


@torch.no_grad()
def generate_speech(thinker, talker, tokeniz, prompt, hidden_layer, media_embeds=None, spk_emb=None, max_steps=400,
                    text_temperature=0.0, audio_temperature=0.8, audio_top_k=50):
    """prompt: tokens ending with <|assistant|>; a single sequence. Returns (reply text, codes (8, frames))"""
    device = next(talker.parameters()).device
    c = talker.config
    end_assistant = tokeniz.special_token["<|/assistant|>"]
    eos = tokeniz.special_token["<|/doc|>"]
    x = torch.tensor([prompt], device=device)
    total = min(len(prompt) + max_steps + NUM_CODEBOOKS + 1, thinker.config.seq_len)
    thinker_cache, talker_cache = thinker.get_kv_cache(1, seq_len=total), talker.get_kv_cache(1, seq_len=total)
    audio = torch.full((1, NUM_CODEBOOKS, len(prompt)), c.audio_pad_token, device=device)
    audio[0, 0, -1] = c.audio_start_token  # codebook j starts j positions later
    out = thinker(x, kv_cache=thinker_cache, media_embeds=media_embeds, hidden_layer=hidden_layer)
    audio_logits = talker(audio, out.hidden, x, kv_cache=talker_cache)
    text_out, codes = [], [[] for _ in range(NUM_CODEBOOKS)]
    text_done, stop_step = False, None
    for step in range(total - len(prompt) - 1):
        if text_done:
            token = eos  # the talker may still be speaking
        else:
            token = int(sample_top_k(out.logit[0, -1], text_temperature, 0))
            if token == end_assistant:
                text_done = True
            else:
                text_out.append(token)
        next_audio = torch.full((1, NUM_CODEBOOKS, 1), c.audio_pad_token, device=device)
        for j in range(NUM_CODEBOOKS):
            if step + 1 == j:
                next_audio[0, j, 0] = c.audio_start_token
            elif step >= j:
                logits = audio_logits[0, j, -1].clone()
                logits[2048:] = -float("inf")
                if j == 0:
                    logits[c.audio_stop_token] = audio_logits[0, 0, -1, c.audio_stop_token]
                code = int(sample_top_k(logits, audio_temperature, audio_top_k))
                if j == 0 and code == c.audio_stop_token:
                    stop_step = step if stop_step is None else stop_step
                elif code < 2048 and (stop_step is None or j > 0):
                    codes[j].append(code)
                    next_audio[0, j, 0] = code
        if stop_step is not None and step >= stop_step + NUM_CODEBOOKS:
            break
        next_text = torch.tensor([[token]], device=device)
        out = thinker(next_text, kv_cache=thinker_cache, hidden_layer=hidden_layer)
        audio_logits = talker(next_audio, out.hidden, next_text, kv_cache=talker_cache)
    frames = min(len(c_) for c_ in codes)
    return tokeniz.decode(text_out), torch.tensor([c_[:frames] for c_ in codes], dtype=torch.long)


class MimiDecoder:
    def __init__(self, model_dir, device):
        from transformers import MimiModel
        self.model = MimiModel.from_pretrained(str(model_dir)).to(device).eval()
        self.device = device
        self.sample_rate = int(self.model.config.sampling_rate)

    @torch.no_grad()
    def decode(self, codes):
        """(8, frames) -> mono float waveform (numpy)"""
        return self.model.decode(codes[None].to(self.device)).audio_values[0, 0].float().cpu().numpy()
