"""Conversational image generation: the chat model decides to draw by writing <|visual prompt|>...<|/visual prompt|>, then the
image decoder draws that description. The decoder reads descriptions with its own frozen chat model (the checkpoint it was
trained with), not with the conversation model, which went on training.

A change to the image is a new full description written by the chat model, drawn from scratch: with the same seed the result
stays similar at best. The conversation only remembers the descriptions, so questions about an image are answered from them.
"""
import torch

from common import common
from common.data_paths import T2I_VQ_MODEL
from inference.engine import Engine
from model import gpt
from model.imagegen import ImageGen, ImageGenConfig, load_vq
from tokenizer import tokenizer


def load_gpt(path, tokeniz, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), device, tokeniz)
    model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
    return model.requires_grad_(False).eval()


def parse_reply(tokens, special, decode):
    """the tokens of one assistant turn -> (text, description or None, the picture it changes or None); an unfinished description
    does not count. <|edit prompt|> holds <|image ref|>N<|/image ref|> and then the complete new description."""
    stops = [special["<|/assistant|>"], special["<|image|>"]]
    for i, t in enumerate(tokens):
        if t in stops:
            tokens = tokens[:i]
            break
    for start, end, is_edit in ((special["<|visual prompt|>"], special["<|/visual prompt|>"], False),
                                (special["<|edit prompt|>"], special["<|/edit prompt|>"], True)):
        if start not in tokens:
            continue
        s = tokens.index(start)
        text = decode(tokens[:s]).strip() if s else ""
        if end not in tokens[s:]:
            return text, None, None
        body = tokens[s + 1:tokens.index(end, s)]
        ref = None
        if is_edit:
            ref_start, ref_end = special["<|image ref|>"], special["<|/image ref|>"]
            if ref_start not in body or ref_end not in body:
                return text, None, None
            ref_text = decode(body[body.index(ref_start) + 1:body.index(ref_end)]).strip()
            ref = int(ref_text) if ref_text.isdigit() else -1
            body = body[body.index(ref_end) + 1:]
        return text, decode(body).strip() if body else None, ref
    return decode(tokens).strip() if tokens else "", None, None


class ChatT2I:
    def __init__(self, chat_ckpt, tokenizer_path, decoder_ckpt=None, vq_dir=str(T2I_VQ_MODEL), device="cuda:0"):
        self.device = device
        self.tokeniz = tokenizer.Tokenizer.from_dict(tokenizer_path, device=device)
        self.chat = load_gpt(chat_ckpt, self.tokeniz, device)
        self.engine = Engine(self.chat, self.tokeniz, device)
        self.decoder = None
        if decoder_ckpt:
            ckpt = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
            self.thinker = load_gpt(ckpt["thinker_ckpt"], self.tokeniz, device)
            self.decoder = ImageGen(ImageGenConfig(**ckpt["config"]), self.thinker.config.model_dim, device)
            self.decoder.load_state_dict(ckpt["model_state"])
            self.decoder.eval()
            self.vq = load_vq(vq_dir, device)

    def reply(self, messages, top_k=2, max_new_tokens=200):  # top_k 2 keeps edits faithful (82% of the words, 8 keeps 71%), first pictures still differ
        """messages end with the user's turn -> the assistant message to append to them, and its description or None. A picture gets
        the next number in an image_id part; an edit's source picture is in its i2i_prompt part, clamped to pictures that exist."""
        special = self.tokeniz.special_token
        history = list(messages)
        while True:
            tokens, _ = self.tokeniz.render_conversation({"source": "chat", "messages": history}, last_is_user=True)
            if len(tokens) + max_new_tokens <= self.chat.config.seq_len or len(history) <= 1:
                break
            history = history[2:]  # the oldest exchange goes first
        parts = [p for m in messages if isinstance(m["content"], list) for p in m["content"]]
        count = sum(p["type"] == "image_id" for p in parts)
        previous = [p["content"] for p in parts if p["type"] in ("t2i_prompt", "i2i_prompt")]
        stops = {special["<|/assistant|>"], special["<|image|>"]}
        for _ in range(3):  # a description repeated word for word draws the same picture while claiming a change
            out = []
            stream = self.engine.generate(torch.tensor(tokens, device=self.device), 1, k=top_k, max_len=len(tokens) + max_new_tokens + 8)
            for _ in range(max_new_tokens):
                token = int(next(stream)[0][0])
                out.append(token)
                if token in stops:
                    break
            stream.close()
            text, prompt, ref = parse_reply(out, special, self.tokeniz.decode)
            if top_k == 1 or not prompt or not previous or prompt != previous[-1] or ref is not None and ref != count - 1:
                break
        content = [{"type": "text", "content": text}] if text else []
        if prompt:
            if ref is not None and count:  # a picture to change exists
                ref = ref if 0 <= ref < count else count - 1
                content.append({"type": "i2i_prompt", "image_ref": str(ref), "content": prompt})
            else:
                content.append({"type": "t2i_prompt", "content": prompt})
            content.append({"type": "image_id", "image_ref": str(count)})
        return {"role": "assistant", "content": content}, prompt

    @torch.no_grad()
    def draw(self, prompt, seed=0, cfg_scale=7.5, temperature=0.9):
        """uint8 array (H, W, 3)"""
        special = self.tokeniz.special_token
        ids = [special["<|user|>"]] + self.tokeniz.encode_ordinary(prompt) + [special["<|/user|>"]]
        ids = ids[:self.decoder.config.prefix_capacity]
        hidden = self.thinker(torch.tensor([ids], device=self.device), hidden_layer=self.thinker.config.n_layer - 1, hidden_only=True).hidden
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.vq(self.decoder.sample(hidden, temperature=temperature, cfg_scale=cfg_scale, generator=generator))
