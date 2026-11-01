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
    """the tokens of one assistant turn -> (text, description or None); an unfinished description does not count"""
    start, end = special["<|visual prompt|>"], special["<|/visual prompt|>"]
    stop = special["<|/assistant|>"]
    if stop in tokens:
        tokens = tokens[:tokens.index(stop)]
    if start not in tokens:
        return decode(tokens).strip() if tokens else "", None
    s = tokens.index(start)
    text = decode(tokens[:s]).strip() if s else ""
    if end not in tokens[s:]:
        return text, None
    prompt = tokens[s + 1:tokens.index(end, s)]
    return text, decode(prompt).strip() if prompt else None


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

    def reply(self, messages, top_k=8, max_new_tokens=200):
        """messages end with the user's turn -> the assistant message to append to them, and its description or None"""
        special = self.tokeniz.special_token
        history = list(messages)
        while True:
            tokens, _ = self.tokeniz.render_conversation({"source": "chat", "messages": history}, last_is_user=True)
            if len(tokens) + max_new_tokens <= self.chat.config.seq_len or len(history) <= 1:
                break
            history = history[2:]  # the oldest exchange goes first
        sentence, _ = self.engine.generate_batch(tokens, 1, max_len=len(tokens) + max_new_tokens, max_new_token=max_new_tokens, k=top_k)
        text, prompt = parse_reply(sentence[0][len(tokens):], special, self.tokeniz.decode)
        content = [{"type": "text", "content": text}] if text else []
        if prompt:
            content.append({"type": "t2i_prompt", "content": prompt})
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
