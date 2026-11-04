"""Conversational image generation: the chat model decides to draw by writing <|visual prompt|>...<|/visual prompt|>, or to change an
earlier picture with <|edit prompt|><|image ref|>N<|/image ref|>...<|/edit prompt|>, always with the complete description. The program
draws it and puts the picture back into the conversation as <|image|><|image ref|>M<|/image ref|><|image pad|>*n<|/image|>: the frozen
SigLIP2 and the projector turn it into embeddings, so the model sees what it drew like a picture from the user.
"""
import numpy as np
import torch
from PIL import Image

from common import common
from common.data_paths import VLM_ENCODER
from inference.engine import Engine
from model import gpt
from model.vision import VisionConfig, VisionTower, preprocess
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
    def __init__(self, chat_ckpt, tokenizer_path, device="cuda:0"):
        self.device = device
        self.tokeniz = tokenizer.Tokenizer.from_dict(tokenizer_path, device=device)
        self.chat = load_gpt(chat_ckpt, self.tokeniz, device)
        ckpt = torch.load(chat_ckpt, map_location="cpu", weights_only=False)
        self.vision_config = VisionConfig(**{**ckpt.get("vision_config", {}), "encoder_path": str(VLM_ENCODER)})
        self.vision = VisionTower(self.vision_config, self.chat.config.model_dim, device)
        self.vision.projector.load_state_dict(ckpt["projector_state"])
        self.vision.requires_grad_(False).eval()
        del ckpt
        self.engine = Engine(self.chat, self.tokeniz, device)

    @torch.no_grad()
    def seen_pictures(self, history, tokens, pictures):
        """GPT media_embeds for the image parts of the rendered history; pictures: {number: uint8 array}, a lost one is plain gray"""
        arrays = []
        for m in history:
            if isinstance(m["content"], list):
                for p in m["content"]:
                    if p["type"] == "image":
                        arrays.append(pictures.get(int(p["image_ref"]), np.full((256, 256, 3), 128, dtype=np.uint8)))
        if not arrays:
            return None
        pixels = preprocess([Image.fromarray(a) for a in arrays], self.vision_config.image_size).to(self.device)
        return self.vision.image_embeds(torch.tensor([tokens], device=self.device), pixels, self.chat.config.model_dim, self.tokeniz.special_token["<|image pad|>"])

    def reply(self, messages, pictures=None, top_k=2, max_new_tokens=200):  # top_k 2 keeps edits faithful (82% of the words, 8 keeps 71%), first pictures still differ
        """messages end with the user's turn, pictures holds the earlier ones -> the assistant message to append, and its description or
        None. The new picture gets the next number in an image part; an edit's source is in its i2i_prompt part, clamped to pictures that exist."""
        special = self.tokeniz.special_token
        history = list(messages)
        while True:
            tokens, _ = self.tokeniz.render_conversation({"source": "chat", "messages": history}, last_is_user=True)
            if len(tokens) + max_new_tokens <= self.chat.config.seq_len or len(history) <= 1:
                break
            history = history[2:]  # the oldest exchange goes first
        parts = [p for m in messages if isinstance(m["content"], list) for p in m["content"]]
        count = sum(p["type"] == "image" for p in parts)
        embeds = self.seen_pictures(history, tokens, pictures or {})
        previous = [p["content"] for p in parts if p["type"] in ("t2i_prompt", "i2i_prompt")]
        stops = {special["<|/assistant|>"], special["<|image|>"]}
        for _ in range(3):  # a description repeated word for word draws the same picture while claiming a change
            out = []
            stream = self.engine.generate(torch.tensor(tokens, device=self.device), 1, k=top_k, max_len=len(tokens) + max_new_tokens + 8, media_embeds=embeds)
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
            content.append({"type": "image", "length": self.vision_config.tokens_per_image, "image_ref": str(count)})
        return {"role": "assistant", "content": content}, prompt
