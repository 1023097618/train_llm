"""Image generation: a frozen chat model reads the caption, a separate GPT decoder writes the LlamaGen VQ tokens of the
image autoregressively (16 x 16 tokens for 256 x 256 pixels), classifier free guidance at sampling time. The route of
mymultigpt's VisualAR; the decoder reuses model/gpt.py: its "vocabulary" is the VQ codebook, the caption positions hold
the COND token and take their embeddings from the caption through GPT.forward's media_embeds.
"""
import importlib.util
import sys
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from common.common import Linear
from model import gpt

CODEBOOK = 16384
VISUAL_EOS = CODEBOOK     # predicted after the last token, keeps "the image is complete" in the loss like VisualAR
COND = CODEBOOK + 1       # input only: caption and padding positions
GRID = 16


class VisualVocab:
    """what GPT needs from a tokenizer"""
    n_vocab = CODEBOOK + 1
    special_token = {"<|image pad|>": COND}


@dataclass
class ImageGenConfig:
    depth: int = 12
    seq_len: int = 512
    prefix_capacity: int = 256   # seq_len - GRID ** 2
    condition_dropout: float = 0.1


class ImageGen(nn.Module):
    def __init__(self, config: ImageGenConfig, thinker_dim, device):
        super().__init__()
        self.config = config
        self.decoder = gpt.GPT(gpt.GPTConfig.from_depth(config.depth, vocab_size=CODEBOOK + 2, seq_len=config.seq_len,
                                                         window_pattern="SSSL"), device, VisualVocab())
        dim = self.decoder.config.model_dim
        self.caption_proj = nn.Sequential(Linear(thinker_dim, dim * 2, bias=False), nn.GELU(), Linear(dim * 2, dim, bias=False))
        nn.init.normal_(self.caption_proj[0].weight, std=thinker_dim ** -0.5)
        nn.init.normal_(self.caption_proj[2].weight, std=(dim * 2) ** -0.5)
        # stands for "no caption" (CFG), and fills the padding after the image
        self.uncond = nn.Parameter(torch.randn(config.seq_len, dim) * 0.02)
        self.to(device)

    def media(self, caption_hidden, lengths, drop=None):
        """dense (B, seq_len, D): projected caption at positions < lengths, otherwise the unconditional embedding"""
        B, cap = caption_hidden.shape[:2]
        T = self.config.seq_len
        projected = self.caption_proj(caption_hidden.float())
        out = self.uncond.unsqueeze(0).expand(B, -1, -1)
        is_caption = torch.arange(T, device=out.device)[None, :] < lengths[:, None]
        if drop is not None:
            is_caption = is_caption & ~drop[:, None]
        padded = F.pad(projected, (0, 0, 0, T - cap))
        return torch.where(is_caption.unsqueeze(-1), padded, out)

    def layout(self, visual_ids, lengths):
        """input x and targets y (B, seq_len): the last caption position predicts token 0, the last token predicts EOS"""
        B = visual_ids.shape[0]
        T = self.config.seq_len
        count = visual_ids.shape[1]
        device = visual_ids.device
        pos = torch.arange(T, device=device)[None, :].expand(B, -1)
        rel_x = pos - lengths[:, None]
        x = torch.where((rel_x >= 0) & (rel_x < count), visual_ids.gather(1, rel_x.clamp(0, count - 1)), torch.full_like(pos, COND))
        rel_y = rel_x + 1
        y = torch.where((rel_y >= 0) & (rel_y < count), visual_ids.gather(1, rel_y.clamp(0, count - 1)), torch.full_like(pos, -1))
        y = torch.where(rel_y == count, torch.full_like(pos, VISUAL_EOS), y)
        return x, y

    def forward(self, decoder, caption_hidden, lengths, visual_ids):
        """decoder: self.decoder or its compiled version. Returns (loss, accuracy)"""
        x, y = self.layout(visual_ids, lengths)
        drop = None
        if self.training and self.config.condition_dropout > 0:
            drop = torch.rand(x.shape[0], device=x.device) < self.config.condition_dropout
        out = decoder(x, y, media_embeds=self.media(caption_hidden, lengths, drop))
        valid = y >= 0
        return out.loss, (out.logit.argmax(-1)[valid] == y[valid]).float().mean()

    @torch.no_grad()
    def sample(self, caption_hidden, temperature=0.9, top_k=256, cfg_scale=3.0, generator=None):
        """caption_hidden (1, L, thinker_dim) -> VQ tokens (GRID, GRID)"""
        L = caption_hidden.shape[1]
        device = caption_hidden.device
        media = torch.cat((self.media(caption_hidden, torch.tensor([L], device=device)), self.uncond[None]), dim=0)[:, :L]
        x = torch.full((2, L), COND, device=device)
        cache = self.decoder.get_kv_cache(2, L + GRID ** 2)
        logits = self.decoder(x, kv_cache=cache, media_embeds=media).logit[:, -1]
        tokens = []
        for i in range(GRID ** 2):
            guided = logits[1] + cfg_scale * (logits[0] - logits[1])
            guided[VISUAL_EOS] = -float("inf")
            if temperature <= 0:
                token = guided.argmax()
            else:
                values, indices = torch.topk(guided / temperature, top_k)
                token = indices[torch.multinomial(values.softmax(-1), 1, generator=generator)][0]
            tokens.append(token)
            if i + 1 < GRID ** 2:
                logits = self.decoder(token.expand(2, 1), kv_cache=cache).logit[:, -1]
        return torch.stack(tokens).view(GRID, GRID)

    def get_param_groups(self, weight_decay=0.0, matrix_lr=0.02, embedding_lr=0.3, lmhead_lr=0.008):
        groups = self.decoder.get_param_groups(lmhead_lr=lmhead_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
        groups.append(dict(kind="adamw", params=list(self.caption_proj.parameters()) + [self.uncond], lr=3e-3,
                           beta1=0.9, beta2=0.95, wd=0.0, eps=1e-8))
        for g in groups:
            g["init_lr"], g["init_wd"] = g["lr"], g["wd"]
        return groups


def load_vq(model_dir, device):
    """the frozen LlamaGen VQ-16 (MIT): decode(tokens (GRID, GRID)) -> uint8 array (H, W, 3)"""
    spec = importlib.util.spec_from_file_location("llamagen_vq_model", f"{model_dir}/tokenizer/tokenizer_image/vq_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.VQ_models["VQ-16"](codebook_size=CODEBOOK, codebook_embed_dim=8)
    checkpoint = torch.load(f"{model_dir}/vq_ds16_t2i.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("ema") or checkpoint.get("model") or checkpoint["state_dict"])
    model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def decode(tokens):
        image = model.decode_code(tokens.flatten().to(device), (1, 8, GRID, GRID))[0].float().clamp(-1, 1)
        return ((image.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype("uint8")

    return decode
