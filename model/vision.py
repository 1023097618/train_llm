"""Image understanding: a frozen SigLIP2 encoder and a trainable projector into the GPT embedding space
(LLaVA / MiniMind-V / nanoVLM style).

256x256 image -> SigLIP2 base patch16 -> 16x16 patch features (768) -> pixel shuffle 2x2 -> 8x8 = 64 tokens (3072)
-> MLP -> 64 embeddings of the GPT width. In the token sequence an image is <|image|> + 64 x <|image pad|> + <|/image|>
(tokenizer.render_conversation, content part {"type": "image", "length": 64}); GPT.forward swaps the pad embeddings
for these features.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from common.common import Linear

IMAGE_MEAN = 0.5  # SigLIP normalization: (x / 255 - 0.5) / 0.5
IMAGE_STD = 0.5


@dataclass
class VisionConfig:
    encoder_path: str = "data/raw/vlm/siglip2-base-patch16-256"
    image_size: int = 256
    encoder_dim: int = 768
    pixel_shuffle: int = 2
    tokens_per_image: int = 64   # (image_size / 16 / pixel_shuffle) ** 2


def preprocess(images, image_size):
    """list of PIL images -> uint8 tensor (N, 3, H, W); normalization happens on the GPU (normalize())"""
    import numpy as np
    arrays = [np.asarray(im.convert("RGB").resize((image_size, image_size), resample=3)) for im in images]
    if not arrays:
        return torch.zeros((0, 3, image_size, image_size), dtype=torch.uint8)
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def normalize(pixels):
    return (pixels.float() / 255 - IMAGE_MEAN) / IMAGE_STD


class Projector(nn.Module):
    def __init__(self, in_dim, model_dim):
        super().__init__()
        self.fc1 = Linear(in_dim, model_dim, bias=False)
        self.fc2 = Linear(model_dim, model_dim, bias=False)
        nn.init.normal_(self.fc1.weight, std=in_dim ** -0.5)
        nn.init.normal_(self.fc2.weight, std=model_dim ** -0.5)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class VisionTower(nn.Module):
    """frozen encoder (not saved in checkpoints) + trainable projector"""

    def __init__(self, config: VisionConfig, model_dim, device, load_encoder=True):
        super().__init__()
        self.config = config
        self.encoder = None
        if load_encoder:
            from transformers import SiglipVisionModel
            self.encoder = SiglipVisionModel.from_pretrained(config.encoder_path, dtype=torch.bfloat16)
            self.encoder.requires_grad_(False).eval()
        self.projector = Projector(config.encoder_dim * config.pixel_shuffle ** 2, model_dim)
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder is not None:
            self.encoder.eval()  # always frozen
        return self

    def pixel_shuffle(self, x):
        """(N, H*W, C) -> (N, H*W/s^2, C*s^2): merges each s x s block of neighbouring patches into one token"""
        n, t, c = x.shape
        h = w = int(t ** 0.5)
        s = self.config.pixel_shuffle
        x = x.view(n, h // s, s, w // s, s, c).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(n, (h // s) * (w // s), s * s * c)

    def forward(self, pixels):
        """uint8 (N, 3, H, W) -> (N, tokens_per_image, model_dim)"""
        with torch.no_grad():
            feats = self.encoder(pixel_values=normalize(pixels).to(torch.bfloat16)).last_hidden_state
        return self.projector(self.pixel_shuffle(feats.float()))

    def image_embeds(self, x, pixels, model_dim, image_pad_id):
        """dense (B, T, model_dim) tensor for GPT.forward: the i-th image's tokens fill the i-th run of
        <|image pad|> positions in row-major order; None when the batch has no image"""
        if pixels is None or pixels.shape[0] == 0:
            return None
        feats = self(pixels)  # (N, K, D)
        mask = x == image_pad_id
        assert int(mask.sum()) == feats.shape[0] * feats.shape[1], \
            f"{int(mask.sum())} image pad tokens but {feats.shape[0]} images x {feats.shape[1]} tokens"
        out = torch.zeros(*x.shape, model_dim, device=x.device, dtype=feats.dtype)
        out[mask] = feats.reshape(-1, model_dim)
        return out
