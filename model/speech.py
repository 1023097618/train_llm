"""Speech in and out (MiniMind-O / Qwen-Omni "thinker-talker" style).

Input:  frozen SenseVoice features (prepared once by script/prepare_audio_data.py) -> AudioProjector -> GPT embeddings
        at the <|audio pad|> positions, like images (GPT.forward media_embeds).
Output: the Talker, a small Transformer running in lockstep with the GPT ("thinker"): at every position it sees the
        thinker's mid layer state and the previous Mimi codes, and predicts the next code of all 8 codebooks. Codebook j
        is delayed by j positions, so it can use the lower codebooks of the same frame. Mimi decodes the codes to 24 kHz.

Audio tokens: 0..2047 Mimi codes, then the specials below; the input ids of a batch are (B, 9, T): text + 8 codebooks.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from common.common import Embedding, Linear, rms_norm
from model import gpt

NUM_CODEBOOKS = 8
AUDIO_EMB_DIM = 512  # SenseVoice
SPK_EMB_DIM = 192    # CAM++


@dataclass
class TalkerConfig:
    model_dim: int = 768
    query_num: int = 6
    n_layer: int = 4
    seq_len: int = 2048
    audio_vocab_size: int = 2112
    audio_pad_token: int = 2049
    audio_stop_token: int = 2050
    audio_start_token: int = 2051
    audio_spk_token: int = 2052
    adapter_rank: int = 256     # per codebook low rank adapters on the shared embedding / head
    stop_loss_weight: float = 10.0
    text_vocab_size: int = 65536
    text_rank: int = 256        # dimension of the talker's own embedding of the thinker's text tokens


class AudioProjector(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.fc1 = Linear(AUDIO_EMB_DIM, model_dim, bias=False)
        self.fc2 = Linear(model_dim, model_dim, bias=False)
        nn.init.normal_(self.fc1.weight, std=AUDIO_EMB_DIM ** -0.5)
        nn.init.normal_(self.fc2.weight, std=model_dim ** -0.5)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

    def media_embeds(self, x, features, model_dim, audio_pad_id):
        """dense (B, T, D) for GPT.forward: features (N, 512) fill the <|audio pad|> positions in row-major order"""
        if features is None or features.shape[0] == 0:
            return None
        mask = x == audio_pad_id
        assert int(mask.sum()) == features.shape[0], f"{int(mask.sum())} audio pad tokens but {features.shape[0]} features"
        projected = self(features.float())
        out = torch.zeros(*x.shape, model_dim, device=x.device, dtype=projected.dtype)
        out[mask] = projected
        return out


class Talker(nn.Module):
    def __init__(self, config: TalkerConfig, thinker_dim, device):
        super().__init__()
        self.config = config
        c = config
        self.base_emb = Embedding(c.audio_vocab_size, c.model_dim)
        self.emb_adapters = nn.ModuleList([nn.Sequential(Embedding(c.audio_vocab_size, c.adapter_rank), nn.ReLU(),
                                                         Linear(c.adapter_rank, c.model_dim, bias=False))
                                           for _ in range(NUM_CODEBOOKS)])
        self.emb_proj = nn.Sequential(Linear(c.model_dim, c.model_dim, bias=False), nn.ReLU(),
                                      Linear(c.model_dim, c.model_dim, bias=False))
        self.spk_proj = nn.Sequential(Linear(SPK_EMB_DIM, c.model_dim, bias=False), nn.ReLU(),
                                      Linear(c.model_dim, c.model_dim, bias=False))
        self.bridge_proj = nn.Sequential(Linear(thinker_dim, thinker_dim, bias=False), nn.ReLU(),
                                         Linear(thinker_dim, c.model_dim, bias=False))
        # the thinker's state carries semantic, not phonetic, similarity (Qwen2.5-Omni 2.3): the talker also sees the text
        # tokens, through its own embedding since the thinker is frozen
        self.text_emb = Embedding(c.text_vocab_size, c.text_rank)
        self.text_proj = nn.Sequential(Linear(c.text_rank, c.model_dim, bias=False), nn.ReLU(),
                                       Linear(c.model_dim, c.model_dim, bias=False))
        self.text_scale = nn.Parameter(torch.tensor(1.0))
        self.audio_scale = nn.Parameter(torch.tensor(1.0))
        self.bridge_scale = nn.Parameter(torch.tensor(3.0))
        # full context blocks; the value embedding gate of gpt.Block is not used here
        block_config = gpt.GPTConfig(n_layer=c.n_layer, model_dim=c.model_dim, query_num=c.query_num, kv_num=c.query_num,
                                     seq_len=c.seq_len, window_pattern="L")
        cos_sin = gpt.get_cos_sin(c.seq_len, block_config.head_dim)
        self.register_buffer("cos", cos_sin[0], persistent=False)
        self.register_buffer("sin", cos_sin[1], persistent=False)
        self.blocks = nn.ModuleList([gpt.Block(block_config, i, device) for i in range(c.n_layer)])
        for block in self.blocks:
            block.attn.ve_gate = None
        self.block_config = block_config
        self.base_head = Linear(c.model_dim, c.audio_vocab_size, bias=False)
        self.head_adapters = nn.ModuleList([nn.Sequential(Linear(c.model_dim, c.adapter_rank, bias=False), nn.ReLU(),
                                                          Linear(c.adapter_rank, c.audio_vocab_size, bias=False))
                                            for _ in range(NUM_CODEBOOKS)])
        self.init_weight()
        self.to(device)

    @torch.no_grad()
    def init_weight(self):
        c = self.config
        s = (3 / c.model_dim) ** 0.5
        nn.init.normal_(self.base_emb.weight, std=0.6)
        nn.init.normal_(self.text_emb.weight, std=0.6)
        for seq in self.emb_adapters:
            nn.init.normal_(seq[0].weight, std=0.6)
            nn.init.uniform_(seq[2].weight, -(3 / c.adapter_rank) ** 0.5, (3 / c.adapter_rank) ** 0.5)
        for seq in (self.emb_proj, self.spk_proj, self.bridge_proj, self.text_proj):
            for layer in (seq[0], seq[2]):
                bound = (3 / layer.weight.shape[1]) ** 0.5
                nn.init.uniform_(layer.weight, -bound, bound)
        for block in self.blocks:
            for layer in (block.attn.q, block.attn.k, block.attn.v):
                nn.init.uniform_(layer.weight, -s, s)
            nn.init.zeros_(block.attn.o.weight)
            nn.init.uniform_(block.mlp.layer[0].weight, -s * 0.4, s * 0.4)
            nn.init.zeros_(block.mlp.layer[2].weight)
        nn.init.normal_(self.base_head.weight, std=0.001)
        for seq in self.head_adapters:
            bound = (3 / c.model_dim) ** 0.5
            nn.init.uniform_(seq[0].weight, -bound, bound)
            nn.init.normal_(seq[2].weight, std=0.001)

    def embed(self, audio_ids, spk_emb=None):
        """(B, 8, T) -> (B, T, C): mean over the codebooks of shared embedding + adapter"""
        base = self.base_emb(audio_ids).float()
        x = sum(base[:, i] + self.emb_adapters[i](audio_ids[:, i]).float() for i in range(NUM_CODEBOOKS)) / NUM_CODEBOOKS
        if spk_emb is not None and spk_emb.shape[0] > 0:
            # one speaker vector for each position holding the speaker token
            mask = audio_ids[:, 0] == self.config.audio_spk_token
            assert int(mask.sum()) == spk_emb.shape[0], f"{int(mask.sum())} speaker tokens but {spk_emb.shape[0]} vectors"
            x = x.clone()
            x[mask] = self.spk_proj(spk_emb.float())
        return x

    def forward(self, audio_ids, bridge, text_ids, targets=None, segment_ids=None, spk_emb=None, kv_cache=None):
        """audio_ids (B, 8, T), bridge: thinker state (B, T, thinker_dim), text_ids (B, T), targets (B, 8, T) with -1 = ignored.
        Returns logits (B, 8, T, V) and with targets also (loss, per codebook losses)"""
        x = rms_norm(self.emb_proj(self.embed(audio_ids, spk_emb))) * self.audio_scale + \
            rms_norm(self.bridge_proj(bridge.float())) * self.bridge_scale + \
            rms_norm(self.text_proj(self.text_emb(text_ids).float())) * self.text_scale
        x = x.to(torch.bfloat16)
        seg_mask = None
        if segment_ids is not None:
            seg_mask = (segment_ids[:, :, None] == segment_ids[:, None, :]).unsqueeze(1)
        for block in self.blocks:
            x = block(x, None, (self.cos, self.sin), kv_cache=kv_cache, seg_mask=seg_mask)
        if kv_cache is not None:
            kv_cache.T_pre += audio_ids.shape[-1]
        h = rms_norm(x).float()
        base = self.base_head(h)
        logits = torch.stack([base + self.head_adapters[i](h) for i in range(NUM_CODEBOOKS)], dim=1)
        if targets is None:
            return logits
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-1, reduction="none")
        weight = torch.ones_like(ce)
        weight[targets.reshape(-1) == self.config.audio_stop_token] = self.config.stop_loss_weight
        valid = (targets.reshape(-1) != -1).sum().clamp(min=1)
        loss = (ce * weight).sum() / valid
        per_codebook = ce.detach().view(*targets.shape).sum(dim=(0, 2)) / (targets != -1).sum(dim=(0, 2)).clamp(min=1)
        return logits, loss, per_codebook

    def get_kv_cache(self, batch_size, seq_len=None):
        return gpt.KVCache(self.block_config, batch_size, device=self.cos.device, dtype=torch.bfloat16, seq_len=seq_len)

    def get_param_groups(self, adamw_lr=1e-3, matrix_lr=0.02):
        matrices = list(self.blocks.parameters())
        ids = {id(p) for p in matrices}
        others = [p for p in self.parameters() if id(p) not in ids]
        groups = [dict(kind="adamw", params=others, lr=adamw_lr, beta1=0.9, beta2=0.95, wd=0.0, eps=1e-8)]
        for shape in sorted({p.shape for p in matrices}):
            groups.append(dict(kind="muon", params=[p for p in matrices if p.shape == shape], lr=matrix_lr, wd=0.0,
                               eps=1e-6, momentum1=0.95, momentum2=0.9))
        return groups
