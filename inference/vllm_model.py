"""vLLM implementation of model/gpt.py (nanochat architecture)

Register before building an LLM:
    from vllm import ModelRegistry
    from inference.vllm_model import NanoGPTForCausalLM
    ModelRegistry.register_model("NanoGPTForCausalLM", NanoGPTForCausalLM)

Weight names follow model/gpt.py state_dict; q/k/v are fused into qkv_proj when loading.

smear needs the previous token's normalized embedding, but vLLM's forward only sees the current
(flattened) tokens. We get it through vLLM's own KV cache: an extra attention layer with a sliding
window of 2 whose value is the normalized embedding and whose q/k are built from the position parity,
so q·k = +A^2 for the previous token and -A^2 for the current one. softmax then puts ~all weight on the
previous token. This works with chunked prefill, prefix caching and batching for free.
"""
import torch
from torch import nn
from vllm.config import VllmConfig, CacheConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, ColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def norm(x, eps):
    return torch.nn.functional.rms_norm(x, (x.size(-1),), eps=eps)


class NanoGPTForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.transformer = NanoGPTModel(config, vllm_config.cache_config, vllm_config.quant_config, prefix="transformer")
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=vllm_config.quant_config, prefix="lm_head")
        self.logits_processor = LogitsProcessor(config.vocab_size, soft_cap=config.final_logit_softcapping)

    def embed_input_ids(self, input_ids):
        return self.transformer.wte(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None, inputs_embeds: torch.Tensor | None = None):
        return self.transformer(input_ids, positions)

    def compute_logits(self, hidden_states):
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            logits[..., self.config.output_vocab_size:] = -torch.inf  # padded vocab
        return logits

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        stacked = [(".attn.q.", ".attn.qkv_proj.", "q"), (".attn.k.", ".attn.qkv_proj.", "k"), (".attn.v.", ".attn.qkv_proj.", "v")]
        for name, w in weights:
            if name in ("cos", "sin"):
                continue
            for old, new, shard_id in stacked:
                if old in name:
                    pname = name.replace(old, new)
                    p = params[pname]
                    p.weight_loader(p, w, shard_id)
                    break
            else:
                pname = self.rename(name)
                p = params[pname]
                getattr(p, "weight_loader", default_weight_loader)(p, w)
            loaded.add(pname)
        return loaded

    @staticmethod
    def rename(name):
        """model/gpt.py state_dict name -> name in this module"""
        if name.startswith("smear_"):
            return "transformer.smear." + name  # smear_gate.weight / smear_lambda
        if name in ("resid_lambdas", "x0_lambdas", "backout_lambda") or name.startswith("value_embeds."):
            return "transformer." + name
        return name.replace(".attn.o.", ".attn.o_proj.")


class Smear(nn.Module):
    """x[t] += smear_lambda * sigmoid(smear_gate(x[t][:24])) * x[t-1] (position 0 has no previous token)"""
    channels = 24
    amplitude = 4.0  # q·k = ±16 -> the current token gets e^-32 of the weight

    def __init__(self, config, cache_config, quant_config, prefix):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.smear_gate = ReplicatedLinear(self.channels, 1, bias=False, quant_config=quant_config, prefix=f"{prefix}.smear_gate")
        self.smear_lambda = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.attn = Attention(self.num_heads, self.head_dim, scale=1.0, num_kv_heads=self.num_heads,
                              cache_config=cache_config, quant_config=quant_config,
                              per_layer_sliding_window=2,
                              # vLLM orders KV caches by the single integer in the layer name: put it after the blocks
                              prefix=f"{prefix}.{config.num_hidden_layers}.prev_token_attn")

    def forward(self, positions, x):
        parity = 1.0 - 2.0 * (positions % 2).to(x.dtype)  # (-1)^p
        qk = torch.zeros(x.shape[0], self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        k = qk.clone()
        k[:, :, 0] = self.amplitude * parity[:, None]
        qk[:, :, 0] = -self.amplitude * parity[:, None]  # (-1)^(p-1)
        prev = self.attn(qk.view(x.shape[0], -1), k.view(x.shape[0], -1), x)
        gate, _ = self.smear_gate(x[:, :self.channels])
        gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(gate)
        gate = gate * (positions > 0).to(x.dtype)[:, None]
        return x + gate * prev


class NanoGPTModel(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str = ""):
        super().__init__()
        self.config = config
        self.eps = config.rms_norm_eps
        n_layer = config.num_hidden_layers
        kv_dim = config.num_key_value_heads * config.head_dim
        self.wte = VocabParallelEmbedding(config.vocab_size, config.hidden_size, prefix=f"{prefix}.wte")
        self.blocks = nn.ModuleList([NanoGPTBlock(config, i, cache_config, quant_config, prefix=f"{prefix}.blocks.{i}") for i in range(n_layer)])
        self.value_embeds = nn.ModuleDict({str(i): VocabParallelEmbedding(config.vocab_size, kv_dim, prefix=f"value_embeds.{i}")
                                           for i in range(n_layer) if has_ve(i, n_layer)})
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer), requires_grad=False)
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer), requires_grad=False)
        self.backout_lambda = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.smear = Smear(config, cache_config, quant_config, prefix="smear")

    def forward(self, input_ids, positions):
        x = norm(self.wte(input_ids), self.eps)
        x = self.smear(positions, x)
        x0 = x
        backout_layer = self.config.num_hidden_layers // 2
        x_backout = None
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i].to(x.dtype) * x + self.x0_lambdas[i].to(x.dtype) * x0
            ve = self.value_embeds[str(i)](input_ids) if str(i) in self.value_embeds else None
            x = block(positions, x, ve)
            if i == backout_layer:
                x_backout = x
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        return norm(x, self.eps)


class NanoGPTBlock(nn.Module):
    def __init__(self, config, layer_idx, cache_config, quant_config, prefix: str = ""):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.attn = NanoGPTAttention(config, layer_idx, cache_config, quant_config, prefix=f"{prefix}.attn")
        self.mlp = NanoGPTMLP(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(self, positions, x, ve):
        h = norm(x, self.eps)
        x = x + self.attn(positions, h, ve)
        x = x + self.mlp(norm(x, self.eps))
        return x


class NanoGPTAttention(nn.Module):
    ve_gate_channels = 12

    def __init__(self, config, layer_idx, cache_config: CacheConfig | None, quant_config: QuantizationConfig | None, prefix: str = ""):
        super().__init__()
        self.eps = config.rms_norm_eps
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = QKVParallelLinear(config.hidden_size, self.head_dim, self.num_heads, self.num_kv_heads,
                                          bias=False, quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(self.q_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.o_proj")
        self.ve_gate = (ReplicatedLinear(self.ve_gate_channels, self.num_kv_heads, bias=False, quant_config=quant_config, prefix=f"{prefix}.ve_gate")
                        if has_ve(layer_idx, config.num_hidden_layers) else None)
        self.rotary_emb = get_rope(self.head_dim, max_position=config.max_position_embeddings,
                                   is_neox_style=True, rope_parameters=config.rope_parameters)
        # window_size previous tokens + the token itself (model/gpt.py: b-a <= window_size)
        window = config.layer_windows[layer_idx]
        sliding = window + 1 if window < config.max_position_embeddings else None
        self.attn = Attention(self.num_heads, self.head_dim, scale=self.head_dim ** -0.5, num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config, quant_config=quant_config,
                              per_layer_sliding_window=sliding, prefix=f"{prefix}.attn")

    def forward(self, positions, x, ve):
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if ve is not None:
            gate, _ = self.ve_gate(x[:, :self.ve_gate_channels])
            gate = 3 * torch.sigmoid(gate)  # (N, kv_heads)
            v = (v.view(-1, self.num_kv_heads, self.head_dim) + gate.unsqueeze(-1) * ve.view(-1, self.num_kv_heads, self.head_dim)).view(v.shape)
        q, k = self.rotary_emb(positions, q, k)
        q = (norm(q.view(-1, self.num_heads, self.head_dim), self.eps) * 1.2).view(q.shape)
        k = (norm(k.view(-1, self.num_kv_heads, self.head_dim), self.eps) * 1.2).view(k.shape)
        out, _ = self.o_proj(self.attn(q, k, v))
        return out


class ReluSquared(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


class NanoGPTMLP(nn.Module):
    def __init__(self, config, quant_config, prefix=""):
        super().__init__()
        self.layer = nn.ModuleList([
            ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.layer.0"),
            ReluSquared(),
            RowParallelLinear(config.intermediate_size, config.hidden_size, bias=False, quant_config=quant_config, prefix=f"{prefix}.layer.2"),
        ])

    def forward(self, x):
        x, _ = self.layer[0](x)
        x = self.layer[1](x)
        x, _ = self.layer[2](x)
        return x


def hf_config(model_config: dict, output_vocab: int, eos_token_id: int):
    """config.json for vLLM from model/gpt.py GPTConfig fields"""
    from model.gpt import GPTConfig
    cfg = GPTConfig(**model_config)
    return {
        "architectures": ["NanoGPTForCausalLM"],
        "model_type": "llama",
        "hidden_size": cfg.model_dim,
        "intermediate_size": 4 * cfg.model_dim,
        "num_hidden_layers": cfg.n_layer,
        "num_attention_heads": cfg.query_num,
        "num_key_value_heads": cfg.kv_num,
        "head_dim": cfg.head_dim,
        "hidden_act": "relu2",
        "mlp_bias": False,
        "attention_bias": False,
        "rms_norm_eps": float(torch.finfo(torch.bfloat16).eps),
        "max_position_embeddings": cfg.seq_len,
        "rope_parameters": {"rope_type": "default", "rope_theta": 100000.0},
        "rope_theta": 100000.0,
        "vocab_size": cfg.vocab_size,
        "output_vocab_size": output_vocab,
        "final_logit_softcapping": 15.0,
        "layer_windows": [cfg.window_size(i) for i in range(cfg.n_layer)],
        "tie_word_embeddings": False,
        "bos_token_id": None,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
        "torch_dtype": "bfloat16",
        "nanogpt_config": model_config,
    }
