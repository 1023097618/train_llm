import os
import pathlib
import sys
import tqdm
import torch
from torch.nn import functional as F
def get_curent_path():
    if __file__ in globals():
        return os.path.dirname(__file__)
    else:
        return pathlib.Path.cwd()

def get_pbar():
    return tqdm.tqdm(
        total=100.,
        bar_format='{l_bar}{bar}| {n:.3f}/{total_fmt} [{elapsed}<{remaining},  {rate_fmt}{postfix}]',
        # `2>&1 | tee run.log` makes stderr a pipe. Disable the repeatedly
        # refreshed bar in that case so carriage-return updates do not bloat
        # the log; regular messages written through pbar.write are preserved.
        disable=not sys.stderr.isatty(),
    )

def to_cpu(obj):
    """recursively move tensors to cpu, e.g. an optimizer state before dist.gather_object
    (otherwise rank0 unpickles every rank's GPU tensors onto its own GPU)"""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj,dict):
        return {k:to_cpu(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj

def strip_prefix(sd,prefix='_orig_mod.'):
    return { (k[len(prefix):] if k.startswith(prefix) else k):v for k,v in sd.items() }

class Linear(torch.nn.Linear):
    def forward(self,x):
        return torch.nn.functional.linear(x,self.weight.to(dtype=x.dtype))

class Conv2d(torch.nn.Conv2d):
    def forward(self, x):
        return F.conv2d(
            x,
            self.weight.to(dtype=x.dtype),
            None if self.bias is None else self.bias.to(dtype=x.dtype),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Embedding(torch.nn.Embedding):
    def forward(self, x):
        return super().forward(x).to(torch.bfloat16)

def rms_norm(x):
    # https://docs.pytorch.org/docs/2.12/generated/torch.nn.RMSNorm.html
    # no learnable parameter (no γ)
    # call this function before x entering attention and mlp
    return F.rms_norm(x,(x.size(-1),))

# https://github.com/karpathy/nanochat/blob/master/nanochat/common.py#L228
# 对于DDP训练来说，多个显卡型号需要是同一款，否则会有木桶效应
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
        (["5060"], 38.35e12)
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    print(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')