import argparse
import json
import os
import torch
from safetensors.torch import save_file
from common import common

parser=argparse.ArgumentParser()
parser.add_argument("--ckpt_path",type=str,default="weight/sft_model.ckpt",help="path of model checkpoint")
parser.add_argument("--out_dir",type=str,default="weight_safetensor",help="output folder (default: <ckpt name> next to the ckpt)")
parser.add_argument("--dtype",type=str,default=None,choices=["fp32","bf16","fp16"],help="cast floating tensors, keep original dtype if not set")
args=parser.parse_args()

out_dir=args.out_dir or os.path.splitext(args.ckpt_path)[0]
os.makedirs(out_dir,exist_ok=True)

resume=torch.load(args.ckpt_path,"cpu",weights_only=False)
state_dict=common.strip_prefix(resume['model_state'])
# cos/sin are rotary buffers rebuilt at init, not real weights
state_dict={k:v for k,v in state_dict.items() if k not in {"cos","sin"}}

dtype={"fp32":torch.float32,"bf16":torch.bfloat16,"fp16":torch.float16}.get(args.dtype)
# safetensors refuses tensors sharing storage, so clone each one
tensors={}
for k,v in state_dict.items():
    if dtype is not None and v.is_floating_point():
        v=v.to(dtype)
    tensors[k]=v.detach().contiguous().clone()

config=resume['model_config']
save_file(tensors,os.path.join(out_dir,"model.safetensors"),metadata={"format":"pt","model_config":json.dumps(config)})
with open(os.path.join(out_dir,"config.json"),"w") as f:
    json.dump(config,f,indent=2)

total=sum(v.numel() for v in tensors.values())
print(f"exported {len(tensors)} tensors ({total/1e6:.1f}M params) -> {out_dir}")
