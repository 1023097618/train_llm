"""Image generation samples: one sheet per checkpoint, a row for each prompt and a column for each seed, to compare
checkpoints by eye. The default prompts include ones train_imagegen.py never samples during training.

usage:
CUDA_VISIBLE_DEVICES=1 python -m script.imagegen_sample --ckpts=weight/d24/imagegen_v2.ckpt --tokenizer_path=weight/d24/tokenizer.pkl
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import os

import torch
from PIL import Image

from common import common
from common.data_paths import T2I_VQ_MODEL
from model import gpt
from model.imagegen import ImageGen, ImageGenConfig, load_vq
from tokenizer import tokenizer

PROMPTS = ["An orange cat sitting on a stone wall in the sunlight.",
           "A red sports car parked on a mountain road at sunset.",
           "A bowl of fresh fruit on a wooden table.",
           "A snowy mountain above a calm blue lake.",
           "A brown dog running on a sandy beach.",
           "A white sailboat on the sea under a cloudy sky.",
           "A yellow school bus on a city street.",
           "A portrait of an old man with a gray beard."]

parser = argparse.ArgumentParser(description="image generation samples")
parser.add_argument("--ckpts", type=str, required=True, help="comma separated")
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--vq_dir", type=str, default=str(T2I_VQ_MODEL))
parser.add_argument("--out_dir", type=str, default="out/imagegen_sample")
parser.add_argument("--prompts", type=str, default="|".join(PROMPTS), help="separated by |")
parser.add_argument("--seeds", type=int, default=4)
parser.add_argument("--cfg_scale", type=float, default=3.0)
parser.add_argument("--temperature", type=float, default=0.9)
args = parser.parse_args()

device = "cuda:0"
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
prompts = args.prompts.split("|")
paths = args.ckpts.split(",")
thinker_path = torch.load(paths[0], map_location="cpu", weights_only=False)["thinker_ckpt"]
ckpt = torch.load(thinker_path, map_location="cpu", weights_only=False)
thinker = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), device, tokeniz)
thinker.load_state_dict(common.strip_prefix(ckpt["model_state"]))
thinker.requires_grad_(False).eval()
del ckpt
last_block = thinker.config.n_layer - 1
decode = load_vq(args.vq_dir, device)
os.makedirs(args.out_dir, exist_ok=True)

with torch.no_grad():
    hiddens = []
    for prompt in prompts:
        ids = [tokeniz.special_token["<|user|>"]] + tokeniz.encode_ordinary(prompt) + [tokeniz.special_token["<|/user|>"]]
        hiddens.append(thinker(torch.tensor([ids], device=device), hidden_layer=last_block, hidden_only=True).hidden)
    for path in paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = ImageGen(ImageGenConfig(**ckpt["config"]), thinker.config.model_dim, device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        sheet = None
        for r, hidden in enumerate(hiddens):
            for seed in range(args.seeds):
                generator = torch.Generator(device=device).manual_seed(seed)
                tokens = model.sample(hidden, temperature=args.temperature, cfg_scale=args.cfg_scale, generator=generator)
                image = Image.fromarray(decode(tokens))
                if sheet is None:
                    sheet = Image.new("RGB", (image.width * args.seeds, image.height * len(prompts)), "white")
                sheet.paste(image, (seed * image.width, r * image.height))
        name = f"{args.out_dir}/{os.path.basename(path)}.png"
        sheet.save(name)
        print(f"step {ckpt['step']} -> {name}")
        del model, ckpt
