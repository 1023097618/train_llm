"""Samples of the open-source image model with the prompts and seeds of script/imagegen_sample.py, also times a drawing and
reports the GPU memory, to compare with our own decoder.

usage:
PYTHONPATH=~/train_llm_pylibs/image CUDA_VISIBLE_DEVICES=2 python -m script.image_backend_sample
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import os
import time

import torch
from PIL import Image

from inference.image_backend import FluxBackend, ZImageBackend

PROMPTS = ["An orange cat sitting on a stone wall in the sunlight.",
           "A red sports car parked on a mountain road at sunset.",
           "A bowl of fresh fruit on a wooden table.",
           "A snowy mountain above a calm blue lake.",
           "A brown dog running on a sandy beach.",
           "A white sailboat on the sea under a cloudy sky.",
           "A yellow school bus on a city street.",
           "A portrait of an old man with a gray beard."]

parser = argparse.ArgumentParser(description="samples of the open-source image model")
parser.add_argument("--backend", choices=("z-image", "flux"), default="z-image")
parser.add_argument("--out_path", type=str, default="")
parser.add_argument("--seeds", type=int, default=4)
parser.add_argument("--size", type=int, default=768)
parser.add_argument("--cell", type=int, default=256, help="size of one picture on the sheet")
args = parser.parse_args()
args.out_path = args.out_path or f"out/image_backend/{args.backend}.png"

t0 = time.time()
backend = ZImageBackend() if args.backend == "z-image" else FluxBackend()
print(f"loaded in {time.time() - t0:.0f}s, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GB")
sheet = Image.new("RGB", (args.cell * args.seeds, args.cell * len(PROMPTS)), "white")
times = []
for r, prompt in enumerate(PROMPTS):
    for seed in range(args.seeds):
        t0 = time.time()
        image = backend.draw(prompt, seed=seed, size=args.size)
        times.append(time.time() - t0)
        sheet.paste(Image.fromarray(image).resize((args.cell, args.cell)), (seed * args.cell, r * args.cell))
os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
sheet.save(args.out_path)
print(f"{len(times)} pictures of {args.size}px, {sum(times[1:]) / max(1, len(times) - 1):.1f}s each, peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB -> {args.out_path}")
