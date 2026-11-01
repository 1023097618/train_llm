"""Download an open-source image model for the conversation into data/raw/t2i, diffusers format only.
FLUX.1-schnell is gated: accept its terms on Hugging Face and set HF_TOKEN.

usage:
HF_HUB_DISABLE_XET=1 python -m script.download_image_backend --model=z-image-turbo
HF_HUB_DISABLE_XET=1 HF_TOKEN=... python -m script.download_image_backend --model=flux-schnell
"""
import argparse

import huggingface_hub

from common.data_paths import RAW_T2I
from common.download import large_file, snapshot

REPOS = {"z-image-turbo": "Tongyi-MAI/Z-Image-Turbo", "flux-schnell": "black-forest-labs/FLUX.1-schnell"}

parser = argparse.ArgumentParser(description="download an image model")
parser.add_argument("--model", choices=REPOS, default="z-image-turbo")
parser.add_argument("--connections", type=int, default=16)
args = parser.parse_args()

repo, dst = REPOS[args.model], RAW_T2I / args.model
snapshot(repo, "model", dst, ["*.json", "*.txt", "*.md", "tokenizer*/*"])
for name in huggingface_hub.list_repo_files(repo):
    if name.endswith(".safetensors") and "/" in name:  # files in the root are other formats of the same weights
        path = dst / name
        path.parent.mkdir(parents=True, exist_ok=True)
        large_file(repo, "model", name, path, args.connections)
print(f"done -> {dst}")
