"""Download Z-Image-Turbo, the open-source model that draws in the conversation, into data/raw/t2i (diffusers format).

usage:
HF_HUB_DISABLE_XET=1 python -m script.download_image_backend
"""
import argparse

import huggingface_hub

from common.data_paths import IMAGE_BACKEND
from common.download import large_file, snapshot

REPO = "Tongyi-MAI/Z-Image-Turbo"

parser = argparse.ArgumentParser(description="download the image model")
parser.add_argument("--connections", type=int, default=16)
args = parser.parse_args()

snapshot(REPO, "model", IMAGE_BACKEND, ["*.json", "*.txt", "*.md", "tokenizer*/*"])
for name in huggingface_hub.list_repo_files(REPO):
    if name.endswith(".safetensors") and "/" in name:  # files in the root are other formats of the same weights
        path = IMAGE_BACKEND / name
        path.parent.mkdir(parents=True, exist_ok=True)
        large_file(REPO, "model", name, path, args.connections)
print(f"done -> {IMAGE_BACKEND}")
