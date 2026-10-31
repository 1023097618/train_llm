"""More English image generation data from the ALLaVA captions already on disk (data/processed/vlm/caption, 256x256 JPEG):
encode each image with the VQ-16 and keep the first sentences of its long description as a prompt-like caption.
Written as data/processed/t2i_en256_allava/en256-NNNNN.parquet, finished shards are skipped. Train with
--data_dir on a merged directory (the last shard, the validation split, stays script/prepare_t2i_data.py's).

usage:
CUDA_VISIBLE_DEVICES=1 python -m script.prepare_t2i_allava
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import glob
import io
import json
import os
import re

import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from pyarrow import parquet as pq
from torch.utils.data import DataLoader, Dataset

from common.data_paths import PROCESSED_ROOT, T2I_VQ_MODEL, VLM_DATA
from model.imagegen import load_vq

parser = argparse.ArgumentParser(description="VQ tokens of the ALLaVA images")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=12)
parser.add_argument("--max_chars", type=int, default=300, help="caption length limit")
parser.add_argument("--max_shards", type=int, default=0)
args = parser.parse_args()

OUT = PROCESSED_ROOT / "t2i_en256_allava"
os.makedirs(OUT, exist_ok=True)
SENTENCE = re.compile(r"(?<=[.!?])\s+")


def prompt_of(description):
    """first sentences up to max_chars; drops the 'The image shows' habit only in that it stays, the data is what it is"""
    out = ""
    for sentence in SENTENCE.split(description.strip()):
        if out and len(out) + len(sentence) + 1 > args.max_chars:
            break
        out = (out + " " + sentence).strip()
        if len(out) >= 80:
            break
    return out[:args.max_chars]


class Rows(Dataset):
    def __init__(self, conversations, images):
        self.conversations, self.images = conversations, images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        reply = json.loads(self.conversations[i])[1]["content"]
        image = np.asarray(Image.open(io.BytesIO(self.images[i])).convert("RGB").resize((256, 256), resample=3))
        return prompt_of(reply), torch.from_numpy(image.copy()).permute(2, 0, 1)


vq = load_vq(T2I_VQ_MODEL, "cuda:0")
files = sorted(glob.glob(str(VLM_DATA / "caption" / "caption-*.parquet")))[:-1]  # the last is VLM validation
if args.max_shards:
    files = files[:args.max_shards]
for f in files:
    n = re.search(r"-(\d+)\.parquet", f).group(1)
    path = OUT / f"en256-{n}.parquet"
    if path.exists():
        continue
    table = pq.read_table(f)
    loader = DataLoader(Rows(table.column("conversations").to_pylist(), table.column("image_bytes").to_pylist()),
                        batch_size=args.batch_size, num_workers=args.num_workers, prefetch_factor=4)
    captions, tokens = [], []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_captions, pixels in loader:
            ids = vq.encode(pixels).cpu().numpy().astype(np.uint16)
            captions += list(batch_captions)
            tokens += [row.tobytes() for row in ids]
    pq.write_table(pa.table({"caption": captions, "vq_token_bytes": tokens}), str(path) + ".tmp", row_group_size=2000)
    os.replace(str(path) + ".tmp", path)
    print(f"{path}: {len(captions)} rows", flush=True)
print("done")
