"""English captions of the 256x256 VQ token data (the thinker only understands English; about 3/4 of the captions are
Chinese). Shuffled shards data/processed/t2i_en256/en256-NNNNN.parquet of {caption, vq_token_bytes}, the last is validation.

usage:
python -m script.prepare_t2i_data
"""
import argparse
import glob
import os
import re

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from common.data_paths import T2I_DATA, T2I_RAW_VQ

parser = argparse.ArgumentParser(description="English image generation data")
parser.add_argument("--rows_per_shard", type=int, default=10000)
parser.add_argument("--val_rows", type=int, default=2000)
args = parser.parse_args()

CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
captions, tokens = [], []
total = 0
for path in sorted(glob.glob(str(T2I_RAW_VQ / "*.parquet"))):
    table = pq.read_table(path, columns=["caption", "vq_token_bytes", "vq_width", "vq_height"])
    for caption, vq, w, h in zip(*(table.column(c).to_pylist() for c in table.column_names)):
        total += 1
        if (w, h) == (16, 16) and caption and not CJK.search(caption):
            captions.append(caption)
            tokens.append(vq)
print(f"{len(captions)} English captions of {total} rows")
order = np.random.default_rng(0).permutation(len(captions))
val, train = order[-args.val_rows:], order[:-args.val_rows]
shards = [train[i:i + args.rows_per_shard] for i in range(0, len(train), args.rows_per_shard)] + [val]
os.makedirs(T2I_DATA, exist_ok=True)
for n, idx in enumerate(shards):
    pq.write_table(pa.table({"caption": [captions[i] for i in idx], "vq_token_bytes": [tokens[i] for i in idx]}),
                   T2I_DATA / f"en256-{n:05d}.parquet", row_group_size=2000)
print(f"{len(train)} train + {len(val)} val rows in {len(shards)} shards -> {T2I_DATA}")
