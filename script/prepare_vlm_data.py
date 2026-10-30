"""English image-text data for train_vlm.py from MiniMind-V's ALLaVA-4V parquet files (script/download_vlm.py):

- caption:  pretrain_i2t English rows ("describe this image" -> long description), alignment stage
- instruct: sft_i2t English rows that are neither one of those captions (sft_i2t merges the whole caption set)
            nor text only (8x8 black placeholder image; our own text SFT data is replayed instead)
Rows with any CJK character are dropped. Each set is shuffled and written as data/processed/vlm/<set>/<set>-NNNNN.parquet
(conversations json, image_bytes); the last shard is the validation split.

usage:
python -m script.prepare_vlm_data
"""
import argparse
import hashlib
import io
import json
import os
import re

import numpy as np
import pyarrow as pa
from PIL import Image
from pyarrow import parquet as pq

from common.data_paths import VLM_DATA, VLM_MINIMIND_V

parser = argparse.ArgumentParser(description="English ALLaVA subset for VLM training")
parser.add_argument("--rows_per_shard", type=int, default=10000)
parser.add_argument("--val_rows", type=int, default=2000, help="rows in the last (validation) shard of each set")
args = parser.parse_args()

CJK = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")


def english_rows(path, skip_keys=None, drop_text_only=False):
    """yield (conversations json, image_bytes, key) of the English rows"""
    f = pq.ParquetFile(path)
    kept = dropped = 0
    for i in range(f.num_row_groups):
        table = f.read_row_group(i, columns=["conversations", "image_bytes"])
        for conv, image in zip(table.column("conversations").to_pylist(), table.column("image_bytes").to_pylist()):
            key = hashlib.md5(conv.encode()).hexdigest()
            if CJK.search(conv) or "<image>" not in conv or (skip_keys is not None and key in skip_keys):
                dropped += 1
                continue
            if drop_text_only and Image.open(io.BytesIO(image)).size[0] < 32:
                dropped += 1
                continue
            kept += 1
            yield conv, image, key
        if i % 50 == 0:
            print(f"{os.path.basename(path)}: row group {i}/{f.num_row_groups}, kept {kept}, dropped {dropped}")


def write_shards(name, rows):
    out_dir = VLM_DATA / name
    os.makedirs(out_dir, exist_ok=True)
    order = np.random.default_rng(0).permutation(len(rows))
    val, train = order[-args.val_rows:], order[:-args.val_rows]
    shards = [train[i:i + args.rows_per_shard] for i in range(0, len(train), args.rows_per_shard)] + [val]
    for n, idx in enumerate(shards):
        table = pa.table({"conversations": [rows[i][0] for i in idx], "image_bytes": [rows[i][1] for i in idx]})
        pq.write_table(table, out_dir / f"{name}-{n:05d}.parquet", row_group_size=1000)
    print(f"{name}: {len(train)} train + {len(val)} val rows in {len(shards)} shards -> {out_dir}")


captions = list(english_rows(VLM_MINIMIND_V / "pretrain_i2t.parquet"))
caption_keys = {key for _, _, key in captions}
write_shards("caption", captions)
del captions
instruct = list(english_rows(VLM_MINIMIND_V / "sft_i2t.parquet", skip_keys=caption_keys, drop_text_only=True))
write_shards("instruct", instruct)
