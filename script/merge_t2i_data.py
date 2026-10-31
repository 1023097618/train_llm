"""Image generation data of both sources in one directory: the training rows of script/prepare_t2i_data.py and
script/prepare_t2i_allava.py shuffled together, so that every shard and batch holds both kinds of captions. The last shard,
the validation split, stays prepare_t2i_data.py's, which keeps the validation loss comparable with earlier runs.

usage:
python -m script.merge_t2i_data
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pyarrow as pa
from pyarrow import parquet as pq

from common.data_paths import PROCESSED_ROOT, T2I_DATA

parser = argparse.ArgumentParser(description="merge the image generation data")
parser.add_argument("--allava_dir", type=str, default=str(PROCESSED_ROOT / "t2i_en256_allava"))
parser.add_argument("--out_dir", type=str, default=str(PROCESSED_ROOT / "t2i_en256_all"))
parser.add_argument("--rows_per_shard", type=int, default=10000)
args = parser.parse_args()

english = sorted(glob.glob(os.path.join(T2I_DATA, "en256-*.parquet")))
allava = sorted(glob.glob(os.path.join(args.allava_dir, "en256-*.parquet")))
if not english or not allava:
    raise FileNotFoundError("run script.prepare_t2i_data and script.prepare_t2i_allava first")
if os.path.exists(args.out_dir):
    raise FileExistsError(f"{args.out_dir} exists")
table = pa.concat_tables([pq.read_table(f) for f in english[:-1] + allava]).combine_chunks()
order = np.random.default_rng(0).permutation(table.num_rows)
tmp_dir = args.out_dir + ".tmp"
os.makedirs(tmp_dir)
shards = range(0, table.num_rows, args.rows_per_shard)
for n, start in enumerate(shards):
    pq.write_table(table.take(pa.array(order[start:start + args.rows_per_shard])), f"{tmp_dir}/en256-{n:05d}.parquet", row_group_size=1000)
shutil.copy(english[-1], f"{tmp_dir}/en256-{len(shards):05d}.parquet")
os.replace(tmp_dir, args.out_dir)
print(f"{table.num_rows} training rows in {len(shards)} shards + validation -> {args.out_dir}")
