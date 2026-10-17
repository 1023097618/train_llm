"""Download pretraining shards of karpathy/fineweb-edu-100b-shuffle into data/processed/pretrain/t2t.

The last shard is held out as the test split (see tasks/pretraindata.py).
usage:
python -m script.download_pretrain                    # 52 shards, enough for d12 (768 dim)
python -m script.download_pretrain --num_shards=180   # for d20 (1280 dim)
"""

import argparse
import os
# 默认走国内镜像；已经设置了 HF_ENDPOINT 时以环境变量为准
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import huggingface_hub
from common.data_paths import PRETRAIN_T2T

parser = argparse.ArgumentParser()
parser.add_argument("--num_shards", default=52, type=int, help="how many shard_*.parquet to download(52 for d12, 180 for d20)")
parser.add_argument("--max_workers", default=4, type=int, help="parallel downloads")
parser.add_argument("--retries", default=50, type=int, help="retry times when the connection breaks")
args = parser.parse_args()

for attempt in range(args.retries):
    try:
        huggingface_hub.snapshot_download(
            repo_id="karpathy/fineweb-edu-100b-shuffle",
            repo_type="dataset",
            local_dir=PRETRAIN_T2T,
            allow_patterns=[f"shard_{i:05d}.parquet" for i in range(args.num_shards)],
            max_workers=args.max_workers,
        )
        break
    except Exception as e:
        print(f"重试 ({attempt + 1}/{args.retries}):", e)
else:
    raise RuntimeError("failed to download pretraining shards")
print(f"{args.num_shards} shards -> {PRETRAIN_T2T}")
