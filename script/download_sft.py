"""Download nanochat's SFT datasets (SmolTalk, MMLU, GSM8K) and hermes function-calling data into data/raw/sft.

usage:
python -m script.download_sft
"""

import os
# 默认走国内镜像；已经设置了 HF_ENDPOINT 时以环境变量为准
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import huggingface_hub
from common.data_paths import RAW_SFT

DATASETS = [
    ("HuggingFaceTB/smol-smoltalk", "smol-smoltalk", ["data/train-*.parquet", "data/test-*.parquet"]),
    ("cais/mmlu", "mmlu", ["all/auxiliary_train-*.parquet", "all/test-*.parquet"]),
    ("openai/gsm8k", "gsm8k", ["main/train-*.parquet", "main/test-*.parquet"]),
    ("NousResearch/hermes-function-calling-v1", "hermes-function-calling-v1",
     ["func-calling.json", "glaive-function-calling-5k.json"]),
]

for repo_id, local_name, patterns in DATASETS:
    for attempt in range(10):
        try:
            huggingface_hub.snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=RAW_SFT / local_name,
                allow_patterns=patterns,
                max_workers=4,
            )
            break
        except Exception as e:
            print(f"重试 {repo_id} ({attempt + 1}/10):", e)
    else:
        raise RuntimeError(f"failed to download {repo_id}")
    print(f"{repo_id} -> {RAW_SFT / local_name}")
