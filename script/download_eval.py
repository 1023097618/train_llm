"""Download evaluation data into data/raw/eval:
- eval_bundle (DCLM CORE tasks, the same bundle nanochat uses)
- allenai/ai2_arc (ARC-Easy / ARC-Challenge test)
- openai/openai_humaneval
MMLU and GSM8K test splits come with script.download_sft.

usage:
python -m script.download_eval
"""

import os
import shutil
import tempfile
import urllib.request
import zipfile
# 默认走国内镜像；已经设置了 HF_ENDPOINT 时以环境变量为准
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import huggingface_hub
from common.data_paths import EVAL_BUNDLE, EVAL_ARC, EVAL_HUMANEVAL

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

DATASETS = [
    ("allenai/ai2_arc", EVAL_ARC, ["ARC-Easy/test-*.parquet", "ARC-Challenge/test-*.parquet"]),
    ("openai/openai_humaneval", EVAL_HUMANEVAL, ["openai_humaneval/test-*.parquet"]),
]

if not os.path.exists(os.path.join(EVAL_BUNDLE, "core.yaml")):
    os.makedirs(os.path.dirname(EVAL_BUNDLE), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = os.path.join(tmp, "eval_bundle.zip")
        print(f"downloading {EVAL_BUNDLE_URL}")
        urllib.request.urlretrieve(EVAL_BUNDLE_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        shutil.rmtree(EVAL_BUNDLE, ignore_errors=True)
        shutil.move(os.path.join(tmp, "eval_bundle"), EVAL_BUNDLE)
print(f"eval_bundle -> {EVAL_BUNDLE}")

for repo_id, local_dir, patterns in DATASETS:
    for attempt in range(10):
        try:
            huggingface_hub.snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir,
                                              allow_patterns=patterns, max_workers=4)
            break
        except Exception as e:
            print(f"重试 {repo_id} ({attempt + 1}/10):", e)
    else:
        raise RuntimeError(f"failed to download {repo_id}")
    print(f"{repo_id} -> {local_dir}")
