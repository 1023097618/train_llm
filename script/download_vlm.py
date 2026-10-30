"""Download the image understanding data, vision encoder and benchmarks into data/raw/vlm.

- jingyaogong/minimind-v_dataset: ALLaVA-4V captions (pretrain_i2t) and instructions (sft_i2t), images already
  resized to 256x256 JPEG inside the parquet files, Chinese and English (prepare_vlm_data.py keeps English)
- google/siglip2-base-patch16-256: frozen vision encoder, 256x256 -> 16x16 patch features
- benchmarks (multiple choice): MMStar, AI2D, ScienceQA-IMG test, MMBench-en dev

usage:
python -m script.download_vlm
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import huggingface_hub

from common.data_paths import VLM_ENCODER, VLM_EVAL, VLM_MINIMIND_V

DOWNLOADS = [
    ("google/siglip2-base-patch16-256", "model", VLM_ENCODER, None),
    ("Lin-Chen/MMStar", "dataset", VLM_EVAL / "MMStar", ["mmstar.parquet"]),
    ("lmms-lab/ai2d", "dataset", VLM_EVAL / "ai2d", ["data/*.parquet"]),
    ("lmms-lab/ScienceQA", "dataset", VLM_EVAL / "ScienceQA", ["ScienceQA-IMG/test-*.parquet"]),
    ("lmms-lab/MMBench", "dataset", VLM_EVAL / "MMBench", ["en/dev-*.parquet"]),
    ("jingyaogong/minimind-v_dataset", "dataset", VLM_MINIMIND_V, ["pretrain_i2t.parquet", "sft_i2t.parquet"]),
]

for repo_id, repo_type, local_dir, patterns in DOWNLOADS:
    for attempt in range(10):
        try:
            huggingface_hub.snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir,
                                              allow_patterns=patterns, max_workers=4)
            break
        except Exception as e:
            print(f"retry {repo_id} ({attempt + 1}/10):", e)
    else:
        raise RuntimeError(f"failed to download {repo_id}")
    print(f"{repo_id} -> {local_dir}")
