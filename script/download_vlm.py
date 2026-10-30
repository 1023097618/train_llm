"""Download the image understanding data, vision encoder and benchmarks into data/raw/vlm.

- jingyaogong/minimind-v_dataset: ALLaVA-4V captions (pretrain_i2t) and instructions (sft_i2t), images already
  resized to 256x256 JPEG inside the parquet files, Chinese and English (prepare_vlm_data.py keeps English)
- google/siglip2-base-patch16-256: frozen vision encoder, 256x256 -> 16x16 patch features
- benchmarks (multiple choice): MMStar, AI2D, ScienceQA-IMG test, MMBench-en dev

Large files are fetched with many parallel range requests (one connection to the mirror is ~1MB/s from the lab),
resumable: finished chunks are listed in <file>.parts. Set HF_TOKEN for faster mirror downloads.

usage:
HF_TOKEN=hf_... python -m script.download_vlm
"""
import argparse

from common.data_paths import VLM_ENCODER, VLM_EVAL, VLM_MINIMIND_V
from common.download import large_file, snapshot

parser = argparse.ArgumentParser(description="download VLM data, encoder and benchmarks")
parser.add_argument("--connections", type=int, default=16)
args = parser.parse_args()

SNAPSHOTS = [  # small files
    ("google/siglip2-base-patch16-256", "model", VLM_ENCODER, ["*.json", "*.model", "*.md"]),
    ("Lin-Chen/MMStar", "dataset", VLM_EVAL / "MMStar", ["mmstar.parquet"]),
    ("lmms-lab/ai2d", "dataset", VLM_EVAL / "ai2d", ["data/*.parquet"]),
    ("lmms-lab/ScienceQA", "dataset", VLM_EVAL / "ScienceQA", ["ScienceQA-IMG/test-*.parquet"]),
    ("lmms-lab/MMBench", "dataset", VLM_EVAL / "MMBench", ["en/dev-*.parquet"]),
]
LARGE = [  # (repo, type, file, local path)
    ("google/siglip2-base-patch16-256", "model", "model.safetensors", VLM_ENCODER / "model.safetensors"),
    ("jingyaogong/minimind-v_dataset", "dataset", "pretrain_i2t.parquet", VLM_MINIMIND_V / "pretrain_i2t.parquet"),
    ("jingyaogong/minimind-v_dataset", "dataset", "sft_i2t.parquet", VLM_MINIMIND_V / "sft_i2t.parquet"),
]
for repo_id, repo_type, local_dir, patterns in SNAPSHOTS:
    snapshot(repo_id, repo_type, local_dir, patterns)
for repo_id, repo_type, filename, path in LARGE:
    large_file(repo_id, repo_type, filename, path, args.connections)
