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
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import threading
from concurrent.futures import ThreadPoolExecutor

import huggingface_hub
import requests

from common.data_paths import VLM_ENCODER, VLM_EVAL, VLM_MINIMIND_V

parser = argparse.ArgumentParser(description="download VLM data, encoder and benchmarks")
parser.add_argument("--connections", type=int, default=16)
parser.add_argument("--chunk_mb", type=int, default=32)
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
HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}


def parallel_download(url, path):
    if os.path.exists(path) and not os.path.exists(str(path) + ".parts"):
        print(f"{path} exists")
        return
    r = requests.get(url, headers={**HEADERS, "Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
    total = int(r.headers["Content-Range"].split("/")[-1])
    chunk = args.chunk_mb << 20
    chunks = [(s, min(s + chunk, total) - 1) for s in range(0, total, chunk)]
    parts_path = str(path) + ".parts"
    done = set()
    if os.path.exists(parts_path) and os.path.exists(str(path) + ".tmp"):
        with open(parts_path) as f:
            done = {int(line) for line in f if line.strip()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = str(path) + ".tmp"
    if not os.path.exists(tmp):
        with open(tmp, "wb") as f:
            f.truncate(total)
    lock = threading.Lock()
    fd = os.open(tmp, os.O_WRONLY)
    progress = [len(done)]

    def fetch(item):
        start, end = item
        for attempt in range(20):
            try:
                resp = requests.get(url, headers={**HEADERS, "Range": f"bytes={start}-{end}"}, timeout=60, stream=True)
                data = resp.content
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)}")
                os.pwrite(fd, data, start)
                with lock:
                    with open(parts_path, "a") as f:
                        f.write(f"{start}\n")
                    progress[0] += 1
                    if progress[0] % 10 == 0 or progress[0] == len(chunks):
                        print(f"{os.path.basename(path)}: {progress[0]}/{len(chunks)} chunks", flush=True)
                return
            except Exception as e:
                print(f"retry chunk {start} ({attempt + 1}/20): {e}", flush=True)
        raise RuntimeError(f"chunk {start} of {url} failed")

    with ThreadPoolExecutor(args.connections) as pool:
        list(pool.map(fetch, [c for c in chunks if c[0] not in done]))
    os.close(fd)
    os.replace(tmp, path)
    os.remove(parts_path)
    print(f"{path}: {total / 1e9:.2f} GB done")


for repo_id, repo_type, local_dir, patterns in SNAPSHOTS:
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
for repo_id, repo_type, filename, path in LARGE:
    parallel_download(huggingface_hub.hf_hub_url(repo_id, filename, repo_type=repo_type), path)
