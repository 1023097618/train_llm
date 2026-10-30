"""Downloads from the Hugging Face mirror: snapshots of small files, and large files with many parallel range
requests (one connection to the mirror is ~1MB/s from the lab), resumable through <file>.parts.
Set HF_TOKEN for faster mirror downloads."""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import threading
from concurrent.futures import ThreadPoolExecutor

import huggingface_hub
import requests

HEADERS = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}


def parallel_download(url, path, connections=16, chunk_mb=32):
    if os.path.exists(path) and not os.path.exists(str(path) + ".parts"):
        print(f"{path} exists")
        return
    r = requests.get(url, headers={**HEADERS, "Range": "bytes=0-0"}, allow_redirects=True, timeout=60)
    total = int(r.headers["Content-Range"].split("/")[-1])
    chunk = chunk_mb << 20
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

    with ThreadPoolExecutor(connections) as pool:
        list(pool.map(fetch, [c for c in chunks if c[0] not in done]))
    os.close(fd)
    os.replace(tmp, path)
    os.remove(parts_path)
    print(f"{path}: {total / 1e9:.2f} GB done")




def snapshot(repo_id, repo_type, local_dir, patterns):
    for attempt in range(10):
        try:
            huggingface_hub.snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir,
                                              allow_patterns=patterns, max_workers=4)
            print(f"{repo_id} -> {local_dir}")
            return
        except Exception as e:
            print(f"retry {repo_id} ({attempt + 1}/10):", e)
    raise RuntimeError(f"failed to download {repo_id}")


def large_file(repo_id, repo_type, filename, path, connections=16):
    parallel_download(huggingface_hub.hf_hub_url(repo_id, filename, repo_type=repo_type), path, connections)
