"""Questions and answers about the pictures the model drew, written by DeepSeek looking at the picture, not at its description: what the
chat model must learn from them is to read the picture. Works through the chunks of script/render_chat_images.py while they appear, about
half of the conversations get one exchange about their last picture; script/render_chat_images.py --merge adds them.

usage:
DEEPSEEK_API_KEY=... python -m datagen.label_chat_images --expected_chunks=56
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import base64
import glob
import json
import os
import random
import re
import time

from pyarrow import parquet as pq

from common.data_paths import VLM_DATA
from datagen.deepseek import is_failed, run_jobs

KINDS = ["a general question about what the picture shows, answered in two or three sentences",
         "a specific question about a color in the picture, answered in one sentence",
         "a specific question about how many of something are visible, answered in one sentence",
         "a specific question about where something is or what is behind it, answered in one sentence",
         "a specific question about what someone or something is doing or wearing, answered in one sentence"]
PROMPT = """This picture was drawn for a user in a chat. Write {kind}, as the user would ask it, and the answer.
Rules:
- The answer comes only from what is clearly visible in this picture. Do not guess at anything you cannot see.
- Ask only about something the picture clearly shows, and speak of "the picture" or "the image".
- The user writes casually in English; the assistant answers in clear, normal English sentences.
Return JSON: {{"question": "...", "answer": "..."}}"""

parser = argparse.ArgumentParser()
parser.add_argument("--expected_chunks", type=int, default=56)
parser.add_argument("--share", type=float, default=0.5)
parser.add_argument("--model", type=str, default="deepseek-v4-flash", help="the DeepSeek model that reads pictures")
parser.add_argument("--workers", type=int, default=16)
args = parser.parse_args()

CHUNKS = VLM_DATA / "chat_chunks"
LABELS = VLM_DATA / "chat_labels"
os.makedirs(LABELS, exist_ok=True)


def validate(_, out):
    question, answer = str(out.get("question") or "").strip(), str(out.get("answer") or "").strip()
    if not question or not 2 <= len(answer.split()) <= 90:
        return None, "length"
    if re.search(r"[一-鿿]", question + answer) or re.search(r"(?i)\b(description|as an ai|language model|i cannot see)\b", answer):
        return None, "content"
    return {"question": question, "answer": answer}, None


def label_chunk(path):
    name = os.path.basename(path).replace(".parquet", "")
    rng = random.Random(name)
    table = pq.read_table(path)
    conversations, images = table.column("conversations").to_pylist(), table.column("image_bytes").to_pylist()
    jobs = []
    for i, pictures in enumerate(images):
        if pictures and rng.random() < args.share:
            jobs.append((f"{name}:{i}", {"image": base64.b64encode(pictures[-1]).decode(), "kind": rng.choice(KINDS)}))

    def make_messages(p):
        return [{"role": "user", "content": [{"type": "text", "text": PROMPT.format(kind=p["kind"])},
                                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + p["image"]}}]}]

    results = run_jobs(jobs, make_messages, validate, str(LABELS / f"{name}.cache.jsonl"), workers=args.workers, temperature=0.5, log_every=10 ** 9, model=args.model)
    good = {job_id.split(":")[1]: r for job_id, r in results.items() if not is_failed(r)}
    with open(LABELS / f"{name}.json.tmp", "w", encoding="utf-8") as f:
        json.dump(good, f, ensure_ascii=False)
    os.replace(LABELS / f"{name}.json.tmp", LABELS / f"{name}.json")
    return len(good), len(jobs)


done, total_good, total = set(), 0, 0
t0 = time.time()
while len(done) < args.expected_chunks:
    new = [f for f in sorted(glob.glob(str(CHUNKS / "chunk-*.parquet"))) if f not in done]
    for path in new:
        good, jobs = label_chunk(path)
        done.add(path)
        total_good, total = total_good + good, total + jobs
        print(f"{os.path.basename(path)}: {good}/{jobs}, total {total_good}/{total}, chunks {len(done)}/{args.expected_chunks}, {time.time() - t0:.0f}s")
    if not new:
        time.sleep(30)
print("labels done")
