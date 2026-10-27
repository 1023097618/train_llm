"""Download and convert math word problems the SFT model has NOT seen into RL data, data/raw/rl/<name>.jsonl
of {"question", "answer"} (numeric answers only, so the GSM8K style "#### <number>" reward applies):

- omi_gsm: nvidia/OpenMathInstruct-2, problems generated from GSM8K by Llama-3.1-405B ("augmented_gsm8k"),
  same style and difficulty as GSM8K but new questions (the dataset is decontaminated against the test sets)
- math_easy: hendrycks MATH train, levels 1-3, integer \\boxed{} answers (SimpleRL-Zoo's "easy/medium" idea)
GSM8K test questions are dropped from both. Score them with script/rl_filter.py before training.

usage:
python -m script.prepare_rl_data
"""
import argparse
import glob
import json
import os
import re
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import huggingface_hub
from pyarrow import parquet as pq

from common.data_paths import RAW_RL, SFT_GSM8K

parser = argparse.ArgumentParser(description="RL data beyond GSM8K train")
parser.add_argument("--omi_shards", type=int, default=2, help="OpenMathInstruct-2 parquet shards to scan (~240MB each)")
parser.add_argument("--omi_max", type=int, default=30000, help="max augmented_gsm8k questions kept")
parser.add_argument("--math_levels", type=str, default="1,2,3")
args = parser.parse_args()

NUMBER = re.compile(r"-?\d+(\.\d+)?")


def download(repo_id, patterns):
    local = RAW_RL / "cache" / repo_id.split("/")[-1]
    for attempt in range(10):
        try:
            huggingface_hub.snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local,
                                              allow_patterns=patterns, max_workers=4)
            return local
        except Exception as e:
            print(f"retry {repo_id} ({attempt + 1}/10):", e)
    raise RuntimeError(f"failed to download {repo_id}")


def normalize(question):
    return " ".join(question.lower().split())


def clean_number(text):
    text = text.strip().replace(",", "").replace("$", "").rstrip(".")
    if not NUMBER.fullmatch(text):
        return None
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


gsm8k_test = {normalize(r["question"]) for f in glob.glob(str(SFT_GSM8K / "test-*.parquet")) for r in pq.read_table(f).to_pylist()}
assert gsm8k_test, "run python -m script.download_sft first"
os.makedirs(RAW_RL, exist_ok=True)


def write(name, rows):
    path = RAW_RL / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{name}: {len(rows)} questions -> {path}")


# ---- OpenMathInstruct-2 augmented_gsm8k ----
local = download("nvidia/OpenMathInstruct-2", [f"data/train-{i:05d}-of-00032.parquet" for i in range(args.omi_shards)])
seen, rows = set(), []
for f in sorted(glob.glob(str(local / "data" / "train-*.parquet"))):
    table = pq.read_table(f, columns=["problem", "expected_answer", "problem_source"])
    for problem, answer, source in zip(*(table.column(c).to_pylist() for c in ("problem", "expected_answer", "problem_source"))):
        if source != "augmented_gsm8k":
            continue
        key = normalize(problem)
        answer = clean_number(answer or "")
        if answer is None or key in seen or key in gsm8k_test:
            continue
        seen.add(key)
        rows.append({"question": problem.strip(), "answer": answer, "source": "omi_gsm"})
        if len(rows) >= args.omi_max:
            break
    if len(rows) >= args.omi_max:
        break
write("omi_gsm", rows)


# ---- MATH levels 1-3, integer answers ----
def last_boxed(text):
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return None


local = download("EleutherAI/hendrycks_math", ["*/train-*.parquet"])
levels = {f"Level {l}" for l in args.math_levels.split(",")}
rows = []
for f in sorted(glob.glob(str(local / "*" / "train-*.parquet"))):
    for r in pq.read_table(f).to_pylist():
        if r["level"] not in levels:
            continue
        answer = last_boxed(r["solution"])
        answer = answer and answer.replace("\\!", "").replace("{,}", "").strip()
        if answer is None or not re.fullmatch(r"-?\d+", answer.replace(",", "")):
            continue
        rows.append({"question": r["problem"].strip(), "answer": answer.replace(",", ""), "source": "math_easy",
                     "level": r["level"], "type": r["type"]})
write("math_easy", rows)
