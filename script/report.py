"""Merge the evaluation results of every stage into one markdown table (like nanochat's report.md)

usage:
python -m script.report --report_dir=weight/d20/report
"""
import argparse
import os

from common.report import read_report
from tasks.chateval import ALL_TASKS

parser = argparse.ArgumentParser(description="summarize base / sft / rl evaluations")
parser.add_argument("--report_dir", type=str, default="weight/report")
args = parser.parse_args()

base = read_report(args.report_dir, "base") or {}
sft = read_report(args.report_dir, "sft") or {}
rl = read_report(args.report_dir, "rl") or {}
rl_gsm8k = read_report(args.report_dir, "rl_gsm8k") or {}


def fmt(v):
    return "-" if v is None else f"{v:.4f}"


lines = ["| metric | BASE | SFT | RL |", "|---|---|---|---|"]
lines.append(f"| CORE | {fmt(base.get('core', {}).get('core_metric'))} | - | - |")
lines.append(f"| val bpb | {fmt(base.get('val_bpb'))} | - | - |")
for task in ALL_TASKS:
    lines.append(f"| {task} | - | {fmt(sft.get('results', {}).get(task))} | {fmt(rl.get('results', {}).get(task))} |")
lines.append(f"| ChatCORE | - | {fmt(sft.get('chatcore'))} | {fmt(rl.get('chatcore'))} |")

out = ["# Training report", ""]
if base:
    out.append(f"- base model: `{base.get('ckpt')}` (step {base.get('step')}), config {base.get('model_config')}")
out += ["", *lines, ""]
if rl_gsm8k:
    start, last = rl_gsm8k["start"], rl_gsm8k["last"]
    keys = [k for k in start if k.startswith("pass@")]
    out += ["## RL on GSM8K (test, sampled at the RL temperature)", "",
            "| | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys),
            f"| step {start['step']} (SFT) | " + " | ".join(fmt(start[k]) for k in keys) + " |",
            f"| step {last['step']} (RL) | " + " | ".join(fmt(last[k]) for k in keys) + " |", ""]
if base.get("core"):
    out += ["## CORE tasks (centered accuracy)", "", "| task | accuracy | centered |", "|---|---|---|"]
    for task, acc in base["core"]["results"].items():
        out.append(f"| {task} | {acc:.4f} | {base['core']['centered_results'][task]:.4f} |")
    out.append("")
text = "\n".join(out)
print(text)
path = os.path.join(args.report_dir, "report.md")
os.makedirs(args.report_dir, exist_ok=True)
with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"-> {path}")
