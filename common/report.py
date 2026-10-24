"""Per-run evaluation results, like nanochat's report.md

Each stage writes <report_dir>/<stage>.json, `python -m script.report --report_dir=...` merges them into one table.
"""
import json
import os
import time


def default_report_dir(ckpt_path):
    return os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "report")


def write_report(report_dir, stage, data):
    os.makedirs(report_dir, exist_ok=True)
    data = {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **data}
    path = os.path.join(report_dir, f"{stage}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def read_report(report_dir, stage):
    path = os.path.join(report_dir, f"{stage}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
