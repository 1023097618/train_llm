"""Download what this project published on Hugging Face, into the paths the scripts expect.

- weights: weight/d24/tokenizer.pkl and the final chat model weight/d24/chat_t2i_v9.ckpt (chat, sees pictures, writes and
  edits drawing descriptions); enough for script/chat_t2i_web.py
- stages:  the checkpoints along the way, to start any step of the pipeline without the ones before it:
  pretrained_model (base), rl_omi_model_step210 (text RL), vlm_align -> vlm_sft2 -> vlm_rl (image understanding)
- data:    the data this project made: DeepSeek identity, tool calling and drawing conversations (data/raw/custom), the
  conversations with their pictures drawn by Z-Image-Turbo (data/processed/vlm/chat) and DeepSeek's answers about those
  pictures (data/processed/vlm/chat_labels)

usage:
python -m script.download_release --what=weights,data
"""
import argparse
import os
import shutil

from common.data_paths import RAW_CUSTOM, VLM_DATA
from common.download import snapshot

MODEL_REPO = "zhvhabqifova/guangmingxing-d24"
DATA_REPO = "zhvhabqifova/guangmingxing-d24-data"
WEIGHT_DIR = "weight/d24"
STAGES = ["pretrained_model", "rl_omi_model_step210", "vlm_align", "vlm_sft2", "vlm_rl"]

parser = argparse.ArgumentParser(description="download the published weights and data")
parser.add_argument("--what", type=str, default="weights", help="comma separated: weights, stages, data")
args = parser.parse_args()
what = args.what.split(",")

if "weights" in what:
    snapshot(MODEL_REPO, "model", WEIGHT_DIR, ["tokenizer.pkl", "chat_t2i_v9.ckpt", "README.md"])
if "stages" in what:
    snapshot(MODEL_REPO, "model", WEIGHT_DIR, [f"{name}.ckpt" for name in STAGES])
if "data" in what:
    staging = "data/release"
    snapshot(DATA_REPO, "dataset", staging, ["custom/*", "processed/*", "README.md"])
    os.makedirs(RAW_CUSTOM, exist_ok=True)
    for name in os.listdir(f"{staging}/custom"):
        shutil.copy2(f"{staging}/custom/{name}", RAW_CUSTOM / name)
    for name in ("chat", "chat_labels"):
        dst = VLM_DATA / name
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(f"{staging}/processed/{name}"):
            shutil.copy2(f"{staging}/processed/{name}/{f}", dst / f)
    print(f"data -> {RAW_CUSTOM}, {VLM_DATA / 'chat'}, {VLM_DATA / 'chat_labels'}")
