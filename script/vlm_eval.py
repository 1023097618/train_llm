"""Image understanding benchmarks (multiple choice): MMStar, AI2D, ScienceQA-IMG test, MMBench-en dev.

Like chat_eval's categorical tasks: the prompt is the image + question + lettered options, the prediction is the
argmax over the option letters' logits right after <|assistant|>. --no_image=1 drops the image from the prompt
(the "blind" score shows how much the answer really comes from the image).

usage:
python -m script.vlm_eval --ckpt_path=weight/d24/vlm_sft.ckpt --tokenizer_path=weight/d24/tokenizer.pkl --stage=vlm_sft
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import ast
import glob
import io
import os
import re

import torch
from PIL import Image
from pyarrow import parquet as pq

from common import common
from common.data_paths import VLM_ENCODER, VLM_EVAL
from common.report import default_report_dir, write_report
from model import gpt
from model.vision import VisionConfig, VisionTower, preprocess
from tasks.sftdata import render_mc
from tokenizer import tokenizer

LETTERS = "ABCDEFGH"


def _rows(pattern):
    files = sorted(glob.glob(str(VLM_EVAL / pattern)))
    if not files:
        raise FileNotFoundError(f"{pattern} not found, run `python -m script.download_vlm` first")
    for f in files:
        yield from pq.read_table(f).to_pylist()


def _image(value):
    return value["bytes"] if isinstance(value, dict) else value


def _listish(value):
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


def mmstar():
    """two layouts: '...\nOptions: A: x, B: y, ...' and 'Hint: ...\nQuestion: ...\nChoices:\n(A) x\n(B) y ...'"""
    for r in _rows("MMStar/mmstar.parquet"):
        text = r["question"]
        if "\nChoices:" in text:
            question, _, options = text.partition("\nChoices:")
            question = question.split("Question:", 1)[-1]
            choices = re.split(r"\n?\([A-H]\)\s*", options.strip())[1:]
        else:
            question, _, options = text.partition("\nOptions:")
            choices = re.split(r",?\s*[A-D]:\s*", " " + options)[1:]
        if len(choices) < 2:
            raise ValueError(f"cannot parse the options of MMStar question {r['index']}: {text!r}")
        letters = LETTERS[:len(choices)]
        yield _image(r["image"]), render_mc(question.strip(), letters, [c.strip() for c in choices]), letters, letters.index(r["answer"].strip())


def ai2d():
    for r in _rows("ai2d/data/test-*.parquet"):
        choices = _listish(r["options"])
        letters = LETTERS[:len(choices)]
        yield _image(r["image"]), render_mc(r["question"], letters, choices), letters, int(r["answer"])


def scienceqa():
    for r in _rows("ScienceQA/ScienceQA-IMG/test-*.parquet"):
        choices = _listish(r["choices"])
        letters = LETTERS[:len(choices)]
        question = f"{r['hint']}\n{r['question']}" if r.get("hint") else r["question"]
        yield _image(r["image"]), render_mc(question, letters, choices), letters, int(r["answer"])


def mmbench():
    for r in _rows("MMBench/en/dev-*.parquet"):
        choices = [r[c] for c in "ABCD" if r.get(c) not in (None, "nan", "")]
        letters = LETTERS[:len(choices)]
        hint = r.get("hint")
        question = f"{hint}\n{r['question']}" if hint and hint != "nan" else r["question"]
        yield _image(r["image"]), render_mc(question, letters, choices), letters, letters.index(r["answer"].strip())


TASKS = {"MMStar": mmstar, "AI2D": ai2d, "ScienceQA-IMG": scienceqa, "MMBench": mmbench}


@torch.no_grad()
def run_task(model, vision, tokeniz, items, device, tokens_per_image, image_size, no_image, batch_size, max_problems):
    """returns accuracy and the chance accuracy of the evaluated questions"""
    image_pad = tokeniz.special_token["<|image pad|>"]
    pad = tokeniz.special_token["<|/assistant|>"]
    correct = total = 0
    chance = 0.0
    batch = []

    def flush():
        nonlocal correct, total, chance
        prompts, letter_ids, golds, images = zip(*batch)
        longest = max(len(p) for p in prompts)
        ids = torch.tensor([p + [pad] * (longest - len(p)) for p in prompts], device=device)
        last = torch.tensor([len(p) - 1 for p in prompts], device=device)
        pixels = None if no_image else preprocess([Image.open(io.BytesIO(b)) for b in images], image_size).to(device)
        embeds = None if no_image else vision.image_embeds(ids, pixels, model.config.model_dim, image_pad)
        logits = model(ids, image_embeds=embeds).logit[torch.arange(len(prompts), device=device), last]
        for row, letters, gold in zip(logits, letter_ids, golds):
            pred = int(torch.tensor([row[l] for l in letters]).argmax())
            correct += pred == gold
            total += 1
            chance += 1 / len(letters)
        batch.clear()

    for i, (image, text, letters, gold) in enumerate(items):
        if max_problems and i >= max_problems:
            break
        content = [{"type": "text", "content": text}] if no_image else \
            [{"type": "image", "length": tokens_per_image}, {"type": "text", "content": text}]
        tokens, _ = tokeniz.render_conversation({"source": "vlm_eval", "messages": [{"role": "user", "content": content}]},
                                                last_is_user=True)
        letter_ids = [tokeniz.encode_ordinary(l)[0] for l in letters]
        batch.append((tokens, letter_ids, gold, image))
        if len(batch) == batch_size:
            flush()
    if batch:
        flush()
    return correct / max(total, 1), chance / max(total, 1), total


def main():
    parser = argparse.ArgumentParser(description="image understanding benchmarks")
    parser.add_argument("--ckpt_path", type=str, required=True, help="VLM checkpoint (train_vlm.py)")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--encoder_path", type=str, default=str(VLM_ENCODER))
    parser.add_argument("--tasks", type=str, default=",".join(TASKS))
    parser.add_argument("--max_problems", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_image", type=int, default=0, help="1 = text only prompts (blind baseline)")
    parser.add_argument("--stage", type=str, default="vlm", help="report name")
    parser.add_argument("--report_dir", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0"
    tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), device, tokeniz)
    model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
    model.eval()
    vision_config = VisionConfig(**{**ckpt.get("vision_config", {}), "encoder_path": args.encoder_path})
    vision = VisionTower(vision_config, model.config.model_dim, device, load_encoder=not args.no_image)
    if "projector_state" in ckpt:
        vision.projector.load_state_dict(ckpt["projector_state"])
    elif not args.no_image:
        raise ValueError(f"{args.ckpt_path} has no projector, evaluate it with --no_image=1")
    vision.eval()
    del ckpt

    results, centered = {}, {}
    for name in args.tasks.split(","):
        acc, chance, n = run_task(model, vision, tokeniz, TASKS[name](), device, vision_config.tokens_per_image,
                                  vision_config.image_size, args.no_image, args.batch_size, args.max_problems)
        results[name] = acc
        centered[name] = (acc - chance) / (1 - chance)
        print(f"{name:<14} accuracy {acc:.4f} (chance {chance:.3f}, centered {centered[name]:.4f}, {n} questions)")
    mean_centered = sum(centered.values()) / len(centered)
    print(f"mean centered accuracy: {mean_centered:.4f}")
    report_dir = args.report_dir or default_report_dir(args.ckpt_path)
    path = write_report(report_dir, args.stage, {"ckpt": args.ckpt_path, "no_image": bool(args.no_image),
                                                  "results": results, "centered": centered, "mean_centered": mean_centered})
    print(f"-> {path}")


if __name__ == "__main__":
    main()
