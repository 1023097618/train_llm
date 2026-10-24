"""Evaluate a base (pretrained) model, like nanochat scripts/base_eval.py

--eval core   : CORE metric (DCLM in-context learning tasks)
--eval bpb    : bits per byte on the held out pretrain shard
--eval sample : greedy completions of a few prompts

usage:
python -m script.download_eval
python -m script.base_eval --ckpt_path=weight/d20/pretrained_model.ckpt --tokenizer_path=weight/d20/tokenizer.pkl
quick check: --max_per_task=100
"""
import argparse

import torch
from torch.utils.data import DataLoader

from common import common
from common.data_paths import PRETRAIN_T2T
from common.report import default_report_dir, write_report
from evaluation import bpbeval
from evaluation.core_eval import evaluate_core
from model import gpt
from tasks import pretraindata
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="evaluate a base model")
parser.add_argument("--ckpt_path", type=str, default="weight/pretrained_model.ckpt")
parser.add_argument("--tokenizer_path", type=str, default="weight/tokenizer.pkl")
parser.add_argument("--eval", type=str, default="core,bpb,sample", help="comma separated: core,bpb,sample")
parser.add_argument("--max_per_task", type=int, default=-1, help="examples per CORE task (-1 = all)")
parser.add_argument("--eval_token", type=int, default=8 * 2**20, help="tokens for the bpb evaluation")
parser.add_argument("--device_batch_size", type=int, default=4)
parser.add_argument("--data_dir", type=str, default=str(PRETRAIN_T2T))
parser.add_argument("--report_dir", type=str, default=None, help="default: <ckpt dir>/report")
parser.add_argument("--stage", type=str, default="base", help="report name")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
modes = {m.strip() for m in args.eval.split(",")}
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
config = gpt.GPTConfig(**ckpt["model_config"])
model = gpt.GPT(config, device, tokeniz)
model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
model.eval()
print(f"model {args.ckpt_path} (step {ckpt.get('step')}): {model.get_model_scale()}")
report = {"ckpt": args.ckpt_path, "step": ckpt.get("step"), "model_config": ckpt["model_config"]}

if "sample" in modes:
    prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ]
    bos = tokeniz.special_token["<|doc|>"]
    samples = []
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([[bos] + tokeniz.encode_ordinary(prompt)], device=device)
            kv = model.get_kv_cache(1)
            nxt = ids
            for _ in range(16):
                nxt = model(nxt, kv_cache=kv).logit[:, -1:].argmax(-1)
                ids = torch.cat([ids, nxt], dim=1)
            samples.append(tokeniz.decode(ids[0, 1:].tolist()))
            print("-" * 60 + "\n" + samples[-1])
    report["samples"] = samples

if "bpb" in modes:
    test_dataset = pretraindata.PretrainDataset(args.data_dir, "test", tokeniz, args.device_batch_size, config.seq_len)
    eval_step = max(1, args.eval_token // (args.device_batch_size * config.seq_len))
    bpb = bpbeval.bpb_eval(model, eval_step, DataLoader(test_dataset, batch_size=None, num_workers=0),
                           tokeniz.get_token_bytes(), device, segment=True)
    print(f"val bpb: {bpb:.4f}")
    report["val_bpb"] = bpb

if "core" in modes:
    core = evaluate_core(model, tokeniz, device, max_per_task=args.max_per_task)
    print(f"CORE metric: {core['core_metric']:.4f}")
    report["core"] = core
    report["max_per_task"] = args.max_per_task

path = write_report(args.report_dir or default_report_dir(args.ckpt_path), args.stage, report)
print(f"report -> {path}")
