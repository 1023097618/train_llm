"""Score RL questions with a chat model: num_samples completions per question (vLLM, RL sampling settings),
pass rate = fraction correct. Writes <input>.scored.jsonl (all rows + pass_rate) and prints the pass rate
histogram; chat_rl.py --train_data can then use rows whose pass rate is neither 0 nor 1 (written to --out).

usage:
python -m script.rl_filter --ckpt_path=weight/d24/sft_model.ckpt --tokenizer_path=weight/d24/tokenizer.pkl \
    --data=data/raw/rl/omi_gsm.jsonl,data/raw/rl/math_easy.jsonl
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import json
from collections import Counter

import torch

from common import common
from evaluation.sample import calculate
from model import gpt
from tasks.chateval import EvalTask, NumericQA
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="pass rate of RL questions under a chat model")
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--data", type=str, required=True, help="comma separated jsonl files")
parser.add_argument("--max_questions", type=int, default=0, help="per file, 0 = all")
parser.add_argument("--num_samples", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
parser.add_argument("--chunk", type=int, default=2000, help="questions per vLLM call")
args = parser.parse_args()

tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device="cuda:0")
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
model = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), "cuda:0", tokeniz)
model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
del ckpt
from inference.vllm_engine import VLLMEngine
engine = VLLMEngine(model, tokeniz, gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
del model
torch.cuda.empty_cache()
end_assistant = tokeniz.special_token["<|/assistant|>"]

for path in args.data.split(","):
    task = NumericQA(path)
    n = len(task.rows) if args.max_questions <= 0 else min(args.max_questions, len(task.rows))
    rows = []
    for c0 in range(0, n, args.chunk):
        idx = range(c0, min(n, c0 + args.chunk))
        conversations = [task.get(i) for i in idx]
        prompts = [tokeniz.render_conversation({"source": "rl", "messages": EvalTask.prompt_messages(c)}, last_is_user=True)[0]
                   for c in conversations]
        outputs = engine.generate(prompts, num_samples=args.num_samples, max_tokens=args.max_new_tokens,
                                  temperature=args.temperature, top_k=args.top_k, tools={"calculate": calculate}, seed=c0)
        for i, c, samples in zip(idx, conversations, outputs):
            texts = [tokeniz.decode([t for t in toks if t != end_assistant]) for toks, _ in samples]
            rows.append({**task.rows[i], "pass_rate": sum(task.evaluate(c, t) for t in texts) / len(texts)})
        print(f"{path}: scored {len(rows)}/{n}")
    out = path.replace(".jsonl", ".scored.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = Counter(round(r["pass_rate"] * args.num_samples) for r in rows)
    mean = sum(r["pass_rate"] for r in rows) / len(rows)
    useful = sum(0 < r["pass_rate"] < 1 for r in rows)
    print(f"{path}: mean pass rate {mean:.3f}, useful (0 < pass < 1) {useful}/{len(rows)} = {useful / len(rows):.2f}")
    print("  correct out of %d: " % args.num_samples + ", ".join(f"{k}: {hist.get(k, 0)}" for k in range(args.num_samples + 1)))
    print(f"  -> {out}")
