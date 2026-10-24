"""Evaluate a chat model (SFT / RL), like nanochat scripts/chat_eval.py

ARC-Easy, ARC-Challenge, MMLU: the answer letter is read from the logits right after <|assistant|>
GSM8K (calculator tool), HumanEval: sampled with vLLM
ChatCORE = mean accuracy centered by the random baseline (needs all 5 tasks)

usage:
python -m script.chat_eval --ckpt_path=weight/d20/sft_model.ckpt --tokenizer_path=weight/d20/tokenizer.pkl --stage=sft
quick check: --max_problems=200
"""
import argparse
import gc
import time

import torch

from common import common
from common.report import default_report_dir, write_report
from evaluation.sample import calculate
from model import gpt
from tasks import chateval
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="evaluate a chat model")
parser.add_argument("--ckpt_path", type=str, default="weight/sft_model.ckpt")
parser.add_argument("--tokenizer_path", type=str, default="weight/tokenizer.pkl")
parser.add_argument("--tasks", type=str, default="|".join(chateval.ALL_TASKS), help="'|' separated task names")
parser.add_argument("--max_problems", type=int, default=-1, help="max problems per task (-1 = all)")
parser.add_argument("--batch_size", type=int, default=8, help="batch size of the categorical evaluation")
parser.add_argument("--num_samples", type=int, default=1, help="samples per generative problem, pass@num_samples")
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.6, help="vLLM memory fraction for the generative tasks")
parser.add_argument("--enforce_eager", type=int, default=0, help="disable vLLM cuda graphs (faster startup, slower decoding)")
parser.add_argument("--report_dir", type=str, default=None, help="default: <ckpt dir>/report")
parser.add_argument("--stage", type=str, default="sft", help="report name, e.g. sft / rl")
args = parser.parse_args()

device = "cuda"
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
model = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), device, tokeniz)
model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
model.eval()
task_names = args.tasks.split("|")
max_problems = None if args.max_problems < 0 else args.max_problems


def render_prompt(conversation):
    messages = chateval.EvalTask.prompt_messages(conversation)
    tokens, _ = tokeniz.render_conversation({"source": conversation.get("source", "eval"), "messages": messages}, last_is_user=True)
    return tokens


@torch.no_grad()
def run_categorical(task):
    n = len(task) if max_problems is None else min(len(task), max_problems)
    pad = tokeniz.special_token["<|/doc|>"]
    letter_ids = {}
    passed = 0
    for i0 in range(0, n, args.batch_size):
        conversations = [task[i] for i in range(i0, min(n, i0 + args.batch_size))]
        prompts = [render_prompt(c) for c in conversations]
        longest = max(len(p) for p in prompts)
        ids = torch.tensor([p + [pad] * (longest - len(p)) for p in prompts], device=device)
        logits = model(ids).logit
        for j, c in enumerate(conversations):
            for letter in c["letters"]:
                if letter not in letter_ids:
                    encoded = tokeniz.encode_ordinary(letter)
                    assert len(encoded) == 1, f"answer letter {letter!r} must be a single token"
                    letter_ids[letter] = encoded[0]
            focus = logits[j, len(prompts[j]) - 1, [letter_ids[l] for l in c["letters"]]]
            passed += int(task.evaluate(c, c["letters"][focus.argmax().item()]))
    return passed / n, n


def run_generative(task, engine):
    n = len(task) if max_problems is None else min(len(task), max_problems)
    conversations = [task[i] for i in range(n)]
    tools = {"calculate": calculate} if task.name == "GSM8K" else None
    end_assistant = tokeniz.special_token["<|/assistant|>"]
    outputs = engine.generate([render_prompt(c) for c in conversations], num_samples=args.num_samples,
                              max_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, tools=tools, seed=0)
    passed = 0
    for c, samples in zip(conversations, outputs):
        texts = [tokeniz.decode([t for t in tokens if t != end_assistant]) for tokens, _ in samples]
        passed += int(any(task.evaluate(c, text) for text in texts))
    return passed / n, n


results, counts = {}, {}
categorical = [t for t in task_names if chateval.EVAL_TYPE[t] == "categorical"]
generative = [t for t in task_names if t not in categorical]
for name in categorical:
    t0 = time.time()
    results[name], counts[name] = run_categorical(chateval.build_task(name))
    print(f"{name}: {100 * results[name]:.2f}% ({counts[name]} problems, {time.time() - t0:.0f}s)")
if generative:
    from inference.vllm_engine import VLLMEngine
    engine = VLLMEngine(model, tokeniz, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=bool(args.enforce_eager))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for name in generative:
        t0 = time.time()
        results[name], counts[name] = run_generative(chateval.build_task(name), engine)
        print(f"{name}: {100 * results[name]:.2f}% ({counts[name]} problems, {time.time() - t0:.0f}s)")

report = {"ckpt": args.ckpt_path, "results": results, "num_problems": counts,
          "num_samples": args.num_samples, "temperature": args.temperature}
if all(t in results for t in chateval.ALL_TASKS):
    centered = [(results[t] - chateval.BASELINE[t]) / (1 - chateval.BASELINE[t]) for t in chateval.ALL_TASKS]
    report["chatcore"] = sum(centered) / len(centered)
    print(f"ChatCORE metric: {report['chatcore']:.4f}")
path = write_report(args.report_dir or default_report_dir(args.ckpt_path), args.stage, report)
print(f"report -> {path}")
