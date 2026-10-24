"""Evaluate a chat model (SFT / RL), like nanochat scripts/chat_eval.py

ARC-Easy, ARC-Challenge, MMLU: the answer letter is read from the logits right after <|assistant|>
GSM8K (calculator tool), HumanEval: sampled with vLLM
ChatCORE = mean accuracy centered by the random baseline (needs all 5 tasks)

usage:
python -m script.chat_eval --ckpt_path=weight/d20/sft_model.ckpt --tokenizer_path=weight/d20/tokenizer.pkl --stage=sft
quick check: --max_problems=200
"""
import sys
sys.stdout.reconfigure(line_buffering=True)  # show progress in nohup logs immediately
import argparse
import gc
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader

from common import common
from common.report import default_report_dir, write_report
from evaluation.sample import calculate
from model import gpt
from tasks import chateval
from tokenizer import tokenizer


def render_prompt(tokeniz, conversation):
    messages = chateval.EvalTask.prompt_messages(conversation)
    tokens, _ = tokeniz.render_conversation({"source": conversation.get("source", "eval"), "messages": messages}, last_is_user=True)
    return tokens


class CategoricalPrompts(Dataset):
    """renders the prompt and the token id of every answer letter in DataLoader workers"""
    def __init__(self, task, tokeniz, n):
        self.task, self.tokeniz, self.n = task, tokeniz, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        c = self.task[i]
        letter_ids = []
        for letter in c["letters"]:
            encoded = self.tokeniz.encode_ordinary(letter)
            assert len(encoded) == 1, f"answer letter {letter!r} must be a single token"
            letter_ids.append(encoded[0])
        return render_prompt(self.tokeniz, c), letter_ids, c["letters"].index(c["messages"][-1]["content"])


def collate_categorical(items, pad):
    longest = max(len(p) for p, _, _ in items)
    most_letters = max(len(l) for _, l, _ in items)
    ids = torch.tensor([p + [pad] * (longest - len(p)) for p, _, _ in items])
    last = torch.tensor([len(p) - 1 for p, _, _ in items])
    letters = torch.tensor([l + [-1] * (most_letters - len(l)) for _, l, _ in items])  # -1: no such option
    gold = torch.tensor([g for _, _, g in items])
    return ids, last, letters, gold


@torch.no_grad()
def run_categorical(model, tokeniz, task, device, max_problems=None, batch_size=32, batch_tokens=16384, num_workers=4):
    """accuracy of the argmax over the answer letters' logits right after <|assistant|>"""
    n = len(task) if max_problems is None else min(len(task), max_problems)
    pad = tokeniz.special_token["<|/doc|>"]
    loader = DataLoader(CategoricalPrompts(task, tokeniz, n), batch_size=batch_size, num_workers=num_workers,
                        collate_fn=lambda items: collate_categorical(items, pad),
                        pin_memory=str(device).startswith("cuda"), prefetch_factor=4 if num_workers > 0 else None)
    passed = 0
    for ids, last, letters, gold in loader:
        ids, last, letters, gold = (t.to(device, non_blocking=True) for t in (ids, last, letters, gold))
        rows = max(1, batch_tokens // ids.shape[1])  # bound the (rows, T, vocab) logits
        preds = []
        for r0 in range(0, ids.shape[0], rows):
            logits = model(ids[r0:r0 + rows]).logit
            final = logits[torch.arange(logits.shape[0], device=device), last[r0:r0 + rows]]  # (rows, vocab)
            letter_logits = final.gather(1, letters[r0:r0 + rows].clamp(min=0))
            letter_logits[letters[r0:r0 + rows] < 0] = -float("inf")
            preds.append(letter_logits.argmax(1))
        passed += (torch.cat(preds) == gold).sum().item()  # one host sync per batch
    return passed / n, n


def run_generative(engine, tokeniz, task, max_problems=None, num_samples=1, max_new_tokens=512, temperature=0.0, top_k=50, exec_workers=16):
    n = len(task) if max_problems is None else min(len(task), max_problems)
    conversations = [task[i] for i in range(n)]
    tools = {"calculate": calculate} if task.name == "GSM8K" else None
    end_assistant = tokeniz.special_token["<|/assistant|>"]
    outputs = engine.generate([render_prompt(tokeniz, c) for c in conversations], num_samples=num_samples,
                              max_tokens=max_new_tokens, temperature=temperature, top_k=top_k, tools=tools, seed=0)
    jobs = [(c, tokeniz.decode([t for t in tokens if t != end_assistant])) for c, samples in zip(conversations, outputs) for tokens, _ in samples]
    # HumanEval runs every completion in its own subprocess, so the checks can run in parallel
    with ThreadPoolExecutor(exec_workers) as pool:
        outcomes = list(pool.map(lambda job: task.evaluate(*job), jobs))
    passed = sum(any(outcomes[k * num_samples:(k + 1) * num_samples]) for k in range(n))
    return passed / n, n


def main():
    parser = argparse.ArgumentParser(description="evaluate a chat model")
    parser.add_argument("--ckpt_path", type=str, default="weight/sft_model.ckpt")
    parser.add_argument("--tokenizer_path", type=str, default="weight/tokenizer.pkl")
    parser.add_argument("--tasks", type=str, default="|".join(chateval.ALL_TASKS), help="'|' separated task names")
    parser.add_argument("--max_problems", type=int, default=-1, help="max problems per task (-1 = all)")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of the categorical evaluation")
    parser.add_argument("--batch_tokens", type=int, default=16384, help="max tokens per categorical forward pass (lower it on small GPUs)")
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

    results, counts = {}, {}
    categorical = [t for t in task_names if chateval.EVAL_TYPE[t] == "categorical"]
    generative = [t for t in task_names if t not in categorical]
    for name in categorical:
        t0 = time.time()
        results[name], counts[name] = run_categorical(model, tokeniz, chateval.build_task(name), device, max_problems,
                                                      args.batch_size, args.batch_tokens)
        print(f"{name}: {100 * results[name]:.2f}% ({counts[name]} problems, {time.time() - t0:.0f}s)")
    if generative:
        from inference.vllm_engine import VLLMEngine
        engine = VLLMEngine(model, tokeniz, gpu_memory_utilization=args.gpu_memory_utilization, enforce_eager=bool(args.enforce_eager))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        for name in generative:
            t0 = time.time()
            results[name], counts[name] = run_generative(engine, tokeniz, chateval.build_task(name), max_problems, args.num_samples,
                                                         args.max_new_tokens, args.temperature, args.top_k)
            print(f"{name}: {100 * results[name]:.2f}% ({counts[name]} problems, {time.time() - t0:.0f}s)")

    report = {"ckpt": args.ckpt_path, "results": results, "num_problems": counts,
              "num_samples": args.num_samples, "temperature": args.temperature}
    if all(t in results for t in chateval.ALL_TASKS):
        centered = [(results[t] - chateval.BASELINE[t]) / (1 - chateval.BASELINE[t]) for t in chateval.ALL_TASKS]
        report["chatcore"] = sum(centered) / len(centered)
        print(f"ChatCORE metric: {report['chatcore']:.4f}")
    path = write_report(args.report_dir or default_report_dir(args.ckpt_path), args.stage, report)
    print(f"report -> {path}")


if __name__ == "__main__":
    main()
