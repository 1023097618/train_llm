"""Reinforcement learning on GSM8K, like nanochat scripts/chat_rl.py ("GRPO", really closer to REINFORCE):
1) no KL regularization to a reference model
2) on policy, so no PPO ratio + clip
3) DAPO style token level normalization
4) advantage = reward - mean reward of the samples of the same question
Two optional additions beyond nanochat (both off by default):
- --kl_beta: per token KL penalty to the frozen SFT model (k3 estimator as in GRPO), pulls back the drift that
  noisy updates cause (without it the d12 and d24 policies degraded after ~100 steps)
- --oversample: DAPO style dynamic sampling, roll out oversample * examples_per_step questions and train on the
  first examples_per_step whose samples are not all right / all wrong (the others carry no learning signal)

Rollouts are sampled with vLLM (same process, weights pushed after every step). vLLM always runs on cuda:0;
--device puts the training model on another GPU, e.g. CUDA_VISIBLE_DEVICES=<vllm gpu>,<train gpu> ... --device=cuda:1
(a d24 model with its optimizer state and vLLM do not fit on one 32GB GPU together).
The model may call the calculator during rollouts; forced tool output tokens are not trained.
pass@k on the GSM8K test set is evaluated at step 0 (= the SFT model) and every --eval_every steps.
Every --save_every steps the full training state (model, optimizer, step, data cursor, eval history) goes to
<save_ckpt_path>.state; after the run is killed, the same command with --resume=<that .state file> continues from it.

usage:
python -m script.chat_rl --ckpt_path=weight/d20/sft_model.ckpt --tokenizer_path=weight/d20/tokenizer.pkl \
    --save_ckpt_path=weight/d20/rl_model.ckpt
two GPUs (vLLM on GPU 3, training on GPU 0):
CUDA_VISIBLE_DEVICES=3,0 python -m script.chat_rl ... --device=cuda:1 --gpu_memory_utilization=0.85
"""
import sys
sys.stdout.reconfigure(line_buffering=True)  # show progress in nohup logs immediately
import argparse
import os
import time

import torch

from common import common
from common.report import default_report_dir, write_report
from evaluation.sample import calculate
from model import gpt
from optimizer import optimizer
from tasks.chateval import GSM8K, EvalTask, NumericQA
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="RL on GSM8K")
parser.add_argument("--ckpt_path", type=str, default="weight/sft_model.ckpt", help="SFT checkpoint to start from")
parser.add_argument("--tokenizer_path", type=str, default="weight/tokenizer.pkl")
parser.add_argument("--save_ckpt_path", type=str, default="weight/rl_model.ckpt")
parser.add_argument("--use_wandb", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=1, help="epochs over GSM8K train")
parser.add_argument("--max_steps", type=int, default=0, help="stop early when > 0")
parser.add_argument("--device_batch_size", type=int, default=8, help="max sequences per forward pass")
# 32 questions per step (nanochat: 16) and init_lr_frac 0.02 (nanochat: 0.05): with nanochat's values the d12 policy
# collapsed after ~70 steps, Muon's fixed size updates followed the noisy RL gradient and broke the matrices
parser.add_argument("--examples_per_step", type=int, default=32, help="questions per optimization step")
parser.add_argument("--num_samples", type=int, default=16, help="samples per question")
# our JSON tool calls are much longer than nanochat's inline <<expr=result>>: with 256 tokens half of the SFT model's
# GSM8K rollouts were cut off, and RL collapsed by learning to answer shorter instead of better
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--embedding_lr", type=float, default=0.2)
parser.add_argument("--lmhead_lr", type=float, default=0.004)
parser.add_argument("--matrix_lr", type=float, default=0.02)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--init_lr_frac", type=float, default=0.02)
parser.add_argument("--kl_beta", type=float, default=0.0, help="KL penalty to the SFT model, 0 = off (nanochat)")
parser.add_argument("--ref_device", type=str, default="cuda:0", help="device of the frozen reference model (with --kl_beta)")
parser.add_argument("--train_data", type=str, default="", help="comma separated jsonl files of {question, answer} to train on instead of GSM8K train (script/prepare_rl_data.py)")
parser.add_argument("--oversample", type=float, default=1.0, help="questions rolled out per step / examples_per_step, 1 = nanochat")
parser.add_argument("--eval_every", type=int, default=60)
parser.add_argument("--eval_examples", type=int, default=400)
parser.add_argument("--device", type=str, default="cuda:0", help="training model device, vLLM is always on cuda:0")
parser.add_argument("--gpu_memory_utilization", type=float, default=0.3, help="vLLM memory fraction of cuda:0 (lower it when the training model shares the GPU)")
parser.add_argument("--enforce_eager", type=int, default=0, help="disable vLLM cuda graphs")
parser.add_argument("--save_every", type=int, default=30, help="save the resumable training state every N steps, 0 = never")
parser.add_argument("--resume", type=str, default="", help="training state file to continue from (<save_ckpt_path>.state), empty = start from --ckpt_path")
parser.add_argument("--report_dir", type=str, default=None, help="default: <save ckpt dir>/report")
args = parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt_path)), exist_ok=True)

device = args.device
torch.manual_seed(42)
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
config = gpt.GPTConfig(**ckpt["model_config"])
model = gpt.GPT(config, device, tokeniz)
model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
ref_model = None
if args.kl_beta > 0:
    ref_model = gpt.GPT(config, args.ref_device, tokeniz)
    ref_model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
    ref_model.eval().requires_grad_(False)
del ckpt

param_groups = model.get_param_groups(lmhead_lr=args.lmhead_lr, embedding_lr=args.embedding_lr,
                                      matrix_lr=args.matrix_lr, weight_decay=args.weight_decay)
for g in param_groups:
    g["lr"] = g["init_lr"] = g["lr"] * args.init_lr_frac
optimiz = optimizer.MuonOptimizer(param_groups)

state_path = args.save_ckpt_path + ".state"
resume = None
if args.resume:
    resume = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(resume["model_state"])  # before vLLM copies the weights
    optimiz.load_state_dict(resume["optimizer_state"])
    print(f"resumed from {args.resume} at step {resume['step']}")

from inference.vllm_engine import VLLMEngine
engine = VLLMEngine(model, tokeniz, gpu_memory_utilization=args.gpu_memory_utilization,
                    enforce_eager=bool(args.enforce_eager), seed=42)

train_task = NumericQA(args.train_data) if args.train_data else GSM8K("train")
val_task = GSM8K("test")
tools = {"calculate": calculate}
end_assistant = tokeniz.special_token["<|/assistant|>"]
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
if args.max_steps > 0:
    num_steps = min(num_steps, args.max_steps)
assert args.num_samples % args.device_batch_size == 0 or args.num_samples < args.device_batch_size
print(f"RL train data: {args.train_data or 'GSM8K train'} ({len(train_task)} questions)")
print(f"RL steps: {num_steps}, sequences per step: {args.examples_per_step * args.num_samples}")
if args.use_wandb:
    import wandb
    run = wandb.init(entity="enyuwang1023097618-123", project="mygpt-rl", config=vars(args),
                     **({"id": resume["wandb_id"], "resume": "allow"} if resume and resume.get("wandb_id") else {}))


def render_prompt(conversation):
    tokens, _ = tokeniz.render_conversation({"source": "gsm8k", "messages": EvalTask.prompt_messages(conversation)}, last_is_user=True)
    return tokens


def completion_text(tokens):
    return tokeniz.decode([t for t in tokens if t != end_assistant])


def eval_passk(step):
    """pass@k for k = 1..device_batch_size, sampled at the RL temperature like nanochat"""
    n = min(args.eval_examples, len(val_task))
    conversations = [val_task[i] for i in range(n)]
    outputs = engine.generate([render_prompt(c) for c in conversations], num_samples=args.device_batch_size,
                              max_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
                              tools=tools, seed=1234)
    correct = [[val_task.evaluate(c, completion_text(t)) for t, _ in samples] for c, samples in zip(conversations, outputs)]
    passk = {f"pass@{k}": sum(any(row[:k]) for row in correct) / n for k in range(1, args.device_batch_size + 1)}
    print(f"[eval] step {step}: " + ", ".join(f"{k} {v:.4f}" for k, v in passk.items()))
    return passk


best = {"step": None, "pass@1": -1.0}


def evaluate_and_save(step, history):
    """eval pass@k; the checkpoint keeps the best pass@1 so far (step 0 is the SFT model itself),
    so a policy that degrades later never overwrites a better one"""
    history.append({"step": step, **eval_passk(step)})
    if args.use_wandb:
        run.log(history[-1], step=step)
    if history[-1]["pass@1"] > best["pass@1"]:
        best.update(step=step, **{"pass@1": history[-1]["pass@1"]})
        tmp = args.save_ckpt_path + ".tmp"
        torch.save({"model_state": model.state_dict(), "model_config": vars(config), "step": step,
                    "source_ckpt": args.ckpt_path}, tmp)
        os.replace(tmp, args.save_ckpt_path)
        print(f"new best pass@1 {best['pass@1']:.4f}: saved {args.save_ckpt_path} (step {step})")
    write_report(args.report_dir or default_report_dir(args.save_ckpt_path), "rl_gsm8k",
                 {"ckpt": args.save_ckpt_path, "start": history[0], "last": history[-1],
                  "best": next(h for h in history if h["step"] == best["step"]), "history": history})


def save_state(step):
    """everything needed to continue at `step` (its eval is already in history), written atomically"""
    tmp = state_path + ".tmp"
    torch.save({"step": step, "model_state": common.to_cpu(model.state_dict()),
                "optimizer_state": common.to_cpu(optimiz.state_dict()), "example_cursor": example_cursor,
                "history": history, "best": dict(best), "wandb_id": run.id if args.use_wandb else None}, tmp)
    os.replace(tmp, state_path)
    print(f"saved training state at step {step} -> {state_path}")


history = []
example_cursor = 0
start_step = 0
if resume:
    start_step, example_cursor, history = resume["step"], resume["example_cursor"], resume["history"]
    best.update(resume["best"])
    del resume
pbar = common.get_pbar()
for step in range(start_step, num_steps):
    if step % args.eval_every == 0 and not (history and history[-1]["step"] == step):
        evaluate_and_save(step, history)
    if args.save_every > 0 and step > start_step and step % args.save_every == 0:
        save_state(step)
    t0 = time.time()
    # ---- rollouts ----
    num_rollout = max(args.examples_per_step, round(args.examples_per_step * args.oversample))
    indices = [(example_cursor + i) % len(train_task) for i in range(num_rollout)]
    example_cursor += num_rollout
    conversations = [train_task[i] for i in indices]
    prompts = [render_prompt(c) for c in conversations]
    outputs = engine.generate(prompts, num_samples=args.num_samples, max_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_k=args.top_k, tools=tools, seed=step)
    t_rollout = time.time() - t0
    # ---- policy gradient ----
    model.train()
    mean_rewards, lengths, kls = [], [], []
    truncated = sum(t[-1] != end_assistant for samples in outputs for t, _ in samples if t) / sum(len(s) for s in outputs)
    trained_questions = 0
    for conversation, prompt, samples in zip(conversations, prompts, outputs):
        rewards = torch.tensor([train_task.reward(conversation, completion_text(t)) for t, _ in samples], device=device)
        advantages = rewards - rewards.mean()
        mean_rewards.append(rewards.mean().item())
        lengths.extend(len(t) for t, _ in samples)
        if advantages.abs().sum() == 0 or trained_questions == args.examples_per_step:
            continue  # all samples equally good or bad: no learning signal for this question (or the step is full)
        trained_questions += 1
        # the forward pass takes seq[:-1], so a sequence may hold at most seq_len + 1 tokens
        seqs = [(prompt + t)[:config.seq_len + 1] for t, _ in samples]
        masks = [([0] * len(prompt) + m)[:config.seq_len + 1] for _, m in samples]
        longest = max(len(s) for s in seqs)
        ids = torch.tensor([s + [end_assistant] * (longest - len(s)) for s in seqs], device=device)
        mask = torch.tensor([m + [0] * (longest - len(m)) for m in masks], device=device)
        inputs, targets = ids[:, :-1], ids[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -1  # prompt, forced tool output and padding are not trained
        num_passes = max(1, len(seqs) // args.device_batch_size)
        for b0 in range(0, len(seqs), args.device_batch_size):
            b1 = b0 + args.device_batch_size
            logp = -model(inputs[b0:b1], targets[b0:b1], reduction="none").loss.view(inputs[b0:b1].shape)
            valid = targets[b0:b1] >= 0
            num_valid = valid.sum().clamp(min=1)
            obj = logp * advantages[b0:b1].unsqueeze(-1)
            if ref_model is not None:
                with torch.no_grad():
                    logp_ref = -ref_model(inputs[b0:b1].to(args.ref_device), targets[b0:b1].to(args.ref_device),
                                          reduction="none").loss.view(inputs[b0:b1].shape).to(device)
                log_ratio = logp_ref - logp  # k3 estimator of KL(policy || ref): ratio - log ratio - 1 >= 0
                kl = (torch.exp(log_ratio) - log_ratio - 1) * valid
                obj = obj - args.kl_beta * kl
                kls.append((kl.sum() / num_valid).item())
            pg_obj = obj.sum() / (num_valid * num_passes * args.examples_per_step)
            (-pg_obj).backward()
    lrm = 1.0 - step / num_steps
    for g in optimiz.param_groups:
        g["lr"] = g["init_lr"] * lrm
    if trained_questions > 0:
        optimiz.step()
        optimiz.zero_grad(set_to_none=True)
        engine.update_weights(model)
    mean_reward = sum(mean_rewards) / len(mean_rewards)
    log = {"train/reward": mean_reward, "train/truncated": truncated, "train/trained_questions": trained_questions, "train/sequence_length": sum(lengths) / len(lengths), "train/lrm": lrm,
           "train/kl": sum(kls) / len(kls) if kls else 0.0,
           "train/rollout_time": t_rollout, "train/step_time": time.time() - t0}
    if args.use_wandb:
        run.log(log, step=step)
    pbar.n = (step + 1) / num_steps * 100
    pbar.refresh()
    pbar.set_postfix_str(f"step {step}: reward {mean_reward:.3f}, len {log['train/sequence_length']:.0f}, "
                         f"rollout {t_rollout:.1f}s, step {log['train/step_time']:.1f}s")
    print(f"step {step}/{num_steps} | reward {mean_reward:.3f} | trained {trained_questions}/{len(conversations)} | kl {log['train/kl']:.4f} | seq len {log['train/sequence_length']:.0f} | truncated {truncated:.2f} | "
          f"rollout {t_rollout:.1f}s | step {log['train/step_time']:.1f}s", flush=True)

evaluate_and_save(num_steps, history)
print(f"best pass@1 {best['pass@1']:.4f} at step {best['step']}")
pbar.close()
engine.close()
if args.use_wandb:
    run.finish()
