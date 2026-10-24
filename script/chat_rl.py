"""Reinforcement learning on GSM8K, like nanochat scripts/chat_rl.py ("GRPO", really closer to REINFORCE):
1) no KL regularization to a reference model
2) on policy, so no PPO ratio + clip
3) DAPO style token level normalization
4) advantage = reward - mean reward of the samples of the same question

Rollouts are sampled with vLLM (same process, same GPU, weights pushed after every step).
The model may call the calculator during rollouts; forced tool output tokens are not trained.
pass@k on the GSM8K test set is evaluated at step 0 (= the SFT model) and every --eval_every steps.

usage:
python -m script.chat_rl --ckpt_path=weight/d20/sft_model.ckpt --tokenizer_path=weight/d20/tokenizer.pkl \
    --save_ckpt_path=weight/d20/rl_model.ckpt
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
from tasks.chateval import GSM8K, EvalTask
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="RL on GSM8K")
parser.add_argument("--ckpt_path", type=str, default="weight/sft_model.ckpt", help="SFT checkpoint to start from")
parser.add_argument("--tokenizer_path", type=str, default="weight/tokenizer.pkl")
parser.add_argument("--save_ckpt_path", type=str, default="weight/rl_model.ckpt")
parser.add_argument("--use_wandb", type=int, default=1)
parser.add_argument("--num_epochs", type=int, default=1, help="epochs over GSM8K train")
parser.add_argument("--max_steps", type=int, default=0, help="stop early when > 0")
parser.add_argument("--device_batch_size", type=int, default=8, help="max sequences per forward pass")
parser.add_argument("--examples_per_step", type=int, default=16, help="questions per optimization step")
parser.add_argument("--num_samples", type=int, default=16, help="samples per question")
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--embedding_lr", type=float, default=0.2)
parser.add_argument("--lmhead_lr", type=float, default=0.004)
parser.add_argument("--matrix_lr", type=float, default=0.02)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--init_lr_frac", type=float, default=0.05)
parser.add_argument("--eval_every", type=int, default=60)
parser.add_argument("--eval_examples", type=int, default=400)
parser.add_argument("--save_every", type=int, default=60)
parser.add_argument("--gpu_memory_utilization", type=float, default=0.3, help="vLLM memory fraction (the training model shares the GPU)")
parser.add_argument("--enforce_eager", type=int, default=0, help="disable vLLM cuda graphs")
parser.add_argument("--report_dir", type=str, default=None, help="default: <save ckpt dir>/report")
args = parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt_path)), exist_ok=True)

device = "cuda"
torch.manual_seed(42)
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
config = gpt.GPTConfig(**ckpt["model_config"])
model = gpt.GPT(config, device, tokeniz)
model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
del ckpt

param_groups = model.get_param_groups(lmhead_lr=args.lmhead_lr, embedding_lr=args.embedding_lr,
                                      matrix_lr=args.matrix_lr, weight_decay=args.weight_decay)
for g in param_groups:
    g["lr"] = g["init_lr"] = g["lr"] * args.init_lr_frac
optimiz = optimizer.MuonOptimizer(param_groups)

from inference.vllm_engine import VLLMEngine
engine = VLLMEngine(model, tokeniz, gpu_memory_utilization=args.gpu_memory_utilization,
                    enforce_eager=bool(args.enforce_eager), seed=42)

train_task = GSM8K("train")
val_task = GSM8K("test")
tools = {"calculate": calculate}
end_assistant = tokeniz.special_token["<|/assistant|>"]
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
if args.max_steps > 0:
    num_steps = min(num_steps, args.max_steps)
assert args.num_samples % args.device_batch_size == 0 or args.num_samples < args.device_batch_size
print(f"RL steps: {num_steps}, sequences per step: {args.examples_per_step * args.num_samples}")
if args.use_wandb:
    import wandb
    run = wandb.init(entity="enyuwang1023097618-123", project="mygpt-rl", config=vars(args))


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


def save(step, history):
    tmp = args.save_ckpt_path + ".tmp"
    torch.save({"model_state": model.state_dict(), "model_config": vars(config), "step": step,
                "source_ckpt": args.ckpt_path}, tmp)
    os.replace(tmp, args.save_ckpt_path)
    write_report(args.report_dir or default_report_dir(args.save_ckpt_path), "rl_gsm8k",
                 {"ckpt": args.save_ckpt_path, "start": history[0], "last": history[-1], "history": history})
    print(f"saved {args.save_ckpt_path} (step {step})")


history = []
example_cursor = 0
pbar = common.get_pbar()
for step in range(num_steps):
    if step % args.eval_every == 0:
        history.append({"step": step, **eval_passk(step)})
        if args.use_wandb:
            run.log(history[-1], step=step)
    t0 = time.time()
    # ---- rollouts ----
    indices = [(example_cursor + i) % len(train_task) for i in range(args.examples_per_step)]
    example_cursor += args.examples_per_step
    conversations = [train_task[i] for i in indices]
    prompts = [render_prompt(c) for c in conversations]
    outputs = engine.generate(prompts, num_samples=args.num_samples, max_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_k=args.top_k, tools=tools, seed=step)
    t_rollout = time.time() - t0
    # ---- policy gradient ----
    model.train()
    mean_rewards, lengths = [], []
    trained_questions = 0
    for conversation, prompt, samples in zip(conversations, prompts, outputs):
        rewards = torch.tensor([train_task.reward(conversation, completion_text(t)) for t, _ in samples], device=device)
        advantages = rewards - rewards.mean()
        mean_rewards.append(rewards.mean().item())
        lengths.extend(len(t) for t, _ in samples)
        if advantages.abs().sum() == 0:
            continue  # all samples equally good or bad: no learning signal for this question
        trained_questions += 1
        seqs = [prompt + t for t, _ in samples]
        masks = [[0] * len(prompt) + m for _, m in samples]
        longest = max(len(s) for s in seqs)
        ids = torch.tensor([s + [end_assistant] * (longest - len(s)) for s in seqs], device=device)
        mask = torch.tensor([m + [0] * (longest - len(m)) for m in masks], device=device)
        inputs, targets = ids[:, :-1], ids[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -1  # prompt, forced tool output and padding are not trained
        num_passes = max(1, len(seqs) // args.device_batch_size)
        for b0 in range(0, len(seqs), args.device_batch_size):
            b1 = b0 + args.device_batch_size
            logp = -model(inputs[b0:b1], targets[b0:b1], reduction="none").loss.view(inputs[b0:b1].shape)
            num_valid = (targets[b0:b1] >= 0).sum().clamp(min=1)
            pg_obj = (logp * advantages[b0:b1].unsqueeze(-1)).sum() / (num_valid * num_passes * args.examples_per_step)
            (-pg_obj).backward()
    lrm = 1.0 - step / num_steps
    for g in optimiz.param_groups:
        g["lr"] = g["init_lr"] * lrm
    if trained_questions > 0:
        optimiz.step()
        optimiz.zero_grad(set_to_none=True)
        engine.update_weights(model)
    mean_reward = sum(mean_rewards) / len(mean_rewards)
    log = {"train/reward": mean_reward, "train/trained_questions": trained_questions, "train/sequence_length": sum(lengths) / len(lengths), "train/lrm": lrm,
           "train/rollout_time": t_rollout, "train/step_time": time.time() - t0}
    if args.use_wandb:
        run.log(log, step=step)
    pbar.n = (step + 1) / num_steps * 100
    pbar.refresh()
    pbar.set_postfix_str(f"step {step}: reward {mean_reward:.3f}, len {log['train/sequence_length']:.0f}, "
                         f"rollout {t_rollout:.1f}s, step {log['train/step_time']:.1f}s")
    print(f"step {step}/{num_steps} | reward {mean_reward:.3f} | seq len {log['train/sequence_length']:.0f} | "
          f"rollout {t_rollout:.1f}s | step {log['train/step_time']:.1f}s", flush=True)
    if (step > 0 and step % args.save_every == 0) and step != num_steps - 1:
        save(step, history)

history.append({"step": num_steps, **eval_passk(num_steps)})
if args.use_wandb:
    run.log(history[-1], step=num_steps)
save(num_steps, history)
pbar.close()
engine.close()
if args.use_wandb:
    run.finish()
