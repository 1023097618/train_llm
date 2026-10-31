"""Train the image decoder (model/imagegen.py); the chat model only reads the captions and stays frozen.

usage:
python -m script.prepare_t2i_data
CUDA_VISIBLE_DEVICES=2 python -m script.train_imagegen --thinker_ckpt=weight/d24/vlm_rl.ckpt \
    --tokenizer_path=weight/d24/tokenizer.pkl --save_ckpt_path=weight/d24/imagegen.ckpt
Every --save_every steps the training state goes to <save_ckpt_path>.state (--resume=<file>).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import dataclasses
import os
import re
import time

import torch
import torch.distributed as dist
from PIL import Image
from pyarrow import parquet as pq
from torch.utils.data import DataLoader

from common import common
from common.data_paths import T2I_DATA, T2I_VQ_MODEL
from model import gpt
from model.imagegen import ImageGen, ImageGenConfig, load_vq
from optimizer import optimizer
from tasks import t2idata
from tokenizer import tokenizer

PROMPTS = ["An orange cat sitting on a stone wall in the sunlight.",
           "A red sports car parked on a mountain road at sunset.",
           "A bowl of fresh fruit on a wooden table.",
           "A snowy mountain above a calm blue lake."]

parser = argparse.ArgumentParser(description="train image generation")
parser.add_argument("--thinker_ckpt", type=str, required=True)
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--save_ckpt_path", type=str, required=True)
parser.add_argument("--data_dir", type=str, default=str(T2I_DATA))
parser.add_argument("--vq_dir", type=str, default=str(T2I_VQ_MODEL))
parser.add_argument("--depth", type=int, default=12, help="decoder depth, width follows (nanochat scaling)")
parser.add_argument("--epochs", type=float, default=20.0)
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--device_batch_size", type=int, default=16)
parser.add_argument("--batch_token", type=int, default=2**19)
parser.add_argument("--weight_decay", type=float, default=0.2)
parser.add_argument("--matrix_lr", type=float, default=0.02)
parser.add_argument("--embedding_lr", type=float, default=0.3)
parser.add_argument("--lmhead_lr", type=float, default=0.008)
parser.add_argument("--warmup_steps", type=int, default=50)
parser.add_argument("--warm_down_ratio", type=float, default=0.65)
parser.add_argument("--final_lr_frac", type=float, default=0.05)
parser.add_argument("--eval_every", type=int, default=200)
parser.add_argument("--eval_batches", type=int, default=16)
parser.add_argument("--sample_every", type=int, default=400)
parser.add_argument("--save_every", type=int, default=200)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--compile", type=int, default=1)
parser.add_argument("--use_wandb", type=int, default=1)
args = parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt_path)), exist_ok=True)

ddp = all(k in os.environ for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
torch.manual_seed(42)
if ddp:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world_size, device = dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"
else:
    rank, world_size, device = 0, 1, "cuda:0"
is_main = rank == 0
config = ImageGenConfig(depth=args.depth)
rows_per_step = args.batch_token // config.seq_len
assert rows_per_step % (args.device_batch_size * world_size) == 0, "batch_token/seq_len must be a multiple of device_batch_size*world_size"
grad_accum = rows_per_step // (args.device_batch_size * world_size)
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)

# ---- models ----
ckpt = torch.load(args.thinker_ckpt, map_location="cpu", weights_only=False)
thinker = gpt.GPT(gpt.GPTConfig(**ckpt["model_config"]), device, tokeniz)
thinker.load_state_dict(common.strip_prefix(ckpt["model_state"]))
thinker.requires_grad_(False).eval()
del ckpt
last_block = thinker.config.n_layer - 1
model = ImageGen(config, thinker.config.model_dim, device)
groups = model.get_param_groups(weight_decay=args.weight_decay, matrix_lr=args.matrix_lr, embedding_lr=args.embedding_lr, lmhead_lr=args.lmhead_lr)
optimiz = optimizer.MuonOptimizer(groups)
if is_main:
    print(f"decoder {model.decoder.config.n_layer} layers x {model.decoder.config.model_dim}, {sum(p.numel() for p in model.parameters()):,} parameters")

# ---- data ----
train_files = t2idata.shard_files(args.data_dir, "train")
val_files = t2idata.shard_files(args.data_dir, "val")
resume = None
if args.resume:
    resume = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(resume["model_state"])
    optimiz.load_state_dict(resume["optimizer_state"][rank])
loader_state = dict(resume["loader_state"][rank]) if resume else {}
train_set = t2idata.T2IDataset(train_files, tokeniz, args.device_batch_size, config.prefix_capacity, rank, world_size, resume_state_dict=loader_state)
loader_kwargs = dict(batch_size=None, num_workers=args.num_workers, pin_memory=True)
if args.num_workers > 0:
    loader_kwargs["prefetch_factor"] = 4
train_loader = iter(DataLoader(train_set, **loader_kwargs))
rows = sum(pq.ParquetFile(f).metadata.num_rows for f in train_files)
total_steps = max(1, int(rows * args.epochs / rows_per_step))
if args.max_steps > 0:
    total_steps = min(total_steps, args.max_steps)
start_step = resume["step"] if resume else 0
best_val = resume["best_val"] if resume else float("inf")
if is_main:
    print(f"{rows:,} images x {args.epochs} epochs = {total_steps} steps of {rows_per_step} images (grad accum {grad_accum})")
run = None
if args.use_wandb and is_main:
    import wandb
    run = wandb.init(entity="enyuwang1023097618-123", project="mygpt-imagegen", config=vars(args),
                     **({"id": resume["wandb_id"], "resume": "allow"} if resume and resume.get("wandb_id") else {}))
del resume
c_decoder = torch.compile(model.decoder, dynamic=False) if args.compile else model.decoder
c_thinker = torch.compile(thinker, dynamic=False) if args.compile else thinker


def run_batch(prefix, lengths, visual):
    prefix, lengths, visual = prefix.to(device, non_blocking=True), lengths.to(device, non_blocking=True), visual.to(device, non_blocking=True)
    with torch.no_grad():
        hidden = c_thinker(prefix, hidden_layer=last_block, hidden_only=True).hidden
    return model(c_decoder, hidden, lengths, visual)


@torch.no_grad()
def evaluate():
    model.eval()
    data = iter(t2idata.T2IDataset(val_files, tokeniz, args.device_batch_size, config.prefix_capacity))
    loss, acc = 0.0, 0.0
    for _ in range(args.eval_batches):
        prefix, lengths, visual, _ = next(data)
        l, a = run_batch(prefix, lengths, visual)
        loss, acc = loss + l.item(), acc + a.item()
    model.train()
    return loss / args.eval_batches, acc / args.eval_batches


@torch.no_grad()
def sample_images(step):
    model.eval()
    decode = load_vq(args.vq_dir, device)
    out_dir = args.save_ckpt_path + ".samples"
    os.makedirs(out_dir, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(1234)
    for n, prompt in enumerate(PROMPTS):
        ids = [tokeniz.special_token["<|user|>"]] + tokeniz.encode_ordinary(prompt) + [tokeniz.special_token["<|/user|>"]]
        hidden = thinker(torch.tensor([ids], device=device), hidden_layer=last_block, hidden_only=True).hidden
        tokens = model.sample(hidden, generator=generator)
        Image.fromarray(decode(tokens)).save(f"{out_dir}/step{step:06d}_{n}_{re.sub('[^a-z]+', '_', prompt.lower())[:30]}.png")
    del decode
    model.train()
    print(f"[sample] step {step} -> {out_dir}")


def save(step, val=None):
    global best_val
    opt_state = common.to_cpu(optimiz.state_dict())
    states = [None] * world_size if is_main else None
    loaders = [None] * world_size if is_main else None
    if ddp:
        dist.gather_object(opt_state, states, dst=0)
        dist.gather_object(loader_state, loaders, dst=0)
    else:
        states, loaders = [opt_state], [loader_state]
    if is_main:
        payload = {"model_state": common.to_cpu(common.strip_prefix(model.state_dict())), "config": dataclasses.asdict(config),
                   "thinker_ckpt": args.thinker_ckpt, "step": step}
        if val is None or val < best_val:
            best_val = min(best_val, val) if val is not None else best_val
            torch.save(payload, args.save_ckpt_path + ".tmp")
            os.replace(args.save_ckpt_path + ".tmp", args.save_ckpt_path)
            print(f"saved {args.save_ckpt_path} (step {step}, val {val})")
        torch.save({**payload, "optimizer_state": states, "loader_state": loaders, "best_val": best_val,
                    "wandb_id": run.id if run else None}, args.save_ckpt_path + ".state.tmp")
        os.replace(args.save_ckpt_path + ".state.tmp", args.save_ckpt_path + ".state")
    if ddp:
        dist.barrier()


def lr_multiplier(step):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    progress = step / total_steps
    if progress < 1 - args.warm_down_ratio:
        return 1.0
    frac = (progress - 1 + args.warm_down_ratio) / args.warm_down_ratio
    return 1 - frac * (1 - args.final_lr_frac)


model.train()
for step in range(start_step, total_steps):
    val = None
    if step > start_step and step % args.eval_every == 0 and is_main:
        val, acc = evaluate()
        print(f"[eval] step {step}: val loss {val:.4f} | accuracy {acc:.4f}")
        if run:
            run.log({"val/loss": val, "val/accuracy": acc}, step=step)
    if step > start_step and step % args.sample_every == 0 and is_main:
        sample_images(step)
    if step > start_step and step % args.save_every == 0:
        save(step, val)
    t0 = time.time()
    lrm = lr_multiplier(step)
    for g in optimiz.param_groups:
        g["lr"] = g["init_lr"] * lrm
    total = torch.zeros((), device=device)
    t_data = 0.0
    for _ in range(grad_accum):
        t1 = time.time()
        prefix, lengths, visual, state = next(train_loader)
        t_data += time.time() - t1
        loader_state[state["worker"]] = state
        loss, _ = run_batch(prefix, lengths, visual)
        (loss / grad_accum).backward()
        total += loss.detach() / grad_accum
    optimiz.step()
    optimiz.zero_grad(set_to_none=True)
    if ddp:
        dist.all_reduce(total, op=dist.ReduceOp.AVG)
    if is_main and (step % 10 == 0 or step == start_step):
        dt = time.time() - t0
        print(f"step {step}/{total_steps} | loss {total.item():.4f} | lrm {lrm:.3f} | {dt:.1f}s (waiting for data {t_data:.1f}s)")
        if run:
            run.log({"train/loss": total.item(), "train/lrm": lrm, "train/step_time": dt}, step=step)

val = None
if is_main:
    val, acc = evaluate()
    print(f"[eval] step {total_steps}: val loss {val:.4f} | accuracy {acc:.4f}")
    sample_images(total_steps)
save(total_steps, val)
if run:
    run.finish()
if ddp:
    dist.destroy_process_group()
