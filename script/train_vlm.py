"""Image understanding on top of a chat model (LLaVA / MiniMind-V style), one GPU or one process per GPU.

--stage=align: only the projector trains (frozen SigLIP2 + frozen GPT), ALLaVA captions
--stage=sft:   projector + the whole GPT, ALLaVA instructions + captions + replay of the text SFT mixture
SigLIP2 always stays frozen. Saves {"model_state" (GPT, loadable by chat_eval), "model_config", "projector_state",
"vision_config"}; every --save_every steps the full training state goes to <save_ckpt_path>.state (--resume=<file>).

usage:
python -m script.download_vlm && python -m script.prepare_vlm_data
CUDA_VISIBLE_DEVICES=0,4 torchrun --standalone --nproc_per_node=2 -m script.train_vlm --stage=align \
    --ckpt_path=weight/d24/sft_model.ckpt --tokenizer_path=weight/d24/tokenizer.pkl --save_ckpt_path=weight/d24/vlm_align.ckpt
CUDA_VISIBLE_DEVICES=0,4 torchrun --standalone --nproc_per_node=2 -m script.train_vlm --stage=sft \
    --ckpt_path=weight/d24/vlm_align.ckpt --tokenizer_path=weight/d24/tokenizer.pkl --save_ckpt_path=weight/d24/vlm_sft.ckpt
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import dataclasses
import io
import math
import os
import time

import torch
import torch.distributed as dist
from PIL import Image
from pyarrow import parquet as pq
from torch.utils.data import DataLoader

from common import common
from common.data_paths import VLM_DATA, VLM_ENCODER
from model import gpt
from model.vision import VisionConfig, VisionTower, preprocess
from optimizer import optimizer
from tasks import sftdata, vlmdata
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="train image understanding")
parser.add_argument("--stage", choices=("align", "sft"), required=True)
parser.add_argument("--ckpt_path", type=str, required=True, help="chat model (align) or the align checkpoint (sft)")
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--save_ckpt_path", type=str, required=True)
parser.add_argument("--encoder_path", type=str, default=str(VLM_ENCODER))
parser.add_argument("--data_dir", type=str, default=str(VLM_DATA))
parser.add_argument("--caption_weight", type=float, default=None, help="sampling weight of captions (align 1, sft 0.3)")
parser.add_argument("--instruct_weight", type=float, default=None, help="sampling weight of instructions (align 0, sft 0.5)")
parser.add_argument("--text_weight", type=float, default=None, help="sampling weight of the text SFT replay (align 0, sft 0.2)")
parser.add_argument("--epochs", type=float, default=1.0, help="passes over the main data of the stage (align: captions, sft: instructions)")
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--device_batch_size", type=int, default=2)
parser.add_argument("--batch_token", type=int, default=2**18)
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--projector_lr", type=float, default=None, help="AdamW lr of the projector (align 2e-3, sft 2e-4)")
parser.add_argument("--llm_lr_frac", type=float, default=0.1, help="sft: GPT lr as a fraction of the default (pretrain) lr")
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--warm_down_ratio", type=float, default=0.5)
parser.add_argument("--eval_every", type=int, default=100)
parser.add_argument("--eval_rows", type=int, default=64, help="validation batches (device_batch_size rows each)")
parser.add_argument("--save_every", type=int, default=100)
parser.add_argument("--resume", type=str, default="", help="training state file to continue from")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--compile_gpt", type=int, default=1)
parser.add_argument("--use_wandb", type=int, default=1)
args = parser.parse_args()
DEFAULTS = {"align": dict(caption_weight=1.0, instruct_weight=0.0, text_weight=0.0, projector_lr=2e-3),
            "sft": dict(caption_weight=0.3, instruct_weight=0.5, text_weight=0.2, projector_lr=2e-4)}
for k, v in DEFAULTS[args.stage].items():
    if getattr(args, k) is None:
        setattr(args, k, v)
os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt_path)), exist_ok=True)

ddp = all(k in os.environ for k in ("LOCAL_RANK", "RANK", "WORLD_SIZE"))
torch.manual_seed(42)
if ddp:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank, world_size, device = dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}"
else:
    rank, local_rank, world_size, device = 0, 0, 1, "cuda:0"
is_main = rank == 0
rows_per_step = args.batch_token // args.seq_len
assert rows_per_step % (args.device_batch_size * world_size) == 0, "batch_token/seq_len must be a multiple of device_batch_size*world_size"
grad_accum = rows_per_step // (args.device_batch_size * world_size)
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)

# ---- model ----
ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
config = gpt.GPTConfig(**ckpt["model_config"])
config.seq_len = args.seq_len
model = gpt.GPT(config, device, tokeniz)
model.load_state_dict(common.strip_prefix(ckpt["model_state"]))
vision_config = VisionConfig(**{**ckpt.get("vision_config", {}), "encoder_path": args.encoder_path})
vision = VisionTower(vision_config, config.model_dim, device)
if "projector_state" in ckpt:
    vision.projector.load_state_dict(ckpt["projector_state"])
del ckpt
image_pad_id = tokeniz.special_token["<|image pad|>"]

projector_group = dict(kind="adamw", params=list(vision.projector.parameters()), lr=args.projector_lr,
                       beta1=0.9, beta2=0.95, wd=0.0, eps=1e-8)
groups = [projector_group]
if args.stage == "align":
    model.requires_grad_(False)
else:
    batch_scale = (args.batch_token / 2**19) ** 0.5  # same sqrt(batch) rule as pretrain / SFT
    for g in model.get_param_groups():
        g["lr"] *= args.llm_lr_frac * batch_scale
        g["wd"] = 0.0
        groups.append(g)
for g in groups:
    g["init_lr"], g["init_wd"] = g["lr"], g["wd"]
optimiz = optimizer.MuonOptimizer(groups)
trained = [p for g in groups for p in g["params"]]

# ---- data ----
files = {name: vlmdata.shard_files(args.data_dir, name, "train") for name in ("caption", "instruct")}
weights = {"caption": args.caption_weight, "instruct": args.instruct_weight}
image_sources = [(name, files[name], weights[name]) for name in files if weights[name] > 0]
text_mixture = None
if args.text_weight > 0:
    text_mixture = sftdata.TaskMixture([sftdata.SmolTalk("train"), sftdata.MMLU("auxiliary_train"), sftdata.GSM8K("train"),
                                        sftdata.HermesFunctionCalling()])
resume = None
if args.resume:
    resume = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(resume["model_state"])
    vision.projector.load_state_dict(resume["projector_state"])
    optimiz.load_state_dict(resume["optimizer_state"][rank])
loader_state = dict(resume["loader_state"][rank]) if resume else {}
train_set = vlmdata.VLMDataset(image_sources, tokeniz, args.device_batch_size, args.seq_len, vision_config.tokens_per_image,
                               vision_config.image_size, text_mixture, args.text_weight, rank, world_size,
                               resume_state_dict=loader_state)
loader_kwargs = dict(batch_size=None, num_workers=args.num_workers, pin_memory=True)
if args.num_workers > 0:
    loader_kwargs["prefetch_factor"] = 4
train_loader = iter(DataLoader(train_set, **loader_kwargs))
val_sources = [(name, vlmdata.shard_files(args.data_dir, name, "val"), w) for name, _, w in image_sources]

# steps: the main source (align: captions, sft: instructions) is seen `epochs` times, the other sources come along
# with their sampling weights; tokens per conversation are estimated from a sample of the main source
main = "caption" if args.stage == "align" else "instruct"
main_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files[main])
main_share = weights[main] / (args.caption_weight + args.instruct_weight + args.text_weight)
probe_lens = []
for messages, images in vlmdata.ImageTextStream(files[main], 0, 1, vision_config.tokens_per_image):
    r = train_set.render(messages, images)
    if r:
        probe_lens.append(len(r[0]))
    if len(probe_lens) >= 500:
        break
mean_len = sum(probe_lens) / len(probe_lens)
total_steps = max(1, int(main_rows * args.epochs / main_share * mean_len / args.batch_token))
if args.max_steps > 0:
    total_steps = min(total_steps, args.max_steps)
start_step = resume["step"] if resume else 0
best_val = resume["best_val"] if resume else float("inf")
if is_main:
    print(f"stage {args.stage}: {main_rows:,} {main} rows x {args.epochs} epochs, ~{mean_len:.0f} tokens each, "
          f"{total_steps} steps of {args.batch_token:,} tokens (grad accum {grad_accum}), sources {weights} text {args.text_weight}")
run = None
if args.use_wandb and is_main:
    import wandb
    run = wandb.init(entity="enyuwang1023097618-123", project="mygpt-vlm", config=vars(args),
                     **({"id": resume["wandb_id"], "resume": "allow"} if resume and resume.get("wandb_id") else {}))
del resume

c_model = torch.compile(model, dynamic=False) if args.compile_gpt else model


def forward(m, x, y, segments, pixels):
    image_embeds = vision.image_embeds(x, pixels, config.model_dim, image_pad_id)
    return m(x, y, segment_ids=segments, image_embeds=image_embeds).loss


@torch.no_grad()
def evaluate():
    """mean loss on the validation shards (same image mixture, no text replay)"""
    model.eval()
    data = iter(vlmdata.VLMDataset(val_sources, tokeniz, args.device_batch_size, args.seq_len, vision_config.tokens_per_image,
                                   vision_config.image_size, seed=1))
    losses = []
    for _ in range(args.eval_rows):
        x, y, segments, pixels, _ = next(data)
        losses.append(forward(c_model, *(t.to(device) for t in (x, y, segments, pixels))).item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def sample_caption(max_tokens=80):
    """greedy description of the first validation image"""
    stream = vlmdata.ImageTextStream(val_sources[0][1], 0, 1, vision_config.tokens_per_image)
    _, images = next(iter(stream))
    messages = [{"role": "user", "content": [{"type": "image", "length": vision_config.tokens_per_image},
                                             {"type": "text", "content": "Describe this image briefly."}]}]
    tokens, _ = tokeniz.render_conversation({"source": "sample", "messages": messages}, last_is_user=True)
    pixels = preprocess([Image.open(io.BytesIO(images[0]))], vision_config.image_size).to(device)
    end = tokeniz.special_token["<|/assistant|>"]
    out = []
    model.eval()
    for _ in range(max_tokens):
        x = torch.tensor([tokens + out], device=device)
        logits = model(x, image_embeds=vision.image_embeds(x, pixels, config.model_dim, image_pad_id)).logit
        nxt = int(logits[0, -1].argmax())
        if nxt == end:
            break
        out.append(nxt)
    model.train()
    return tokeniz.decode(out)


def save(step, val=None):
    """model checkpoint when val improves (or no val), plus the resumable training state"""
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
        payload = {"model_state": model.state_dict(), "model_config": vars(config),
                   "projector_state": vision.projector.state_dict(), "vision_config": dataclasses.asdict(vision_config),
                   "step": step, "stage": args.stage, "source_ckpt": args.ckpt_path}
        if val is None or val < best_val:
            best_val = min(best_val, val) if val is not None else best_val
            torch.save(payload, args.save_ckpt_path + ".tmp")
            os.replace(args.save_ckpt_path + ".tmp", args.save_ckpt_path)
            print(f"saved {args.save_ckpt_path} (step {step}, val {val})")
        torch.save({**payload, "model_state": common.to_cpu(model.state_dict()), "optimizer_state": states,
                    "loader_state": loaders, "best_val": best_val, "wandb_id": run.id if run else None},
                   args.save_ckpt_path + ".state.tmp")
        os.replace(args.save_ckpt_path + ".state.tmp", args.save_ckpt_path + ".state")
    if ddp:
        dist.barrier()


def lr_multiplier(step):
    if step < args.warmup_steps:
        return (step + 1) / args.warmup_steps
    progress = step / total_steps
    if progress < 1 - args.warm_down_ratio:
        return 1.0
    return (1 - progress) / args.warm_down_ratio


model.train()
vision.train()
for step in range(start_step, total_steps):
    if step > start_step and step % args.eval_every == 0:
        val = evaluate() if is_main else None
        if is_main:
            print(f"[eval] step {step}: val loss {val:.4f} | sample: {sample_caption()!r}")
            if run:
                run.log({"val/loss": val}, step=step)
        if step % args.save_every == 0:
            save(step, val)
    elif step > start_step and step % args.save_every == 0:
        save(step)
    t0 = time.time()
    lrm = lr_multiplier(step)
    for g in optimiz.param_groups:
        g["lr"] = g["init_lr"] * lrm
    total = torch.zeros((), device=device)
    n_images = 0
    for _ in range(grad_accum):
        x, y, segments, pixels, state = next(train_loader)
        loader_state[state["worker"]] = state
        n_images += pixels.shape[0]
        loss = forward(c_model, *(t.to(device, non_blocking=True) for t in (x, y, segments, pixels))) / grad_accum
        loss.backward()
        total += loss.detach()
    for p in trained:  # a step with only text rows leaves the projector without gradient
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    optimiz.step()
    optimiz.zero_grad(set_to_none=True)
    if ddp:
        dist.all_reduce(total, op=dist.ReduceOp.AVG)
    if is_main and (step % 10 == 0 or step == start_step):
        dt = time.time() - t0
        print(f"step {step}/{total_steps} | loss {total.item():.4f} | lrm {lrm:.3f} | images {n_images} | {dt:.1f}s")
        if run:
            run.log({"train/loss": total.item(), "train/lrm": lrm, "train/step_time": dt, "train/images": n_images}, step=step)

val = evaluate() if is_main else None
if is_main:
    print(f"[eval] step {total_steps}: val loss {val:.4f} | sample: {sample_caption()!r}")
save(total_steps, val)
if run:
    run.finish()
if ddp:
    dist.destroy_process_group()
