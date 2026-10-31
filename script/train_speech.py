"""Speech in and out on top of a chat model; the GPT ("thinker") stays frozen, so the language ability cannot degrade.

--stage=t2a: text question -> spoken answer, only the talker trains (the thinker only runs up to its mid layer)
--stage=a2a: spoken question -> spoken answer + text answer, talker + audio projector (t2a data mixed in);
             the text loss through the frozen thinker teaches the projector to understand speech
Saves {"talker_state", "audio_proj_state", "talker_config", "hidden_layer", "thinker_ckpt"}; every --save_every steps
the full training state goes to <save_ckpt_path>.state (--resume=<file>).

usage:
python -m script.download_audio && python -m script.prepare_audio_data
CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 -m script.train_speech --stage=t2a \
    --thinker_ckpt=weight/d24/vlm_rl.ckpt --tokenizer_path=weight/d24/tokenizer.pkl --save_ckpt_path=weight/d24/speech_t2a.ckpt
CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 -m script.train_speech --stage=a2a \
    --thinker_ckpt=weight/d24/vlm_rl.ckpt --speech_ckpt=weight/d24/speech_t2a.ckpt --tokenizer_path=weight/d24/tokenizer.pkl \
    --save_ckpt_path=weight/d24/speech_a2a.ckpt
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import argparse
import dataclasses
import os
import time

import torch
import torch.distributed as dist
from pyarrow import parquet as pq
from torch.utils.data import DataLoader

from common import common
from common.data_paths import AUDIO_DATA, AUDIO_MIMI
from inference.speech import MimiDecoder, generate_speech
from model import gpt
from model.speech import AudioProjector, Talker, TalkerConfig
from optimizer import optimizer
from tasks import speechdata
from tokenizer import tokenizer

parser = argparse.ArgumentParser(description="train speech in and out")
parser.add_argument("--stage", choices=("t2a", "a2a"), required=True)
parser.add_argument("--thinker_ckpt", type=str, required=True)
parser.add_argument("--speech_ckpt", type=str, default="", help="a2a: the t2a checkpoint")
parser.add_argument("--tokenizer_path", type=str, required=True)
parser.add_argument("--save_ckpt_path", type=str, required=True)
parser.add_argument("--data_dir", type=str, default=str(AUDIO_DATA))
parser.add_argument("--t2a_weight", type=float, default=None, help="sampling weight of t2a data (t2a 1, a2a 0.3)")
parser.add_argument("--a2a_weight", type=float, default=None, help="sampling weight of a2a data (t2a 0, a2a 0.7)")
parser.add_argument("--epochs", type=float, default=None, help="passes over the main set (t2a 3, a2a 3)")
parser.add_argument("--max_steps", type=int, default=0)
parser.add_argument("--device_batch_size", type=int, default=2)
parser.add_argument("--batch_token", type=int, default=2**17)
parser.add_argument("--seq_len", type=int, default=2048)
parser.add_argument("--adamw_lr", type=float, default=None, help="talker / projector AdamW lr (t2a 2e-3, a2a 5e-4)")
parser.add_argument("--matrix_lr", type=float, default=None, help="talker Muon lr (t2a 0.02, a2a 0.005)")
parser.add_argument("--warmup_steps", type=int, default=30)
parser.add_argument("--warm_down_ratio", type=float, default=0.5)
parser.add_argument("--eval_every", type=int, default=200)
parser.add_argument("--eval_batches", type=int, default=32)
parser.add_argument("--sample_every", type=int, default=400, help="write a spoken sample wav next to the checkpoint")
parser.add_argument("--save_every", type=int, default=200)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--compile_thinker", type=int, default=1)
parser.add_argument("--use_wandb", type=int, default=1)
parser.add_argument("--device_type", choices=("cuda", "cpu"), default="cuda", help="cpu: quick single process test")
args = parser.parse_args()
DEFAULTS = {"t2a": dict(t2a_weight=1.0, a2a_weight=0.0, epochs=3.0, adamw_lr=2e-3, matrix_lr=0.02),
            "a2a": dict(t2a_weight=0.3, a2a_weight=0.7, epochs=3.0, adamw_lr=5e-4, matrix_lr=0.005)}
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
    rank, world_size, device = 0, 1, "cuda:0" if args.device_type == "cuda" else "cpu"
is_main = rank == 0
rows_per_step = args.batch_token // args.seq_len
assert rows_per_step % (args.device_batch_size * world_size) == 0, "batch_token/seq_len must be a multiple of device_batch_size*world_size"
grad_accum = rows_per_step // (args.device_batch_size * world_size)
tokeniz = tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)

# ---- model ----
ckpt = torch.load(args.thinker_ckpt, map_location="cpu", weights_only=False)
config = gpt.GPTConfig(**ckpt["model_config"])
config.seq_len = args.seq_len
thinker = gpt.GPT(config, device, tokeniz)
thinker.load_state_dict(common.strip_prefix(ckpt["model_state"]))
thinker.requires_grad_(False).eval()
del ckpt
hidden_layer = config.n_layer // 2 - 1  # MiniMind-O's bridge layer
talker_config = TalkerConfig(seq_len=args.seq_len)
talker = Talker(talker_config, config.model_dim, device)
audio_proj = AudioProjector(config.model_dim).to(device)
if args.speech_ckpt:
    speech = torch.load(args.speech_ckpt, map_location="cpu", weights_only=False)
    talker.load_state_dict(speech["talker_state"])
    audio_proj.load_state_dict(speech["audio_proj_state"])
    del speech
audio_pad_id = tokeniz.special_token["<|audio pad|>"]

groups = talker.get_param_groups(adamw_lr=args.adamw_lr, matrix_lr=args.matrix_lr)
if args.stage == "a2a":
    groups.append(dict(kind="adamw", params=list(audio_proj.parameters()), lr=args.adamw_lr, beta1=0.9, beta2=0.95, wd=0.0, eps=1e-8))
for g in groups:
    g["init_lr"], g["init_wd"] = g["lr"], g["wd"]
optimiz = optimizer.MuonOptimizer(groups)
trained = [p for g in groups for p in g["params"]]
if is_main:
    print(f"talker: {sum(p.numel() for p in talker.parameters()):,} parameters, bridge from thinker block {hidden_layer}")

# ---- data ----
weights = {"t2a": args.t2a_weight, "a2a": args.a2a_weight}
sources = [(name, speechdata.shard_files(args.data_dir, name, "train"), w) for name, w in weights.items() if w > 0]
val_sources = [(name, speechdata.shard_files(args.data_dir, name, "val"), w) for name, w in weights.items() if w > 0]
resume = None
if args.resume:
    resume = torch.load(args.resume, map_location="cpu", weights_only=False)
    talker.load_state_dict(resume["talker_state"])
    audio_proj.load_state_dict(resume["audio_proj_state"])
    optimiz.load_state_dict(resume["optimizer_state"][rank])
loader_state = dict(resume["loader_state"][rank]) if resume else {}
train_set = speechdata.SpeechDataset(sources, tokeniz, talker_config, args.device_batch_size, args.seq_len, rank, world_size,
                                     resume_state_dict=loader_state)
loader_kwargs = dict(batch_size=None, num_workers=args.num_workers, pin_memory=device != "cpu")
if args.num_workers > 0:
    loader_kwargs["prefetch_factor"] = 4
train_loader = iter(DataLoader(train_set, **loader_kwargs))

# steps: the stage's main set is seen `epochs` times, tokens per turn estimated from a sample of it
main = args.stage
main_files = dict((n, f) for n, f, _ in sources)[main]
main_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in main_files)
main_share = weights[main] / sum(weights.values())
lens = []
for ids, _, _, _ in speechdata.SpeechStream(main_files, 0, 1, tokeniz, talker_config, args.seq_len):
    lens.append(len(ids[0]))
    if len(lens) >= 500:
        break
mean_len = sum(lens) / len(lens)
total_steps = max(1, int(main_rows * args.epochs / main_share * mean_len / args.batch_token))
if args.max_steps > 0:
    total_steps = min(total_steps, args.max_steps)
start_step = resume["step"] if resume else 0
best_val = resume["best_val"] if resume else float("inf")
if is_main:
    print(f"stage {args.stage}: {main_rows:,} {main} rows x {args.epochs} epochs, ~{mean_len:.0f} positions per turn, "
          f"{total_steps} steps of {args.batch_token:,} tokens (grad accum {grad_accum})")
run = None
if args.use_wandb and is_main:
    import wandb
    run = wandb.init(entity="enyuwang1023097618-123", project="mygpt-speech", config=vars(args),
                     **({"id": resume["wandb_id"], "resume": "allow"} if resume and resume.get("wandb_id") else {}))
del resume
c_thinker = torch.compile(thinker, dynamic=False) if args.compile_thinker and device != "cpu" else thinker


def forward(x, y, segments, features, speakers):
    """total loss, audio loss, text loss, per codebook losses"""
    x, y, segments = x.to(device, non_blocking=True), y.to(device, non_blocking=True), segments.to(device, non_blocking=True)
    text_x, audio_x = x[:, 0], x[:, 1:]
    text_y, audio_y = y[:, 0], y[:, 1:]
    speakers = speakers.to(device, non_blocking=True)
    if args.stage == "t2a":
        with torch.no_grad():
            bridge = c_thinker(text_x, segment_ids=segments, hidden_layer=hidden_layer, hidden_only=True).hidden
        text_loss = torch.zeros((), device=device)
    else:
        media = audio_proj.media_embeds(text_x, features.to(device, non_blocking=True), config.model_dim, audio_pad_id)
        out = c_thinker(text_x, text_y, segment_ids=segments, media_embeds=media, hidden_layer=hidden_layer)
        bridge, text_loss = out.hidden, out.loss
    _, audio_loss, per_codebook = talker(audio_x, bridge, text_x, audio_y, segments, spk_emb=speakers)
    return audio_loss + text_loss, audio_loss, text_loss, per_codebook


@torch.no_grad()
def evaluate():
    talker.eval()
    data = iter(speechdata.SpeechDataset(val_sources, tokeniz, talker_config, args.device_batch_size, args.seq_len, seed=1))
    audio, text, codebooks = 0.0, 0.0, 0.0
    for _ in range(args.eval_batches):
        x, y, segments, features, speakers, _ = next(data)
        _, a, t, per = forward(x, y, segments, features, speakers)
        audio, text, codebooks = audio + a.item(), text + t.item(), codebooks + per
    talker.train()
    n = args.eval_batches
    return audio / n, text / n, (codebooks / n).tolist()


def speak(step):
    """a spoken answer to a typed question: reply text on the log, audio next to the checkpoint"""
    import soundfile as sf
    talker.eval()
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    prompt, _ = tokeniz.render_conversation({"source": "sample", "messages": messages}, last_is_user=True)
    text, codes = generate_speech(thinker, talker, tokeniz, prompt, hidden_layer)
    path = f"{args.save_ckpt_path}.sample{step}.wav"
    if codes.shape[1] > 0:
        mimi = MimiDecoder(AUDIO_MIMI, device)
        sf.write(path, mimi.decode(codes), mimi.sample_rate)
        del mimi
    talker.train()
    print(f"[sample] step {step}: {codes.shape[1]} frames ({codes.shape[1] / 12.5:.1f}s) -> {path} | text: {text[:200]!r}")


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
        payload = {"talker_state": common.to_cpu(talker.state_dict()), "audio_proj_state": common.to_cpu(audio_proj.state_dict()),
                   "talker_config": dataclasses.asdict(talker_config), "hidden_layer": hidden_layer,
                   "thinker_ckpt": args.thinker_ckpt, "step": step, "stage": args.stage}
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
    return (1 - progress) / args.warm_down_ratio


def eval_and_log(step):
    val = None
    if is_main:
        audio, text, per = evaluate()
        val = audio + text
        print(f"[eval] step {step}: audio loss {audio:.4f} | text loss {text:.4f} | codebooks " + " ".join(f"{p:.2f}" for p in per))
        if run:
            run.log({"val/audio_loss": audio, "val/text_loss": text}, step=step)
    return val


talker.train()
for step in range(start_step, total_steps):
    val = None
    if step > start_step and args.eval_every > 0 and step % args.eval_every == 0:
        val = eval_and_log(step)
    if is_main and step > start_step and args.sample_every > 0 and step % args.sample_every == 0:
        speak(step)
    if step > start_step and args.save_every > 0 and step % args.save_every == 0:
        save(step, val)
    t0 = time.time()
    lrm = lr_multiplier(step)
    for g in optimiz.param_groups:
        g["lr"] = g["init_lr"] * lrm
    total_audio, total_text = torch.zeros((), device=device), torch.zeros((), device=device)
    t_data = 0.0
    for _ in range(grad_accum):
        t1 = time.time()
        x, y, segments, features, speakers, state = next(train_loader)
        t_data += time.time() - t1
        loader_state[state["worker"]] = state
        loss, audio_loss, text_loss, _ = forward(x, y, segments, features, speakers)
        (loss / grad_accum).backward()
        total_audio += audio_loss.detach() / grad_accum
        total_text += text_loss.detach() / grad_accum
    for p in trained:  # a batch without speakers or spoken questions leaves those parts without gradient
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    optimiz.step()
    optimiz.zero_grad(set_to_none=True)
    if ddp:
        dist.all_reduce(total_audio, op=dist.ReduceOp.AVG)
        dist.all_reduce(total_text, op=dist.ReduceOp.AVG)
    if is_main and (step % 10 == 0 or step == start_step):
        dt = time.time() - t0
        print(f"step {step}/{total_steps} | audio loss {total_audio.item():.4f} | text loss {total_text.item():.4f} | "
              f"lrm {lrm:.3f} | {dt:.1f}s (waiting for data {t_data:.1f}s)")
        if run:
            run.log({"train/audio_loss": total_audio.item(), "train/text_loss": total_text.item(), "train/lrm": lrm, "train/step_time": dt}, step=step)

val = eval_and_log(total_steps)
if is_main:
    speak(total_steps)
save(total_steps, val)
if run:
    run.finish()
if ddp:
    dist.destroy_process_group()
