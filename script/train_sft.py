"""Supervised fine-tuning, on one GPU or with one process per GPU.

Examples:
python -m script.train_sft --datasets=data/sft_datasets.json --use_wandb=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    -m script.train_sft --datasets=data/sft_datasets.json --use_wandb=0
"""

import argparse
import json
import math
import os
parser=argparse.ArgumentParser()
parser.add_argument("--device_batch_size", default=4, type=int, help="batch size per gradient step")
parser.add_argument("--batch_token", default=2**19, type=int, help="how many token should have in one step")
parser.add_argument("--eval_token", default=80*2048, type=int, help="how many token we use for eval")
parser.add_argument("--eval_every", default=200, type=int, help="evaluate the model per eval_every steps")
parser.add_argument("--sample_every", default=200, type=int, help="sample the model per sample_every steps")
parser.add_argument("--save_every", default=200, type=int, help="save the model per save_every steps")
parser.add_argument("--tokenizer_path", default="weight/tokenizer.pkl", type=str, help="path of tokenizer.pkl, use train_token.py to get tokenizer")
parser.add_argument("--device_type", default="cuda", choices=("cuda", "cpu"), help="cuda or cpu")
parser.add_argument("--use_wandb", type=int, default=1, help="if use wandb to log(0 to disable wandb)")
parser.add_argument(
    "--datasets", type=str, required=True,
    help=("path to a JSON dataset manifest; each item supports path, "
          "type(parquet/jsonl), holdout_last and eval"),
)
parser.add_argument("--pretrained_ckpt_path", type=str, default="weight/pretrained_model.ckpt", help="path of pretrained model checkpoint")
parser.add_argument("--save_ckpt_path", type=str, default="weight/sft_model.ckpt", help="path to save sft model checkpoint")
parser.add_argument("--compile_gpt",type=int,default=1,help="compile gpt to speed up training")
parser.add_argument("--epochs", default=1.0, type=float, help="approximate passes over the selected dataset")
parser.add_argument("--max_steps", default=0, type=int, help="override calculated steps when greater than zero")
parser.add_argument("--final_eval", default=1, type=int, help="evaluate before the final save")
# hyper parameter
parser.add_argument("--init_lr_frac", default=0.8, type=float, help="initial lr fraction relative to pretrain lr")
parser.add_argument("--warm_down_ratio", default=0.5, type=float, help="lr warmdown ratio")
parser.add_argument("--final_lr_frac", default=0., type=float, help="min lr fraction")
parser.add_argument("--lmhead_lr",default=0.008,type=float,help="lmhead learning rate")
parser.add_argument("--embedding_lr",default=0.3,type=float,help="learning rate for embedding")
parser.add_argument("--matrix_lr",default=0.02,type=float,help="matrix learning rate")
# model config
parser.add_argument("--seq_len", default=1024, type=int, help="model sequence len")
from tasks import sftdata,custom_json,basedata
from tokenizer import tokenizer
import torch
import torch.distributed as dist
from model import gpt
from evaluation import bpbeval,sample
import time
from common import common
from optimizer import optimizer
args=parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt_path)), exist_ok=True)
ddp=all(name in os.environ for name in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'))
if ddp and args.device_type != "cuda":
    raise ValueError("torchrun multi-GPU SFT requires --device_type=cuda")
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
if ddp:
    local_rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    device=f"cuda:{local_rank}"
else:
    rank=0
    local_rank=0
    world_size=1
    device="cuda:0" if args.device_type == "cuda" else "cpu"
is_main=rank == 0
batch_size=int(args.batch_token/args.seq_len)
global_micro_batch=args.device_batch_size*world_size
if batch_size < global_micro_batch or batch_size % global_micro_batch != 0:
    raise ValueError(
        f"batch_token/seq_len ({batch_size}) must be a positive multiple of "
        f"device_batch_size*world_size ({global_micro_batch})"
    )
gradient_step=batch_size//global_micro_batch
tokeniz=tokenizer.Tokenizer.from_dict(args.tokenizer_path, device=device)


def load_dataset_specs(path):
    """Load unified dataset definitions from a JSON manifest file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--datasets manifest does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        specs = json.load(f)
    if not isinstance(specs, list) or not specs:
        raise ValueError("--datasets manifest must contain a non-empty JSON array")
    return specs


def make_dataset(spec, split, dataset_batch_size, dataset_device=device):
    if not isinstance(spec, dict) or not spec.get("path"):
        raise ValueError("each --datasets item must be an object containing path")
    path = os.path.expanduser(str(spec["path"]))
    source_type = spec.get("type")
    if source_type is None:
        source_type = "jsonl" if path.endswith(".jsonl") or ".jsonl" in path else "parquet"
    if source_type == "jsonl":
        if split == "test":
            raise ValueError("JSONL sources do not provide an automatic evaluation split")
        dataset = custom_json.CustomJson(
            path, tokeniz, dataset_batch_size, args.seq_len, dataset_device
        )
    elif source_type == "parquet":
        # SFTDataset accepts a directory and a filename glob separately.
        data_dir, file_pattern = os.path.split(path)
        data_dir = data_dir or "."
        dataset = sftdata.SFTDataset(
            data_dir, split, tokeniz, dataset_batch_size, args.seq_len, dataset_device,
            file_pattern=file_pattern or "*.parquet",
            holdout_last=bool(spec.get("holdout_last", True)),
        )
    else:
        raise ValueError(f"unsupported dataset type {source_type!r}; use parquet or jsonl")
    if len(dataset) == 0:
        raise ValueError(f"dataset matched no {split} rows: {path}")
    return dataset


dataset_specs = load_dataset_specs(args.datasets)
eval_data_set = None
source_batch_size=args.device_batch_size*world_size
source_device="cpu" if ddp else device
children = [make_dataset(spec, "train", source_batch_size, source_device)
            for spec in dataset_specs]
eval_specs = [spec for spec in dataset_specs
              if spec.get("type", "jsonl" if ".jsonl" in str(spec["path"]) else "parquet") == "parquet"
              and spec.get("eval", True)]
if eval_specs:
    eval_children = [make_dataset(spec, "test", args.device_batch_size) for spec in eval_specs]
    eval_data_set = basedata.MixedDataset(*eval_children)
train_data_set = basedata.MixedDataset(*children)
if ddp:
    train_data_set=basedata.BatchShardDataset(
        train_data_set, rank, world_size, args.device_batch_size, device
    )
if is_main:
    ESTIMATE_TOTAL_TOKEN=train_data_set.estimate_total_token()
else:
    ESTIMATE_TOTAL_TOKEN=0
if ddp:
    total_token_holder=[ESTIMATE_TOTAL_TOKEN]
    dist.broadcast_object_list(total_token_holder, src=0)
    ESTIMATE_TOTAL_TOKEN=total_token_holder[0]
train_data_loader=iter(train_data_set)
SCALE_RATIO=1 #to make the progress more quick
if device.startswith("cuda"):
    device_name=torch.cuda.get_device_name(local_rank)
    OPTIMAL_FLOPS=common.get_peak_flops(device_name)*world_size
else:
    OPTIMAL_FLOPS=float("inf")
eval_step=int(args.eval_token/args.seq_len/args.device_batch_size)
total_steps=max(1, int(ESTIMATE_TOTAL_TOKEN * args.epochs / args.batch_token / SCALE_RATIO))
if args.max_steps > 0:
    total_steps=args.max_steps
progress=0.
now_step=0
start_iter=0
best_bpb=float('inf')
pbar=common.get_pbar() if is_main else None
config_all=vars(args)
if args.use_wandb and is_main:
    import wandb
    run=wandb.init(entity="enyuwang1023097618-123",project="mygpt",config=config_all)



with open(args.pretrained_ckpt_path,"rb") as f:
    resume_dict=torch.load(f,map_location='cpu')
    new_dict=common.strip_prefix(resume_dict['model_state'])
    config=gpt.GPTConfig(**resume_dict['model_config'])
    config.seq_len=args.seq_len
    orig_gpt=gpt.GPT(config,device,tokeniz)
    orig_gpt.init_weight()
    orig_gpt.load_state_dict(new_dict)
    param_groups=orig_gpt.get_param_groups(
        lmhead_lr=args.lmhead_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr
    )
    optimiz=optimizer.MuonOptimizer(param_groups)
    optimiz_data=optimiz.param_groups
    optimizer_state=resume_dict['optimizer_state_dict']
    if isinstance(optimizer_state, list):
        if len(optimizer_state) != world_size:
            raise ValueError(
                "pretrained checkpoint optimizer state was saved for "
                f"{len(optimizer_state)} ranks, but this run uses {world_size}"
            )
        optimizer_state=optimizer_state[rank]
    optimiz.load_state_dict(optimizer_state)
    if ddp and not isinstance(resume_dict['optimizer_state_dict'], list):
        # A single-rank checkpoint stores full optimizer buffers. MuonOptimizer
        # shards those buffers by parameter rows/parameter groups under DDP.
        for group in optimiz.param_groups:
            if group['kind'] == 'muon':
                state=optimiz.state.get(group['params'][0], {})
                chunk_size=math.ceil(len(group['params'])/world_size)
                start=rank*chunk_size
                stop=min(start+chunk_size, len(group['params']))
                for key in ('momentum_buffer1', 'momentum_buffer2'):
                    if key in state:
                        state[key]=state[key][start:stop].clone()
            elif group['kind'] == 'adamw':
                for param in group['params']:
                    if param.numel() < 1024 or param.shape[0] < world_size:
                        continue
                    state=optimiz.state.get(param, {})
                    chunk_size=math.ceil(param.shape[0]/world_size)
                    start=rank*chunk_size
                    stop=min(start+chunk_size, param.shape[0])
                    for key in ('param_avg', 'param_avg_sq'):
                        if key in state:
                            state[key]=state[key][start:stop].clone()
    for (group,orig_group) in zip(optimiz.param_groups,optimiz_data):
        group['init_lr']=orig_group['init_lr']*args.init_lr_frac
        group['lr']=group['init_lr']
    del param_groups
    del new_dict
    del optimiz_data
    del resume_dict


if eval_data_set is None and (args.final_eval or args.eval_every > 0):
    raise ValueError("--datasets needs at least one parquet source with eval=true when evaluation is enabled")
test_data_loader = lambda: iter(eval_data_set)
if args.compile_gpt:
    c_gpt=torch.compile(orig_gpt,dynamic=False)
else:
    c_gpt=orig_gpt
token_bytes=tokeniz.get_token_bytes()

flops_per_token=orig_gpt.get_flops_per_token()
batch_flops=flops_per_token*args.batch_token
optimal_time=batch_flops/OPTIMAL_FLOPS


def get_lr_multiply(progress):
    if progress<1-args.warm_down_ratio:
        return 1
    else:
        now_percent=(progress+args.warm_down_ratio-1)/args.warm_down_ratio
        return (1-now_percent)*1+now_percent*args.final_lr_frac

def get_momentum(step):
    if step<300:
        now_percent=step/300
        return now_percent*0.95+(1-now_percent)*0.85
    else:
        return 0.95

def save_model(train_loader_state_dict,bpb=None):
    global best_bpb
    optimizer_state=optimiz.state_dict()
    if ddp:
        train_loader_states=[None for _ in range(world_size)] if is_main else None
        optimizer_states=[None for _ in range(world_size)] if is_main else None
        dist.gather_object(train_loader_state_dict, train_loader_states, dst=0)
        dist.gather_object(optimizer_state, optimizer_states, dst=0)
    else:
        train_loader_states=train_loader_state_dict
        optimizer_states=optimizer_state
    if is_main and bpb is None and args.final_eval:
        bpb=bpbeval.bpb_eval(orig_gpt,eval_step,test_data_loader(),token_bytes,device,segment=True)
    if is_main and (bpb is None or bpb<best_bpb):
        if bpb is not None:
            best_bpb=bpb
        tmp=args.save_ckpt_path+".tmp"
        torch.save(
            {
                "model_state":orig_gpt.state_dict(),
                "model_config":vars(config),
                "optimizer_state_dict":optimizer_states,
                "train_loader_state_dict":train_loader_states,
                "step":now_step,
                "best_bpb":best_bpb,
                "runid":run.id if args.use_wandb else None
            },tmp
        )
        os.replace(tmp,args.save_ckpt_path)
        print("model saved")
    if ddp:
        dist.barrier()
train_loader_state_dict=None
max_dump_save=3
for now_step in range(start_iter,total_steps):
    bpb=None
    if args.eval_every > 0 and now_step%args.eval_every==0 and now_step!=0:
        if is_main:
            bpb=bpbeval.bpb_eval(orig_gpt,eval_step,test_data_loader(),token_bytes,device,segment=True)
            val_log_data={"val/bpb":bpb}
            print(val_log_data)
            if args.use_wandb:
                run.log(data=val_log_data,step=now_step)
        if ddp:
            dist.barrier()
    if args.sample_every > 0 and now_step%args.sample_every==0 and now_step!=0:
        if is_main:
            sample.sample_chat(orig_gpt,tokeniz,args.seq_len,device)
        if ddp:
            dist.barrier()
    if args.save_every > 0 and now_step%args.save_every==0 and now_step!=0:
        save_model(train_loader_state_dict,bpb)
    t1=time.perf_counter()
    progress = now_step/total_steps
    total_loss=torch.tensor(0.,device=device)
    lr_ratio=get_lr_multiply(progress)
    momentum=get_momentum(now_step)
    for g in optimiz.param_groups:
        g['lr']=g['init_lr']*lr_ratio
        if g['kind'] == 'muon':
            g['momentum1']=momentum
    optimiz.zero_grad(set_to_none=True)
    for i in range(gradient_step):
        x,y,segment_batch,train_loader_state_dict=next(train_data_loader)
        output=c_gpt(x,y,segment_ids=segment_batch)
        loss=output.loss
        loss/=gradient_step
        loss.backward()
        total_loss+=loss.detach()
    optimiz.step()
    if ddp:
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    t4=time.perf_counter()
    usedtime=t4-t1
    train_log_data={"train/total_time_use":usedtime,"train/total_loss":total_loss.item(),"train/gpu_usage":optimal_time/usedtime,"train/lr_ratio":lr_ratio}
    if args.use_wandb and is_main:
        run.log(data=train_log_data,step=now_step)
    if is_main:
        pbar.n=progress*100
        pbar.refresh()
        pbar.set_postfix_str(f"step {now_step}: time used {usedtime:.2f}, gpu usage {optimal_time/usedtime:.2f}, loss: {total_loss.item()}")

save_model(train_loader_state_dict)
if args.use_wandb and is_main:
    run.finish()
if is_main:
    pbar.close()
if ddp:
    dist.destroy_process_group()
