"""Supervised fine-tuning, on one GPU or with one process per GPU.

Examples:
python -m script.download_sft
python -m script.train_sft --use_wandb=0
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
    -m script.train_sft --use_wandb=0

Data mixture follows nanochat's chat_sft.py: SmolTalk + MMLU x3 + GSM8K x4,
plus hermes function-calling data and our own identity conversations.
"""

import argparse
import math
import os
from common.data_paths import SFT_IDENTITY, SFT_IDENTITY_QA, SFT_TOOLCALL_SYNTH
parser=argparse.ArgumentParser()
parser.add_argument("--device_batch_size", default=4, type=int, help="batch size per gradient step")
parser.add_argument("--batch_token", default=2**19, type=int, help="how many token should have in one step")
parser.add_argument("--eval_token", default=8*2**20, type=int, help="how many token we use for eval")
parser.add_argument("--eval_every", default=200, type=int, help="evaluate the model per eval_every steps")
parser.add_argument("--sample_every", default=200, type=int, help="sample the model per sample_every steps")
parser.add_argument("--save_every", default=200, type=int, help="save the model per save_every steps")
parser.add_argument("--tokenizer_path", default="weight/tokenizer.pkl", type=str, help="path of tokenizer.pkl, use train_token.py to get tokenizer")
parser.add_argument("--device_type", default="cuda", choices=("cuda", "cpu"), help="cuda or cpu")
parser.add_argument("--use_wandb", type=int, default=1, help="if use wandb to log(0 to disable wandb)")
parser.add_argument("--mmlu_epochs", default=3, type=int, help="epochs of MMLU auxiliary_train in the mixture (teaches multiple choice)")
parser.add_argument("--gsm8k_epochs", default=4, type=int, help="epochs of GSM8K train in the mixture (teaches math and calculator use)")
parser.add_argument("--toolcall_epochs", default=4, type=int, help="epochs of hermes function-calling data in the mixture (teaches general tool use, 0 to disable)")
parser.add_argument("--toolcall_synth_path", default=str(SFT_TOOLCALL_SYNTH), type=str, help="jsonl of synthetic tool calls (retry / ask / chain / irrelevant)")
parser.add_argument("--toolcall_synth_epochs", default=4, type=int, help="epochs of the synthetic tool calls (0 to disable)")
parser.add_argument("--identity_path", default=str(SFT_IDENTITY), type=str, help="jsonl of identity conversations")
parser.add_argument("--identity_epochs", default=2, type=int, help="epochs of identity conversations in the mixture(0 to disable)")
parser.add_argument("--identity_qa_path", default=str(SFT_IDENTITY_QA), type=str, help="jsonl of short direct identity Q&A")
parser.add_argument("--identity_qa_epochs", default=2, type=int, help="epochs of the short identity Q&A (0 to disable)")
parser.add_argument("--pretrained_ckpt_path", type=str, default="weight/pretrained_model.ckpt", help="path of pretrained model checkpoint")
parser.add_argument("--save_ckpt_path", type=str, default="weight/sft_model.ckpt", help="path to save sft model checkpoint")
parser.add_argument("--resume",default=-1,type=int,help="resume from --save_ckpt_path (-1 is disable resume)")
parser.add_argument("--num_workers",type=int,default=4,help="dataloader worker number")
parser.add_argument("--compile_gpt",type=int,default=1,help="compile gpt to speed up training")
parser.add_argument("--epochs", default=1.0, type=float, help="approximate passes over the selected dataset")
parser.add_argument("--max_steps", default=0, type=int, help="override calculated steps when greater than zero")
parser.add_argument("--final_eval", default=1, type=int, help="evaluate before the final save")
# hyper parameter
parser.add_argument("--init_lr_frac", default=0.8, type=float, help="initial lr fraction relative to pretrain lr")
parser.add_argument("--warm_down_ratio", default=0.5, type=float, help="lr warmdown ratio")
parser.add_argument("--final_lr_frac", default=0., type=float, help="min lr fraction")
# lr 默认继承预训练（和 nanochat 一样）；显式指定时按 pretrain 的规则乘 sqrt(batch_token/B_REF)
parser.add_argument("--lmhead_lr",default=None,type=float,help="lmhead learning rate (default: inherit from pretrain)")
parser.add_argument("--embedding_lr",default=None,type=float,help="learning rate for embedding (default: inherit from pretrain)")
parser.add_argument("--matrix_lr",default=None,type=float,help="matrix learning rate (default: inherit from pretrain)")
# model config
parser.add_argument("--seq_len", default=2048, type=int, help="model sequence len (same as pretrain, like nanochat)")
from tasks import sftdata
from tokenizer import tokenizer
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
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


def build_train_mixture():
    mmlu=sftdata.MMLU("auxiliary_train") # 100K rows per epoch
    gsm8k=sftdata.GSM8K("train") # 7.5K rows per epoch
    tasks=[sftdata.SmolTalk("train")] # 460K rows of general conversations
    tasks+=[mmlu]*args.mmlu_epochs
    tasks+=[gsm8k]*args.gsm8k_epochs
    if args.toolcall_epochs > 0:
        toolcall=sftdata.HermesFunctionCalling() # 7K rows of multi-turn function calling
        tasks+=[toolcall]*args.toolcall_epochs
    if args.toolcall_synth_epochs > 0:
        tasks+=[sftdata.CustomJSON(args.toolcall_synth_path)]*args.toolcall_synth_epochs
    if args.identity_epochs > 0:
        identity=sftdata.CustomJSON(args.identity_path)
        tasks+=[identity]*args.identity_epochs
    if args.identity_qa_epochs > 0:
        tasks+=[sftdata.CustomJSON(args.identity_qa_path)]*args.identity_qa_epochs
    return sftdata.TaskMixture(tasks)


def build_eval_mixture():
    return sftdata.TaskMixture([
        sftdata.SmolTalk("test"), # 24K rows
        sftdata.MMLU("test", stop=5200), # 14K rows in test, use 5.2K to match the train ratios
        sftdata.GSM8K("test", stop=420), # 1.32K rows in test, use 420 to match the train ratios
    ])


resume_ckpt=None
train_resume_state=None
if args.resume!=-1:
    with open(args.save_ckpt_path,"rb") as f:
        resume_ckpt=torch.load(f,map_location="cpu",weights_only=False)
    assert resume_ckpt.get("num_workers") == args.num_workers, "resume 时 num_workers 必须与保存时一致"
    train_resume_state=resume_ckpt['train_loader_state_dict']
    if ddp:
        train_resume_state=train_resume_state[rank]

train_mixture=build_train_mixture()
if is_main:
    print(f"SFT train mixture: {len(train_mixture):,} rows "
          f"(MMLU x{args.mmlu_epochs}, GSM8K x{args.gsm8k_epochs}, "
          f"toolcall x{args.toolcall_epochs}, toolcall_synth x{args.toolcall_synth_epochs}, "
          f"identity x{args.identity_epochs}, identity_qa x{args.identity_qa_epochs})")
train_data_set=sftdata.SFTDataset(
    train_mixture, tokeniz, args.device_batch_size, args.seq_len,
    rank=rank, world_size=world_size, resume_state_dict=train_resume_state,
)
eval_data_set=None
if is_main and (args.final_eval or args.eval_every > 0):
    eval_data_set=sftdata.SFTDataset(
        build_eval_mixture(), tokeniz, args.device_batch_size, args.seq_len,
    )
if is_main:
    ESTIMATE_TOTAL_TOKEN=train_data_set.estimate_total_token()
else:
    ESTIMATE_TOTAL_TOKEN=0
if ddp:
    total_token_holder=[ESTIMATE_TOTAL_TOKEN]
    dist.broadcast_object_list(total_token_holder, src=0)
    ESTIMATE_TOTAL_TOKEN=total_token_holder[0]
loader_kwargs = dict(batch_size=None, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))
if args.num_workers > 0:
    loader_kwargs["prefetch_factor"] = 4
train_data_loader=iter(DataLoader(train_data_set,**loader_kwargs))
loader_state=dict(train_resume_state or {})
SCALE_RATIO=1 #to make the progress more quick
B_REF=2**19 # 与 pretrain.py 的 B_REF 保持一致
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
    if resume_ckpt is not None:
        run=wandb.init(entity="enyuwang1023097618-123",project="mygpt",config=config_all,resume='allow',id=resume_ckpt['runid'])
    else:
        run=wandb.init(entity="enyuwang1023097618-123",project="mygpt",config=config_all)


# resume 时从 SFT checkpoint 恢复模型和优化器，否则从预训练 checkpoint 开始
if resume_ckpt is not None:
    resume_dict=resume_ckpt
else:
    with open(args.pretrained_ckpt_path,"rb") as f:
        resume_dict=torch.load(f,map_location='cpu')
new_dict=common.strip_prefix(resume_dict['model_state'])
config=gpt.GPTConfig(**resume_dict['model_config'])
config.seq_len=args.seq_len
orig_gpt=gpt.GPT(config,device,tokeniz)
orig_gpt.init_weight()
orig_gpt.load_state_dict(new_dict)
# lr/wd 等超参数随后由 checkpoint 里的 param_groups 覆盖
param_groups=orig_gpt.get_param_groups()
optimiz=optimizer.MuonOptimizer(param_groups)
optimizer_state=resume_dict['optimizer_state_dict']
if isinstance(optimizer_state, list):
    if len(optimizer_state) != world_size:
        raise ValueError(
            "checkpoint optimizer state was saved for "
            f"{len(optimizer_state)} ranks, but this run uses {world_size}"
        )
    optimizer_state=optimizer_state[rank]
elif ddp and resume_ckpt is not None:
    raise ValueError("SFT checkpoint was saved by a single process, resume it without torchrun")
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
if resume_ckpt is None:
    # 预训练保存的 init_lr = 配置 lr * sqrt(pretrain_batch_token/B_REF)，SFT batch 不同时再按 sqrt(B) 换算
    pretrain_batch_token=resume_dict.get("batch_token", args.batch_token)
    batch_scale=(args.batch_token/pretrain_batch_token)**0.5
    override_scale=(args.batch_token/B_REF)**0.5
    # 显式指定的 lr 也要乘 get_param_groups 里的 1/sqrt(d_model) 缩放
    dmodel_lr_scale=(config.model_dim/768)**-0.5
    value_embed_params={id(p) for p in orig_gpt.value_embeds.parameters()}
    for group in optimiz.param_groups:
        first=group['params'][0]
        if group['kind']=='muon':
            override=None if args.matrix_lr is None else args.matrix_lr
        elif first is orig_gpt.lm_head.weight:
            override=None if args.lmhead_lr is None else args.lmhead_lr*dmodel_lr_scale
        elif first is orig_gpt.transformer.wte.weight:
            override=None if args.embedding_lr is None else args.embedding_lr*dmodel_lr_scale
        elif id(first) in value_embed_params:
            override=None if args.embedding_lr is None else args.embedding_lr*dmodel_lr_scale*0.5
        else:
            override=None # scalars always follow pretrain
        base_lr=override*override_scale if override is not None else group['init_lr']*batch_scale
        group['init_lr']=base_lr*args.init_lr_frac
        group['lr']=group['init_lr']
        # 预训练末尾 wd 已经衰减，SFT 不再使用 weight decay（和 nanochat 一致）
        group['wd']=0.
        group['init_wd']=0.
    if is_main:
        print("SFT init lr:", [f"{g['kind']}:{g['init_lr']:.5f}" for g in optimiz.param_groups])
else:
    # 保存的 param_groups 里 init_lr 已经乘过 init_lr_frac，不能再缩放一次
    start_iter=resume_ckpt['step']+1
    best_bpb=resume_ckpt['best_bpb']
    if is_main:
        print(f"resumed from {args.save_ckpt_path}, continue at step {start_iter}/{total_steps}")
del param_groups
del new_dict
del resume_dict
resume_ckpt=None


test_data_loader = lambda: DataLoader(eval_data_set,batch_size=None,num_workers=0)
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

def save_model(step,train_loader_state_dict,bpb=None):
    """step 是已经完成的最后一个 optimizer step，resume 时从 step+1 继续（和 pretrain 一致）。"""
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
                "step":step,
                "best_bpb":best_bpb,
                "runid":run.id if args.use_wandb else None,
                "num_workers":args.num_workers
            },tmp
        )
        os.replace(tmp,args.save_ckpt_path)
        print("model saved")
    if ddp:
        dist.barrier()
max_dump_save=3
for now_step in range(start_iter,total_steps):
    bpb=None
    if args.eval_every > 0 and now_step%args.eval_every==0 and now_step!=start_iter:
        if is_main:
            bpb=bpbeval.bpb_eval(orig_gpt,eval_step,test_data_loader(),token_bytes,device,segment=True)
            val_log_data={"val/bpb":bpb}
            print(val_log_data)
            if args.use_wandb:
                run.log(data=val_log_data,step=now_step)
        if ddp:
            dist.barrier()
    if args.sample_every > 0 and now_step%args.sample_every==0 and now_step!=start_iter:
        if is_main:
            sample.sample_chat(orig_gpt,tokeniz,args.seq_len,device)
        if ddp:
            dist.barrier()
    if args.save_every > 0 and now_step%args.save_every==0 and now_step!=start_iter:
        # 在本 step 训练之前保存，所以已完成的是 now_step-1
        save_model(now_step-1,loader_state,bpb)
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
        x,y,segment_batch,data_loader_state=next(train_data_loader)
        loader_state[data_loader_state['worker']]=data_loader_state
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        segment_batch = segment_batch.to(device, non_blocking=True)
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
        if pbar.disable and (now_step%10==0 or now_step==start_iter):
            # progress bar is off when logging to a file
            print(f"step {now_step}/{total_steps}: time used {usedtime:.2f}, gpu usage {optimal_time/usedtime:.3f}, loss {total_loss.item():.4f}, lr_ratio {lr_ratio:.3f}",flush=True)

save_model(total_steps-1,loader_state)
if args.use_wandb and is_main:
    run.finish()
if is_main:
    pbar.close()
if ddp:
    dist.destroy_process_group()
