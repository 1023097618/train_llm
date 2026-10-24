"""
pretrain a language model
before run this script, you need download data files(script.download_pretrain, 14.6 GB) and train a tokenizer(script.train_token)
uasge:
python -m script.pretrain --use_wandb=0
or
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 -m script.pretrain --use_wandb=0
"""
import sys
sys.stdout.reconfigure(line_buffering=True)  # show progress in nohup logs immediately


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from model import gpt
from tasks import pretraindata
import time
import math
from evaluation import bpbeval
import tqdm
import argparse
from common.data_paths import PRETRAIN_T2T
from tokenizer import tokenizer
from common import common
import os
from optimizer import optimizer
parser=argparse.ArgumentParser(description="Pretrain base model")


#train config
parser.add_argument("--device_batch_size",default=4,type=int,help="batch size per gradient step")
parser.add_argument("--param_ratio",default=12,type=int,help="(the token we need) = (number of parameters in the model)* (param ratio)")
parser.add_argument("--batch_token",default=-1,type=int,help="how many token should have in one step (-1: auto, nanochat Power Lines rule B ∝ D^0.383)")
parser.add_argument("--eval_token",default=80*2048,type=int,help="how many token we use for eval")
parser.add_argument("--eval_every",default=250,type=int,help="evaluate the model per eval_every steps")
parser.add_argument("--sample_every",default=500,type=int,help="sample the model per sample_every steps")
parser.add_argument("--save_every",default=500,type=int,help="save the model per save_every steps")
parser.add_argument("--max_steps",default=0,type=int,help="limit optimizer steps when greater than zero; 0 keeps the calculated total")
parser.add_argument("--resume",default=-1,type=int,help="resume the model (-1 is disable resume)")
parser.add_argument("--tokenizer_path",default="weight/tokenizer.pkl",type=str,help="path of tokenizer.pkl, use train_token.py to get tokenizer")
parser.add_argument("--device_type",default="cuda",choices=("cuda","cpu"),help="cuda or cpu")
parser.add_argument("--use_wandb",type=int,default=1,help="if use wandb to log(-1 to disable wandb)")
parser.add_argument("--data_dir", type=str, default=str(PRETRAIN_T2T), help="processed pretrain dataset dir")
parser.add_argument("--save_ckpt_path",type=str,default="weight/pretrained_model.ckpt",help="path to save model checkpoint")
parser.add_argument("--num_workers",type=int,default=8,help="dataloader worker number")
#hyper parameter
parser.add_argument("--lr_warmup_iter",default=40,type=int,help="lr warmup iter")
parser.add_argument("--warmdown_ratio",default=0.65,type=float,help="lr warmdown ratio")
parser.add_argument("--lr_final",default=0.05,type=float,help="min lr")
parser.add_argument("--lmhead_lr",default=0.008,type=float,help="lmhead learning rate")
parser.add_argument("--embedding_lr",default=0.3,type=float,help="learning rate for embedding")
parser.add_argument("--weight_decay",default=0.28,type=float,help="weight decay")
parser.add_argument("--min_weight_decay_ratio",default=0.,type=float,help="min weight decay ratio (0: cosine decay to zero like nanochat)")
parser.add_argument("--matrix_lr",default=0.02,type=float,help="matrix learning rate")
parser.add_argument("--scalar_lr",default=0.5,type=float,help="learning rate for scalars (resid_lambdas, x0_lambdas)")
#model config
parser.add_argument("--seq_len",default=2048,type=int,help="model sequence len")
parser.add_argument("--window_pattern",default="SSSL",type=str,help="sliding window pattern tiled across layers, L=full context, S=quarter context")
parser.add_argument("--depth",type=int,default=12,help="number of blocks of the model; model_dim and head num are derived from it (nanochat style)")
parser.add_argument("--vocab_size",type=int,default=65536,help="vocab size of the model")
parser.add_argument("--kv_num",type=int,default=None,help="kv head of the model (default: same as q head num, set smaller to use GQA)")

args=parser.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(args.save_ckpt_path)), exist_ok=True)
config_all = vars(args).copy()
if args.use_wandb:
    import wandb

ddp=all(name in os.environ for name in ('LOCAL_RANK','RANK','WORLD_SIZE'))
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
if ddp:
    local_rank=int(os.getenv("LOCAL_RANK",0))
    # NCCL object collectives use current_device(), not the model tensor device.
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    rank=dist.get_rank()
    world_size=dist.get_world_size()
    device=f"cuda:{local_rank}"
else:
    rank=0
    local_rank=0
    world_size=1
    if args.device_type=='cuda':
        device="cuda:0"
    else:
        device="cpu"
is_main=(rank==0)
if device.startswith("cuda"):
    device_name=torch.cuda.get_device_name(int(local_rank))
    OPTIMAL_FLOPS=common.get_peak_flops(device_name)*world_size
else:
    OPTIMAL_FLOPS=float('inf')

prompts=[
    "The capital of France is",
    "The chemical symbol of gold is",
    "If yesterday was Friday, then tomorrow will be",
    "The opposite of hot is",
    "The planets of the solar system are:",
    "My favorite color is",
    "If 5*x + 3 = 13, then x is",
]
# prompts=[
#     "法国的首都是",
#     "金的化学表达式是",
#     "如果昨天是周五，那么今天是",
#     "热的反义词是",
#     "太阳系的行星是",
#     "我最爱的颜色是",
#     "如果5*x+3=13，那么x是",
#     "王恩宇是一个"
# ]
tokeniz=tokenizer.Tokenizer.from_dict(args.tokenizer_path,device=device)
token_bytes=tokeniz.get_token_bytes()
config=gpt.GPTConfig.from_depth(args.depth,kv_num=args.kv_num,seq_len=args.seq_len,vocab_size=args.vocab_size,window_pattern=args.window_pattern)
orig_gpt=gpt.GPT(config,device,tokeniz)
compiled_gpt=torch.compile(orig_gpt,dynamic=False)

# scaling params: transformer matrices + lm_head give the cleanest scaling laws (nanochat)
P=orig_gpt.get_model_scale()['transformer_matrix']+orig_gpt.get_model_scale()['lm_head_count']
target_token=args.param_ratio*P        #token we need to train model   12*model_params
# nanochat style hyperparameter transfer: the default lr/wd are tuned on d12 with batch_token=B_REF
B_REF=2**19
with torch.device("meta"):
    ref_gpt=gpt.GPT(gpt.GPTConfig.from_depth(12,seq_len=args.seq_len,vocab_size=args.vocab_size),"meta",tokeniz)
ref_scale=ref_gpt.get_model_scale()
D_REF=args.param_ratio*(ref_scale['transformer_matrix']+ref_scale['lm_head_count'])
del ref_gpt
if args.batch_token==-1:
    # Power Lines (https://arxiv.org/abs/2505.13738): optimal batch grows as D^0.383, rounded to a power of 2
    args.batch_token=2**round(math.log2(B_REF*(target_token/D_REF)**0.383))
    config_all["batch_token"]=args.batch_token
if args.batch_token % args.seq_len != 0:
    raise ValueError(f"batch_token={args.batch_token} must be divisible by seq_len={args.seq_len}")
batch_size=args.batch_token//args.seq_len
per_step=args.device_batch_size*world_size
if batch_size < per_step or batch_size % per_step != 0:
    raise ValueError(
        f"batch_size={batch_size} (batch_token/seq_len) must be a positive multiple of "
        f"device_batch_size*world_size={args.device_batch_size}*{world_size}={per_step}"
    )
gradient_step=batch_size//per_step
num_iteration=int(target_token/args.batch_token)
if args.max_steps > 0:
    num_iteration=min(num_iteration,args.max_steps)
# Adam/Muon updates are normalized, so lr ∝ sqrt(B)
batch_lr_scale=(args.batch_token/B_REF)**0.5
# keep the AdamW timescale 1/(lr*wd) a fixed fraction of training: wd ∝ sqrt(B)/D
weight_decay_scaled=args.weight_decay*batch_lr_scale*(D_REF/target_token)
config_all.update(batch_lr_scale=batch_lr_scale,weight_decay_scaled=weight_decay_scaled)

# 只在 rank0 统计 token，再广播给其他 rank，避免多进程同时写缓存/重复统计
select_result=[None]
if is_main:
    try:
        # with --max_steps only the tokens we will actually train on need to exist
        select_result[0]=pretraindata.select_train_files(args.data_dir,tokeniz,min(target_token,num_iteration*args.batch_token))
    except RuntimeError as e:
        select_result[0]=str(e)
if ddp:
    dist.broadcast_object_list(select_result,src=0)
if isinstance(select_result[0],str):
    if ddp:
        dist.destroy_process_group()
    raise SystemExit(select_result[0])
train_files,train_token_total=select_result[0]
if is_main:
    print("model scale:",{k:f"{v:,}" for k,v in orig_gpt.get_model_scale().items()})
    print(f"use {len(train_files)} train files, {train_token_total:,} tokens >= {args.param_ratio} * {P:,} params")
    print(f"total steps: {num_iteration:,} (target {target_token:,} tokens / {args.batch_token:,} tokens per step, "
          f"gradient_step {gradient_step})")

test_dataset = pretraindata.PretrainDataset(
    args.data_dir, 'test', tokeniz, args.device_batch_size, args.seq_len
)
loader_kwargs = dict(batch_size=None, num_workers=args.num_workers, pin_memory=True)
if args.num_workers > 0:
    loader_kwargs["prefetch_factor"] = 4

test_loader_builder = lambda: DataLoader(test_dataset,batch_size=None, num_workers=0)
param_groups=orig_gpt.get_param_groups(
    lmhead_lr=args.lmhead_lr*batch_lr_scale,
    embedding_lr=args.embedding_lr*batch_lr_scale,
    matrix_lr=args.matrix_lr*batch_lr_scale,
    scalar_lr=args.scalar_lr*batch_lr_scale,
    weight_decay=weight_decay_scaled,
)
optimiz=optimizer.MuonOptimizer(param_groups)
flops_per_token=orig_gpt.get_flops_per_token()
batch_flops=flops_per_token*args.batch_token
optimal_time=batch_flops/OPTIMAL_FLOPS
eval_step=int(args.eval_token/(args.device_batch_size*args.seq_len))
if is_main:
    print(config_all)
if args.resume!=-1:
    with open(args.save_ckpt_path,"rb") as f:
        resume_dict=torch.load(f, map_location="cpu", weights_only=False)
        assert resume_dict.get("num_workers") == args.num_workers, "resume 时 num_workers 必须与保存时一致"
        orig_gpt.load_state_dict(resume_dict['model_state'])
        train_resume_state = resume_dict['train_loader_state_dict']
        optimizer_resume_state=resume_dict['optimizer_state_dict']
        if ddp:
            train_resume_state=train_resume_state[rank]
            optimizer_resume_state=optimizer_resume_state[rank]
        # Checkpoints are written after the recorded optimizer step completes.
        optimiz.load_state_dict(optimizer_resume_state)
        start_step=resume_dict['step']+1
        best_bpb=resume_dict['best_bpb']
        if args.use_wandb and is_main:
            run=wandb.init(entity="enyuwang1023097618-123",project="mygpt",config=config_all,resume='allow',id=resume_dict['runid'])
        print("resumed")
else:
    start_step=1
    train_resume_state = None
    if args.use_wandb and is_main:
        run=wandb.init(entity="enyuwang1023097618-123",project="mygpt",config=config_all)
    best_bpb=float('inf')
train_dataset = pretraindata.PretrainDataset(
    args.data_dir, 'train', tokeniz, batch_size//world_size, args.seq_len,
    resume_state_dict=train_resume_state,rank=rank,world_size=world_size,data_files=train_files
)
train_loader = iter(DataLoader(train_dataset,**loader_kwargs))
loader_state = dict(train_resume_state or {})
def get_lr_multiply(it):
    if it<=args.lr_warmup_iter:
        return it/args.lr_warmup_iter
    elif it<=num_iteration-num_iteration*args.warmdown_ratio:
        return 1.
    else:
        it_process=it-(num_iteration*(1-args.warmdown_ratio))
        num_iteration_warmdown=num_iteration*args.warmdown_ratio
        process=it_process/num_iteration_warmdown
        return process*args.lr_final+(1-process)

def get_momentum(it):
    if it<=400:
        process=it/400
        return process*0.97+(1-process)*0.85
    elif it<=num_iteration-num_iteration*args.warmdown_ratio:
        return 0.97
    else:
        it_process=it-(num_iteration*(1-args.warmdown_ratio))
        num_iteration_warmdown=num_iteration*args.warmdown_ratio
        process=it_process/num_iteration_warmdown
        return process*0.9+(1-process)*0.97

def get_wd(it):
    scale=0.5*(1+math.cos(math.pi*it/num_iteration))
    result= max(scale,args.min_weight_decay_ratio)*weight_decay_scaled
    return result

@torch.no_grad()
def sample():
    orig_gpt.eval()
    for i,prompt in enumerate(prompts):
        x=tokeniz.encode_ordinary(prompt)
        bos=tokeniz.special_token['<|doc|>']
        eos=tokeniz.special_token['<|/doc|>']
        x=[bos]+x
        x=torch.tensor(x,device=device).unsqueeze(0)
        kv_cache=orig_gpt.get_kv_cache(1)
        next_token=x.detach().clone()
        while x.shape[-1]<48:
            out:gpt.GPTOutPut=orig_gpt(next_token,kv_cache=kv_cache)
            target=out.logit[:,-1,:].squeeze(0)
            topk_logit,topk_index=torch.topk(target,k=10)
            topk_prob=torch.softmax(topk_logit,dim=-1)
            topk_prob_index=torch.multinomial(topk_prob,num_samples=1)
            next_token=topk_index[topk_prob_index].unsqueeze(0)
            x=torch.concatenate((x,next_token),dim=-1)
            if next_token[0,0].item()==eos:
                break
        pbar.write(f"step {step1}: prompt{i}/{len(prompts)} :\n{tokeniz.decode(x.tolist()[0])}")
    orig_gpt.train()

def save_model(step1,train_loader_state_dict,optimizer_states):
    global best_bpb
    bpb=bpbeval.bpb_eval(compiled_gpt,eval_step,test_loader_builder(),token_bytes,device,segment=True)
    if best_bpb>bpb:
        pbar.write(f"step {step1}: new best_bpb ({bpb:.4f}),model saved -> {args.save_ckpt_path}")
        best_bpb=bpb
        tmp=args.save_ckpt_path+".tmp"
        torch.save(
            {
                "model_state":orig_gpt.state_dict(),
                "optimizer_state_dict":optimizer_states,
                "train_loader_state_dict":train_loader_state_dict,
                "step":step1,
                "best_bpb":best_bpb,
                "runid":run.id if args.use_wandb else None,
                "model_config":vars(config),
                "num_workers": args.num_workers,
                "batch_token": args.batch_token
            },tmp
        ),
        os.replace(tmp,args.save_ckpt_path)
    else:
        pbar.write(f"step {step1}: model not saved, bpb {bpb:.4f} (bestbpb {best_bpb:.4f})")
if is_main:
    pbar = common.get_pbar()
for step1 in range(start_step,num_iteration+1):
    if step1%args.eval_every==0 and is_main:
        bpb=bpbeval.bpb_eval(compiled_gpt,eval_step,test_loader_builder(),token_bytes,device,segment=True)
        eval_log_data={"val/bpb":bpb}
        if args.use_wandb:
            run.log(data=eval_log_data,step=step1)
        pbar.write(f"[eval]step {step1}: bpb:{bpb}")
    if step1%args.sample_every==0 and is_main:
        sample()
    t1=time.perf_counter()
    total_loss=torch.tensor(0.,device=device)
    optimiz.zero_grad()
    lr_ratio=get_lr_multiply(step1)
    momentum=get_momentum(step1)
    wd=get_wd(step1)
    for g in optimiz.param_groups:
        g['lr']=g['init_lr']*lr_ratio
        if g['kind'] == 'muon':
            g['momentum1']=momentum
            g['wd']=wd
    try:
        x,y,segment_batch,data_loader_state=next(train_loader)
        loader_state[data_loader_state['worker']]=data_loader_state
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        segment_batch = segment_batch.to(device, non_blocking=True)
    except StopIteration:
        print("data insuficient, exit early")
        break
    for step2 in range(gradient_step):
        batch_slice = slice(step2 * args.device_batch_size, (step2 + 1) * args.device_batch_size)
        output = compiled_gpt(x[batch_slice], y[batch_slice], segment_ids=segment_batch[batch_slice])
        loss = output.loss
        loss/=gradient_step
        loss.backward()
        total_loss+=loss.detach()
    optimiz.step()
    if ddp:
        dist.all_reduce(total_loss,op=dist.ReduceOp.AVG)
    if step1%args.save_every==0:
        optimizer_state=optimiz.state_dict()
        if ddp:
            data_loader_states = [None for _ in range(world_size)] if is_main else None
            optimizer_states = [None for _ in range(world_size)] if is_main else None
            dist.gather_object(loader_state,data_loader_states)
            dist.gather_object(optimizer_state,optimizer_states)
        else:
            data_loader_states=loader_state
            optimizer_states=optimizer_state
        if is_main:
            save_model(step1,data_loader_states,optimizer_states)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    t4=time.perf_counter()
    usedtime=t4-t1
    train_log_data={"train/total_time_use":usedtime,
                    "train/total_loss":total_loss.item(),
                    "train/mfu":optimal_time/usedtime}
    if args.use_wandb and is_main:
        run.log(data=train_log_data,step=step1)
    if is_main:
        pbar.n=(step1/num_iteration)*100
        pbar.refresh()
        pbar.set_postfix_str(f"step {step1}: time use:{usedtime:.2f} loss:{total_loss.item():.2f} mfu:{optimal_time/usedtime:.2f}")
        if pbar.disable and (step1%10==0 or step1==start_step):
            # progress bar is off when logging to a file
            print(f"step {step1}/{num_iteration}: time use:{usedtime:.2f} loss:{total_loss.item():.4f} mfu:{optimal_time/usedtime:.3f} lr_ratio:{lr_ratio:.3f}",flush=True)
optimizer_state=optimiz.state_dict()
if ddp:
    data_loader_states = [None for _ in range(world_size)] if is_main else None
    optimizer_states = [None for _ in range(world_size)] if is_main else None
    dist.gather_object(loader_state,data_loader_states)
    dist.gather_object(optimizer_state,optimizer_states)
else:
    data_loader_states=loader_state
    optimizer_states=optimizer_state
if is_main:
    save_model(step1,data_loader_states,optimizer_states)
    if args.use_wandb:
        run.finish()
    pbar.close()
if ddp:
    dist.destroy_process_group()
