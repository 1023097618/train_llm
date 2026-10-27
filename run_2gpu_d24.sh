#!/usr/bin/env bash
# d24 (nanochat's GPT-2 grade reference size) on two 32GB GPUs
#   pretrain + SFT: torchrun on both GPUs; RL: vLLM + reference model on the first GPU, training on the second;
#   CORE / chat eval: first GPU
# setsid nohup bash run_2gpu_d24.sh > log/d24.log 2>&1 < /dev/null &
# ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E "run_2gpu_d24.sh|script\." | grep -v grep
# kill -- -<pgid>
set -e

# 找到当前用户的 conda（nohup 下不一定加载 .bashrc，PATH 里可能没有 conda）
if ! command -v conda >/dev/null 2>&1; then
    for p in "$HOME/miniconda3" "$HOME/anaconda3" /home/vislab/anaconda3; do
        [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
    done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY
cd "$(dirname "$0")"

GPUS=0,4           # pretrain / SFT on both
EVAL_GPU=0         # single process stages
DEPTH=24
# tokens = PARAM_RATIO * 0.78B scaling params; 8 like nanochat's d24 speedrun (~6.2B tokens, ~35h on 2x5090)
PARAM_RATIO=8
WEIGHT_DIR=weight/d24
TOKENIZER=$WEIGHT_DIR/tokenizer.pkl
PRETRAIN_CKPT=$WEIGHT_DIR/pretrained_model.ckpt
SFT_CKPT=$WEIGHT_DIR/sft_model.ckpt
RL_CKPT=$WEIGHT_DIR/rl_model.ckpt
REPORT=$WEIGHT_DIR/report
mkdir -p "$WEIGHT_DIR" log
# d24 fills a 32GB card: device batch 2 is the largest that fits, expandable segments avoid fragmentation OOMs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCHRUN="torchrun --standalone --nproc_per_node=2"

# ---- data ----
python -m script.download_pretrain --num_shards=180
python -m script.download_sft
python -m script.download_eval
python -m script.prepare_rl_data  # RL questions the SFT model has not seen (OpenMathInstruct-2 augmented_gsm8k)

# ---- tokenizer ----
python -m script.train_token --save_path=$TOKENIZER
python -m script.eval_token --tokenizer_path=$TOKENIZER

# ---- pretrain (2 GPUs), then CORE / bpb ----
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN -m script.pretrain --depth=$DEPTH --param_ratio=$PARAM_RATIO --device_batch_size=2 \
    --tokenizer_path=$TOKENIZER --save_ckpt_path=$PRETRAIN_CKPT
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.base_eval --ckpt_path=$PRETRAIN_CKPT --tokenizer_path=$TOKENIZER --report_dir=$REPORT

# ---- SFT (2 GPUs, the pretrain optimizer state is sharded for 2 ranks), then chat eval ----
CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN -m script.train_sft --device_batch_size=2 --tokenizer_path=$TOKENIZER \
    --pretrained_ckpt_path=$PRETRAIN_CKPT --save_ckpt_path=$SFT_CKPT \
    --identity_path=data/raw/custom/identity_conversations_en_d24.jsonl --identity_qa_path=data/raw/custom/identity_qa_d24.jsonl
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$SFT_CKPT --tokenizer_path=$TOKENIZER --stage=sft --report_dir=$REPORT

# ---- RL, then chat eval ----
# vLLM + the frozen SFT reference model on the first GPU, training model on the second (d24 + optimizer state + vLLM
# do not fit one GPU). Trained on GSM8K style questions the SFT model has not seen (GSM8K train is in the SFT mixture
# 4 times: RL on it barely moved GSM8K test), KL penalty to the SFT model + dynamic sampling against the collapse
# seen without them. d24: GSM8K 0.124 -> 0.178 (best pass@1 at step 180), the other chat tasks unchanged.
# Saves a resumable state every 30 steps; after a kill, run the chat_rl command by hand with --resume=$RL_CKPT.state
CUDA_VISIBLE_DEVICES=$GPUS python -m script.chat_rl --ckpt_path=$SFT_CKPT --tokenizer_path=$TOKENIZER --save_ckpt_path=$RL_CKPT \
    --report_dir=$REPORT --device_batch_size=4 --device=cuda:1 --ref_device=cuda:0 --gpu_memory_utilization=0.6 \
    --kl_beta=0.04 --oversample=2.5 --eval_every=30 --train_data=data/raw/rl/omi_gsm.jsonl --max_steps=233
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$RL_CKPT --tokenizer_path=$TOKENIZER --stage=rl --report_dir=$REPORT

python -m script.report --report_dir=$REPORT
