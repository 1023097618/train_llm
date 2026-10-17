#!/usr/bin/env bash
# nohup setsid bash run.sh > log/train.log 2>&1 &
# ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E "run.sh|script\." | grep -v grep
# kill -- -12345
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
python -m script.download_pretrain
python -m script.train_token
# python -m script.eval_token
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 -m script.pretrain

python -m script.download_sft
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --standalone --nproc_per_node=4 -m script.train_sft
