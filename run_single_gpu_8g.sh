#!/usr/bin/env bash
# nohup setsid bash run.sh > log/train.log 2>&1 &
# ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E "run.sh|script\." | grep -v grep
# kill -- -12345
set -e

source /home/vislab/anaconda3/etc/profile.d/conda.sh
conda activate gpt
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY
python -m script.download_pretrain
python -m script.train_token
python -m script.eval_token
python -m script.pretrain

python -m script.download_sft
python -m script.train_sft
