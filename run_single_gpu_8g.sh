#!/usr/bin/env bash
# nohup setsid bash run_single_gpu_8g.sh > log/d12.log 2>&1 &
# ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E "run_single_gpu_8g.sh|script\." | grep -v grep
# kill -- -12345
set -e

source /home/vislab/anaconda3/etc/profile.d/conda.sh
conda activate gpt
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY
# 每个配置的 tokenizer 和权重放在各自目录，可以和其他 run_*.sh 同时运行
WEIGHT_DIR=weight/d12
TOKENIZER=$WEIGHT_DIR/tokenizer.pkl
PRETRAIN_CKPT=$WEIGHT_DIR/pretrained_model.ckpt
SFT_CKPT=$WEIGHT_DIR/sft_model.ckpt
mkdir -p "$WEIGHT_DIR" log

python -m script.download_pretrain
python -m script.train_token --save_path=$TOKENIZER
python -m script.eval_token --tokenizer_path=$TOKENIZER
python -m script.pretrain --tokenizer_path=$TOKENIZER --save_ckpt_path=$PRETRAIN_CKPT

python -m script.download_sft
python -m script.train_sft --tokenizer_path=$TOKENIZER --pretrained_ckpt_path=$PRETRAIN_CKPT --save_ckpt_path=$SFT_CKPT \
    --identity_path=data/raw/custom/identity_conversations_en_d12.jsonl
