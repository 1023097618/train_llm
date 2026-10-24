#!/usr/bin/env bash
# nohup setsid bash run_single_gpu_32g.sh > log/d20.log 2>&1 &
# ps -eo pid,ppid,pgid,stat,etime,cmd | grep -E "run_single_gpu_32g.sh|script\." | grep -v grep
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
# 每个配置的 tokenizer 和权重放在各自目录，可以和其他 run_*.sh 同时运行
WEIGHT_DIR=weight/d20
TOKENIZER=$WEIGHT_DIR/tokenizer.pkl
PRETRAIN_CKPT=$WEIGHT_DIR/pretrained_model.ckpt
SFT_CKPT=$WEIGHT_DIR/sft_model.ckpt
RL_CKPT=$WEIGHT_DIR/rl_model.ckpt
REPORT=$WEIGHT_DIR/report
mkdir -p "$WEIGHT_DIR" log

# ---- data ----
python -m script.download_pretrain --num_shards=180
python -m script.download_sft
python -m script.download_eval

# ---- tokenizer ----
python -m script.train_token --save_path=$TOKENIZER
python -m script.eval_token --tokenizer_path=$TOKENIZER

# ---- pretrain, then CORE / bpb (like nanochat base_eval) ----
python -m script.pretrain --depth=20 --device_batch_size=4 --tokenizer_path=$TOKENIZER --save_ckpt_path=$PRETRAIN_CKPT
python -m script.base_eval --ckpt_path=$PRETRAIN_CKPT --tokenizer_path=$TOKENIZER --report_dir=$REPORT

# ---- SFT, then chat eval (ARC / MMLU / GSM8K / HumanEval / ChatCORE) ----
python -m script.train_sft --device_batch_size=4 --tokenizer_path=$TOKENIZER --pretrained_ckpt_path=$PRETRAIN_CKPT --save_ckpt_path=$SFT_CKPT \
    --identity_path=data/raw/custom/identity_conversations_en_d20.jsonl --identity_qa_path=data/raw/custom/identity_qa_d20.jsonl
python -m script.chat_eval --ckpt_path=$SFT_CKPT --tokenizer_path=$TOKENIZER --stage=sft --report_dir=$REPORT

# ---- RL on GSM8K (pass@k at the start and the end), then chat eval again ----
python -m script.chat_rl --ckpt_path=$SFT_CKPT --tokenizer_path=$TOKENIZER --save_ckpt_path=$RL_CKPT --report_dir=$REPORT
python -m script.chat_eval --ckpt_path=$RL_CKPT --tokenizer_path=$TOKENIZER --stage=rl --report_dir=$REPORT

python -m script.report --report_dir=$REPORT
