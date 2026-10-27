#!/usr/bin/env bash
# d24 RL on OpenMathInstruct-2 augmented_gsm8k questions (unseen by SFT) with KL penalty + dynamic sampling
# python -m script.prepare_rl_data first; GSM8K test is still the eval set
# setsid nohup bash run_d24_rl_omi.sh > log/d24_rl_omi.log 2>&1 < /dev/null &
# after it was killed, continue from the last saved training state (every 30 steps):
# RESUME=weight/d24/rl_omi_model.ckpt.state setsid nohup bash run_d24_rl_omi.sh >> log/d24_rl_omi.log 2>&1 < /dev/null &
set -e

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

RL_GPUS=3,0        # vLLM + frozen reference model, training model
EVAL_GPU=0
WEIGHT_DIR=weight/d24
TOKENIZER=$WEIGHT_DIR/tokenizer.pkl
SFT_CKPT=$WEIGHT_DIR/sft_model.ckpt
RL_CKPT=$WEIGHT_DIR/rl_omi_model.ckpt
REPORT=$WEIGHT_DIR/report_omi
mkdir -p "$REPORT"
cp $WEIGHT_DIR/report/base.json $WEIGHT_DIR/report/sft.json "$REPORT"/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RESUME=${RESUME:-}  # training state file to continue from, empty = start from the SFT model

CUDA_VISIBLE_DEVICES=$RL_GPUS python -m script.chat_rl --ckpt_path=$SFT_CKPT --tokenizer_path=$TOKENIZER --save_ckpt_path=$RL_CKPT \
    --report_dir=$REPORT --device_batch_size=4 --device=cuda:1 --ref_device=cuda:0 --gpu_memory_utilization=0.6 \
    --kl_beta=0.04 --oversample=2.5 --eval_every=30 \
    --train_data=data/raw/rl/omi_gsm.jsonl --max_steps=233 --resume="$RESUME"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$RL_CKPT --tokenizer_path=$TOKENIZER --stage=rl --report_dir=$REPORT
python -m script.report --report_dir=$REPORT
