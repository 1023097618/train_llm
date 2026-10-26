#!/usr/bin/env bash
# d24 RL with a KL penalty to the SFT model + dynamic sampling, then chat eval and report
# writes next to the plain RL run (rl_model.ckpt / report) so the two can be compared
# setsid nohup bash run_d24_rl_kl.sh > log/d24_rl_kl.log 2>&1 < /dev/null &
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
RL_CKPT=$WEIGHT_DIR/rl_kl_model.ckpt
REPORT=$WEIGHT_DIR/report_kl
mkdir -p "$REPORT"
cp $WEIGHT_DIR/report/base.json $WEIGHT_DIR/report/sft.json "$REPORT"/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=$RL_GPUS python -m script.chat_rl --ckpt_path=$SFT_CKPT --tokenizer_path=$TOKENIZER --save_ckpt_path=$RL_CKPT \
    --report_dir=$REPORT --device_batch_size=4 --device=cuda:1 --ref_device=cuda:0 --gpu_memory_utilization=0.6 \
    --kl_beta=0.04 --oversample=2 --eval_every=30
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$RL_CKPT --tokenizer_path=$TOKENIZER --stage=rl --report_dir=$REPORT
python -m script.report --report_dir=$REPORT
