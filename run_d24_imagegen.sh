#!/usr/bin/env bash
# d24 image generation: decoder over VQ tokens, captions read by the frozen chat model
# python -m script.prepare_t2i_data first (data/raw/t2i: caption + VQ token parquet, LlamaGen VQ-16)
# GPU=2 setsid nohup bash run_d24_imagegen.sh >> log/d24_imagegen.log 2>&1 < /dev/null &
# Rerunning the same command after a kill continues from the training state.
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

GPU=${GPU:-2}
CKPT=weight/d24/imagegen.ckpt
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wandb: the server's direct connection gets 403, offline when the forwarded proxy is down
if curl -s -o /dev/null --max-time 10 -x http://127.0.0.1:7890 https://api.wandb.ai/healthz; then
    export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1
else
    export WANDB_MODE=offline
fi
resume=""
[ -f "$CKPT.state" ] && resume="--resume=$CKPT.state"
CUDA_VISIBLE_DEVICES=$GPU python -m script.train_imagegen --thinker_ckpt=weight/d24/vlm_rl.ckpt \
    --tokenizer_path=weight/d24/tokenizer.pkl --save_ckpt_path=$CKPT $resume
