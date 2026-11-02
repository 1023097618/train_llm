#!/usr/bin/env bash
# d24 conversational image generation: the chat model learns when to draw and how to write the description, on top of the
# VLM + RL model, with text replay so the other abilities stay. vlm_rl.ckpt is not touched: the image decoder keeps reading
# captions with it. Benchmarks after training show what the extra round cost.
# python -m datagen.synth_t2i_chat, then script.render_chat_images (the pictures the model will see) first.
# GPUS=1,2 setsid nohup bash run_d24_t2i_chat.sh >> log/d24_t2i_chat.log 2>&1 < /dev/null &
# Rerunning the same command after a kill continues where it stopped.
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

GPUS=${GPUS:-1,2}
EVAL_GPU=${GPUS%%,*}
WEIGHT_DIR=weight/d24
TOKENIZER=$WEIGHT_DIR/tokenizer.pkl
BASE_CKPT=$WEIGHT_DIR/vlm_rl.ckpt
CKPT=$WEIGHT_DIR/chat_t2i_v7.ckpt
REPORT=$WEIGHT_DIR/report_t2i_chat_v7
mkdir -p "$REPORT" log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wandb: the server's direct connection gets 403, offline when the forwarded proxy is down
if curl -s -o /dev/null --max-time 10 -x http://127.0.0.1:7890 https://api.wandb.ai/healthz; then
    export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1
else
    export WANDB_MODE=offline
fi

if [ ! -f "$CKPT.done" ]; then
    resume=""
    [ -f "$CKPT.state" ] && resume="--resume=$CKPT.state"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$(echo "$GPUS" | tr ',' '\n' | wc -l) -m script.train_vlm --stage=sft \
        --ckpt_path=$BASE_CKPT --tokenizer_path=$TOKENIZER --save_ckpt_path=$CKPT $resume \
        --t2i_chat_path=data/raw/custom/t2i_chat_text.jsonl --t2i_chat_epochs=2 --caption_weight=0.1 --instruct_weight=0.25 --text_weight=0.45 --chat_weight=0.2 \
        --llm_lr_frac=0.07 --max_steps=1600
    touch "$CKPT.done"
fi
unset HTTPS_PROXY HTTP_PROXY NO_PROXY
[ -f "$REPORT/t2i_chat_behaviour.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.t2i_chat_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --report_dir=$REPORT
[ -f "$REPORT/t2i_chat_vlm.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --stage=t2i_chat_vlm --report_dir=$REPORT
[ -f "$REPORT/t2i_chat.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --stage=t2i_chat --report_dir=$REPORT
