#!/usr/bin/env bash
# d24 conversational image generation: the chat model learns when to draw, how to write the description and to see its pictures,
# with text replay so the other abilities stay. This round is short, on top of chat_t2i_v7: it brings back who the model is, which
# the earlier rounds of image training never replayed (those trained from vlm_rl.ckpt with 1600 steps at --llm_lr_frac=0.07).
# Benchmarks after training show what the extra round cost.
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
BASE_CKPT=${BASE_CKPT:-$WEIGHT_DIR/chat_t2i_v7.ckpt}
CKPT=${CKPT:-$WEIGHT_DIR/chat_t2i_v8.ckpt}
REPORT=${REPORT:-$WEIGHT_DIR/report_t2i_chat_v8}
D=data/raw/custom
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
        --identity_paths=$D/identity_conversations_en_d24.jsonl:3,$D/identity_qa_d24.jsonl:10,$D/identity_direct_d24.jsonl:15 \
        --llm_lr_frac=0.03 --max_steps=300 --eval_every=50 --save_every=50
    touch "$CKPT.done"
fi
unset HTTPS_PROXY HTTP_PROXY NO_PROXY
[ -f "$REPORT/t2i_chat_behaviour.json" ] || HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/train_llm_pylibs/image CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.t2i_chat_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --report_dir=$REPORT
[ -f "$REPORT/t2i_chat_vlm.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --stage=t2i_chat_vlm --report_dir=$REPORT
[ -f "$REPORT/t2i_chat.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --stage=t2i_chat --report_dir=$REPORT
