#!/usr/bin/env bash
# d24 conversational image generation, on top of the image understanding model vlm_rl.ckpt (run_d24_vlm_rl.sh):
# the chat model learns when to draw, how to write and change the description, and to see the pictures it drew, with text and
# image replay so the other abilities stay. Two rounds:
#   1. chat_t2i_v7: 1600 steps from vlm_rl.ckpt on the drawing conversations and their pictures
#   2. chat_t2i_v9 (final): 300 steps from chat_t2i_v7 with identity data at a quarter of the learning rate, which brings
#      back who the model is; the last step is kept, the validation loss does not measure what this round is for
# Data first, either the published one (python -m script.download_release --what=data) or made again:
#   DEEPSEEK_API_KEY=... python -m datagen.synth_t2i_chat
#   script.render_chat_images (both shards, then --merge=1) and datagen.label_chat_images, see their docstrings
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
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
WEIGHT_DIR=weight/d24
TOKENIZER=$WEIGHT_DIR/tokenizer.pkl
D=data/raw/custom
REPORT=$WEIGHT_DIR/report_t2i_chat
mkdir -p "$REPORT" log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# wandb: the server's direct connection gets 403, offline when the forwarded proxy is down
if curl -s -o /dev/null --max-time 10 -x http://127.0.0.1:7890 https://api.wandb.ai/healthz; then
    export HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 NO_PROXY=localhost,127.0.0.1
else
    export WANDB_MODE=offline
fi
MIX="--t2i_chat_path=$D/t2i_chat_text.jsonl --t2i_chat_epochs=2 --caption_weight=0.1 --instruct_weight=0.25 --text_weight=0.45 --chat_weight=0.2"

stage() {  # name, base checkpoint, extra arguments
    local ckpt=$WEIGHT_DIR/$1.ckpt base=$2
    shift 2
    if [ -f "$ckpt.done" ]; then echo "== $(basename $ckpt) already done"; return; fi
    local resume=""
    [ -f "$ckpt.state" ] && resume="--resume=$ckpt.state"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m script.train_vlm --stage=sft \
        --ckpt_path=$base --tokenizer_path=$TOKENIZER --save_ckpt_path=$ckpt $resume $MIX "$@"
    touch "$ckpt.done"
}

stage chat_t2i_v7 $WEIGHT_DIR/vlm_rl.ckpt --llm_lr_frac=0.07 --max_steps=1600
stage chat_t2i_v9 $WEIGHT_DIR/chat_t2i_v7.ckpt --llm_lr_frac=0.015 --max_steps=300 --eval_every=50 --save_every=50 \
    --identity_paths=$D/identity_conversations_en_d24.jsonl:2,$D/identity_qa_d24.jsonl:5,$D/identity_direct_d24.jsonl:8
unset HTTPS_PROXY HTTP_PROXY NO_PROXY
CKPT=$WEIGHT_DIR/chat_t2i_v9.ckpt
# train_vlm keeps the checkpoint of the best validation loss in chat_t2i_v9.ckpt; the final model is the last step, in the state
[ -f "$CKPT.last" ] || python - "$CKPT" <<'PY'
import sys
import torch
state = torch.load(sys.argv[1] + ".state", map_location="cpu", weights_only=False)
torch.save({k: v for k, v in state.items() if k not in ("optimizer_state", "loader_state", "best_val", "wandb_id")}, sys.argv[1] + ".last")
PY
CKPT=$CKPT.last
[ -f "$REPORT/t2i_chat_behaviour.json" ] || HF_HUB_OFFLINE=1 PYTHONPATH=$HOME/train_llm_pylibs/image CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.t2i_chat_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --report_dir=$REPORT
[ -f "$REPORT/t2i_chat_vlm.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --stage=t2i_chat_vlm --report_dir=$REPORT
[ -f "$REPORT/t2i_chat.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$CKPT \
    --tokenizer_path=$TOKENIZER --stage=t2i_chat --report_dir=$REPORT
