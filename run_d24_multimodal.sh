#!/usr/bin/env bash
# d24 multimodal, from the text SFT model of run_2gpu_d24.sh to the final model chat_t2i_v9, on two 32GB GPUs:
#   vlm_align    image understanding: frozen SigLIP2, only the projector trains, ALLaVA captions
#   vlm_sft2     projector + GPT on ALLaVA instructions and captions, with replay of the text SFT data
#   vlm_rl       RL on math questions SFT has not seen, like run_2gpu_d24.sh (the projector is carried through)
#   chat_t2i_v7  drawing conversations: when to draw, writing and changing the description, seeing the pictures it drew
#   chat_t2i_v9  a short round with identity data at a lower learning rate; the last step is the final model
# run_2gpu_d24.sh first (weight/d24/sft_model.ckpt, tokenizer, RL questions in data/raw/rl).
# GPUS=0,1 setsid nohup bash run_d24_multimodal.sh >> log/d24_multimodal.log 2>&1 < /dev/null &
# Rerunning the same command after a kill continues: finished stages are skipped (<ckpt>.done), an unfinished one
# resumes from its <ckpt>.state.
set -e

if ! command -v conda >/dev/null 2>&1; then
    for p in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
    done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gpt
cd "$(dirname "$0")"

GPUS=${GPUS:-0,1}
EVAL_GPU=${GPUS%%,*}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
W=weight/d24
TOKENIZER=$W/tokenizer.pkl
D=data/raw/custom
REPORT=$W/report_multimodal
mkdir -p "$REPORT" log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- data: ALLaVA + SigLIP2 + image benchmarks, our drawing conversations, Z-Image-Turbo for the drawing test ----
python -m script.download_vlm
python -m script.prepare_vlm_data
python -m script.download_release --what=data
python -m script.download_image_backend

done_or() {  # run the rest of the line unless $1.done exists, then mark it done
    local ckpt=$1; shift
    if [ -f "$ckpt.done" ]; then echo "== $(basename $ckpt) already done"; return; fi
    "$@"
    touch "$ckpt.done"
}
resume() { [ -f "$1.state" ] && echo "--resume=$1.state" || true; }
vlm() {  # stage, checkpoint to save, base checkpoint, extra arguments
    local stage=$1 ckpt=$2 base=$3; shift 3
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=$NPROC -m script.train_vlm --stage=$stage \
        --ckpt_path=$base --tokenizer_path=$TOKENIZER --save_ckpt_path=$ckpt $(resume $ckpt) "$@"
}

# ---- image understanding ----
done_or $W/vlm_align.ckpt vlm align $W/vlm_align.ckpt $W/sft_model.ckpt
done_or $W/vlm_sft2.ckpt vlm sft $W/vlm_sft2.ckpt $W/vlm_align.ckpt --text_weight=0.35 --llm_lr_frac=0.07
done_or $W/vlm_rl.ckpt env CUDA_VISIBLE_DEVICES=$GPUS python -m script.chat_rl --ckpt_path=$W/vlm_sft2.ckpt \
    --tokenizer_path=$TOKENIZER --save_ckpt_path=$W/vlm_rl.ckpt --report_dir=$REPORT --device_batch_size=4 \
    --device=cuda:1 --ref_device=cuda:0 --gpu_memory_utilization=0.6 --kl_beta=0.04 --oversample=2.5 --eval_every=30 \
    --train_data=data/raw/rl/omi_gsm.jsonl --max_steps=233 $(resume $W/vlm_rl.ckpt)

# ---- conversational drawing ----
MIX="--t2i_chat_path=$D/t2i_chat_text.jsonl --t2i_chat_epochs=2 --caption_weight=0.1 --instruct_weight=0.25 --text_weight=0.45 --chat_weight=0.2"
done_or $W/chat_t2i_v7.ckpt vlm sft $W/chat_t2i_v7.ckpt $W/vlm_rl.ckpt $MIX --llm_lr_frac=0.07 --max_steps=1600
done_or $W/chat_t2i_v9.ckpt vlm sft $W/chat_t2i_v9.ckpt $W/chat_t2i_v7.ckpt $MIX --llm_lr_frac=0.015 --max_steps=300 \
    --eval_every=50 --save_every=50 \
    --identity_paths=$D/identity_conversations_en_d24.jsonl:2,$D/identity_qa_d24.jsonl:5,$D/identity_direct_d24.jsonl:8
# train_vlm keeps the checkpoint of the best validation loss; the final model is the last step, in the training state
FINAL=$W/chat_t2i_v9.ckpt.last
[ -f "$FINAL" ] || python - $W/chat_t2i_v9.ckpt <<'PY'
import sys
import torch
state = torch.load(sys.argv[1] + ".state", map_location="cpu", weights_only=False)
torch.save({k: v for k, v in state.items() if k not in ("optimizer_state", "loader_state", "best_val", "wandb_id")}, sys.argv[1] + ".last")
PY

# ---- benchmarks of the final model: chat, image understanding, the drawing conversations ----
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$FINAL --tokenizer_path=$TOKENIZER --stage=final --report_dir=$REPORT
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$FINAL --tokenizer_path=$TOKENIZER --stage=final_vlm --report_dir=$REPORT
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.t2i_chat_eval --ckpt_path=$FINAL --tokenizer_path=$TOKENIZER --report_dir=$REPORT
