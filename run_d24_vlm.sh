#!/usr/bin/env bash
# d24 image understanding on two GPUs: align (projector only) -> sft (projector + GPT), benchmarks before and after
# python -m script.download_vlm && python -m script.prepare_vlm_data first (HF_TOKEN speeds up the mirror)
# GPUS=1,2 setsid nohup bash run_d24_vlm.sh >> log/d24_vlm.log 2>&1 < /dev/null &
# Rerunning the same command after a kill continues where it stopped: finished stages are skipped
# (<ckpt>.done), an unfinished stage resumes from its <ckpt>.state.
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
SFT_CKPT=$WEIGHT_DIR/sft_model.ckpt           # text chat model (run_2gpu_d24.sh)
ALIGN_CKPT=$WEIGHT_DIR/vlm_align.ckpt
VLM_CKPT=$WEIGHT_DIR/vlm_sft.ckpt
REPORT=$WEIGHT_DIR/report_vlm
mkdir -p "$REPORT" log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCHRUN="torchrun --standalone --nproc_per_node=2"

# run a train_vlm stage unless it finished, resuming from its training state when there is one
stage() {
    local name=$1 ckpt=$2; shift 2
    if [ -f "$ckpt.done" ]; then echo "== $name already done"; return; fi
    local resume=""
    [ -f "$ckpt.state" ] && resume="--resume=$ckpt.state" && echo "== $name resumes from $ckpt.state"
    CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN -m script.train_vlm --stage=$name --tokenizer_path=$TOKENIZER \
        --save_ckpt_path=$ckpt $resume "$@"
    touch "$ckpt.done"
}

# ---- before: the text model answers the benchmarks without seeing the image ----
[ -f "$REPORT/vlm_blind_text_model.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$SFT_CKPT \
    --tokenizer_path=$TOKENIZER --no_image=1 --stage=vlm_blind_text_model --report_dir=$REPORT

# ---- align: captions, projector only ----
stage align $ALIGN_CKPT --ckpt_path=$SFT_CKPT
[ -f "$REPORT/vlm_align.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$ALIGN_CKPT \
    --tokenizer_path=$TOKENIZER --stage=vlm_align --report_dir=$REPORT

# ---- sft: instructions + captions + text SFT replay, projector + GPT ----
stage sft $VLM_CKPT --ckpt_path=$ALIGN_CKPT
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$VLM_CKPT --tokenizer_path=$TOKENIZER \
    --stage=vlm_sft --report_dir=$REPORT
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$VLM_CKPT --tokenizer_path=$TOKENIZER \
    --no_image=1 --stage=vlm_sft_blind --report_dir=$REPORT
# text ability after the image training (compare with weight/d24/report/sft.json)
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$VLM_CKPT --tokenizer_path=$TOKENIZER \
    --stage=vlm --report_dir=$REPORT
