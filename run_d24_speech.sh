#!/usr/bin/env bash
# d24 speech on two GPUs, thinker = the VLM + RL model (frozen): t2a (talker) -> a2a (talker + audio projector)
# python -m script.download_audio && python -m script.prepare_audio_data first
# GPUS=1,2 setsid nohup bash run_d24_speech.sh >> log/d24_speech.log 2>&1 < /dev/null &
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
WEIGHT_DIR=weight/d24
TOKENIZER=$WEIGHT_DIR/tokenizer.pkl
THINKER=$WEIGHT_DIR/vlm_rl.ckpt
T2A_CKPT=$WEIGHT_DIR/speech_t2a.ckpt
A2A_CKPT=$WEIGHT_DIR/speech_a2a.ckpt
mkdir -p log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCHRUN="torchrun --standalone --nproc_per_node=2"

stage() {
    local name=$1 ckpt=$2; shift 2
    if [ -f "$ckpt.done" ]; then echo "== $name already done"; return; fi
    local resume=""
    [ -f "$ckpt.state" ] && resume="--resume=$ckpt.state" && echo "== $name resumes from $ckpt.state"
    CUDA_VISIBLE_DEVICES=$GPUS $TORCHRUN -m script.train_speech --stage=$name --thinker_ckpt=$THINKER \
        --tokenizer_path=$TOKENIZER --save_ckpt_path=$ckpt $resume "$@"
    touch "$ckpt.done"
}

stage t2a $T2A_CKPT
stage a2a $A2A_CKPT --speech_ckpt=$T2A_CKPT
