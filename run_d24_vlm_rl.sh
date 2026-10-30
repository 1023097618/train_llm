#!/usr/bin/env bash
# d24 VLM, second round: VLM sft again from the align checkpoint with more text replay and a lower GPT lr
# (the first round kept ARC / MMLU but HumanEval fell 0.146 -> 0.079), then RL on top of the VLM (text only,
# the image projector is carried through), benchmarks after each stage.
# run_d24_vlm.sh (align) and run_d24_rl_omi.sh's data (prepare_rl_data) first.
# GPUS=1,2 setsid nohup bash run_d24_vlm_rl.sh >> log/d24_vlm_rl.log 2>&1 < /dev/null &
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
ALIGN_CKPT=$WEIGHT_DIR/vlm_align.ckpt
VLM_CKPT=$WEIGHT_DIR/vlm_sft2.ckpt
RL_CKPT=$WEIGHT_DIR/vlm_rl.ckpt
REPORT=$WEIGHT_DIR/report_vlm2
mkdir -p "$REPORT" log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- VLM sft, round 2: text replay weight 0.2 -> 0.35, GPT lr fraction 0.1 -> 0.07 ----
if [ ! -f "$VLM_CKPT.done" ]; then
    resume=""
    [ -f "$VLM_CKPT.state" ] && resume="--resume=$VLM_CKPT.state"
    CUDA_VISIBLE_DEVICES=$GPUS torchrun --standalone --nproc_per_node=2 -m script.train_vlm --stage=sft \
        --ckpt_path=$ALIGN_CKPT --tokenizer_path=$TOKENIZER --save_ckpt_path=$VLM_CKPT $resume \
        --text_weight=0.35 --llm_lr_frac=0.07
    touch "$VLM_CKPT.done"
fi
[ -f "$REPORT/vlm_sft2.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$VLM_CKPT \
    --tokenizer_path=$TOKENIZER --stage=vlm_sft2 --report_dir=$REPORT
[ -f "$REPORT/sft2.json" ] || CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$VLM_CKPT \
    --tokenizer_path=$TOKENIZER --stage=sft2 --report_dir=$REPORT

# ---- RL on the VLM: same setup as run_d24_rl_omi.sh (vLLM + reference model on the first GPU, training on the second) ----
if [ ! -f "$RL_CKPT.done" ]; then
    resume=""
    [ -f "$RL_CKPT.state" ] && resume="--resume=$RL_CKPT.state"
    CUDA_VISIBLE_DEVICES=$GPUS python -m script.chat_rl --ckpt_path=$VLM_CKPT --tokenizer_path=$TOKENIZER \
        --save_ckpt_path=$RL_CKPT --report_dir=$REPORT --device_batch_size=4 --device=cuda:1 --ref_device=cuda:0 \
        --gpu_memory_utilization=0.6 --kl_beta=0.04 --oversample=2.5 --eval_every=30 \
        --train_data=data/raw/rl/omi_gsm.jsonl --max_steps=233 $resume
    touch "$RL_CKPT.done"
fi
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.vlm_eval --ckpt_path=$RL_CKPT --tokenizer_path=$TOKENIZER \
    --stage=vlm_rl --report_dir=$REPORT
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m script.chat_eval --ckpt_path=$RL_CKPT --tokenizer_path=$TOKENIZER \
    --stage=rl --report_dir=$REPORT
