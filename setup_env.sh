#!/usr/bin/env bash
# Create / update this project's conda environment
# Usage:
#   bash setup_env.sh                 # default env name gpt, CUDA 12.8
#   ENV_NAME=gpt2 bash setup_env.sh   # custom env name
#   CUDA=cu126 bash setup_env.sh      # pick the CUDA build of PyTorch (cu126 / cu128 / cu130 / cpu)
#   RUSTBPE_SRC=/path/to/rustbpe bash setup_env.sh   # build rustbpe from a local checkout instead of the fork
# Note: RTX 50-series (Blackwell) cards need cu128 or newer; older drivers can use cu126
set -e

ENV_NAME=${ENV_NAME:-gpt}
CUDA=${CUDA:-cu128}
TORCH_VERSION=${TORCH_VERSION:-2.11.0}
# Modified rustbpe (i64 counts + min_freq), pinned to a commit on the fork
# PR: https://github.com/karpathy/rustbpe/pull/8
RUSTBPE_SRC=${RUSTBPE_SRC:-git+https://github.com/1023097618/rustbpe@27ce874ed07e5e53e9a805e5b0b5294b06f078fa}
PROJECT_DIR=$(cd "$(dirname "$0")" && pwd)

# Locate conda
if ! command -v conda >/dev/null 2>&1; then
    for p in "$HOME/miniconda3" "$HOME/anaconda3" /opt/conda; do
        [ -f "$p/etc/profile.d/conda.sh" ] && source "$p/etc/profile.d/conda.sh" && break
    done
fi
command -v conda >/dev/null 2>&1 || { echo "conda not found, please install Miniconda/Anaconda first"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create or update the environment
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo ">>> Env $ENV_NAME already exists, updating"
    conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml"
else
    echo ">>> Creating env $ENV_NAME"
    conda env create -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml"
fi
conda activate "$ENV_NAME"

# Install PyTorch (with CUDA)
echo ">>> Installing torch==$TORCH_VERSION ($CUDA)"
pip install "torch==$TORCH_VERSION" --index-url "https://download.pytorch.org/whl/$CUDA"

# Install the modified rustbpe (built from source with maturin, needs a Rust toolchain)
if ! command -v cargo >/dev/null 2>&1; then
    [ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo ">>> cargo not found, installing Rust toolchain into the env from conda-forge"
    conda install -y -n "$ENV_NAME" -c conda-forge rust
fi
echo ">>> Installing rustbpe from $RUSTBPE_SRC"
pip install "$RUSTBPE_SRC"

# Register a Jupyter kernel so notebooks (e.g. optimizer/get_coefficient.ipynb) can use it
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# Self-check
echo ">>> Checking environment"
python - <<'EOF'
import torch, tiktoken, rustbpe, pyarrow, pandas, wandb, docstring_parser, huggingface_hub
print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU{i}: {torch.cuda.get_device_name(i)}, bf16 support: {torch.cuda.is_bf16_supported()}")
print(f"NCCL available: {torch.distributed.is_nccl_available()}")
EOF

echo ">>> Done. Activate with: conda activate $ENV_NAME"
