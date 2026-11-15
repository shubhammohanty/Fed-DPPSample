#!/usr/bin/env bash
# ==============================================================================
# Environment Setup for Topology-Aware Client Sampling (Conda)
# ==============================================================================
set -e

ENV_NAME="dpp_fl"
PYTHON_VERSION="3.12"

echo "[1/4] Creating conda environment ($ENV_NAME)..."
if ! conda env list | grep -q "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=$PYTHON_VERSION
    echo "Conda environment created: $ENV_NAME"
else
    echo "Conda environment $ENV_NAME already exists."
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[2/4] Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "[3/4] Installing PyTorch and dependencies..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA 12.1 support..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected. Installing CPU version of PyTorch..."
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

echo "[4/4] Configuring Flower Simulation engine (228 SuperNodes)..."
flwr federation simulation-config @none/default --num-supernodes 228 --client-resources-num-gpus 0.5

echo ""
echo "Setup complete. Activate environment using:"
echo "    conda activate $ENV_NAME"
echo ""
