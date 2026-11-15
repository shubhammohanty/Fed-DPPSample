#!/bin/bash
#SBATCH --job-name=poc_test
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

module purge
module load conda/miniconda3 2>/dev/null || true
source $(conda info --base 2>/dev/null)/etc/profile.d/conda.sh 2>/dev/null || true
conda activate dpp_fl 2>/dev/null || true

PROJECT_DIR="${PROJECT_DIR:-/driveB/FL/group9/flproject/Fed-DPPSample}"
if [ ! -d "$PROJECT_DIR" ]; then
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_DIR"
mkdir -p logs

flwr run . --run-config "sampling-mode='poc' poc-d=20 num-server-rounds=3 seed=42"
echo "Power-of-Choice test completed."
