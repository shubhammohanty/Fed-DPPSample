#!/bin/bash
#SBATCH --job-name=dpp_stage2
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
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

python run_stage2_dpp_suite.py
echo "Stage 2 execution completed."
