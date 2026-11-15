#!/bin/bash
#SBATCH --job-name=grid_stage1
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
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

python grid_search_stage1_random.py
python export_grid_summary.py --results-dir results/grid_search_stage3
echo "Stage 1 grid search completed."
