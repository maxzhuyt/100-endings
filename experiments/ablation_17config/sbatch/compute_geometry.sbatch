#!/bin/bash
#SBATCH --job-name=narr-geometry
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/geometry_%j.out
#SBATCH --error=results/logs/geometry_%j.err

# Stage 3: Compute embedding geometry (Qwen3-Embedding-0.6B)

export PYTHONUNBUFFERED=1
module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/vllm_env

cd /project/jevans/maxzhuyt/narrative_ablation
mkdir -p results/logs results/geometry

echo "=== Starting geometry computation at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python compute_geometry.py

echo "=== Finished at $(date) ==="
