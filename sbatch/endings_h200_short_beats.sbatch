#!/bin/bash
#SBATCH --job-name=end-short
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=results/logs/endings_short_%j.out
#SBATCH --error=results/logs/endings_short_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/vllm_env

cd /project/jevans/maxzhuyt/narrative_ablation
mkdir -p results/logs

echo "=== Starting short_beats endings at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python generate_endings.py --config short_beats --batch-size 64

echo "=== Finished at $(date) ==="
