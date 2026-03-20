#!/bin/bash
#SBATCH --job-name=jdg-retry
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=results/logs/judge_retry_%j.out
#SBATCH --error=results/logs/judge_retry_%j.err

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
mkdir -p results/logs results/judge

echo "=== Judge for retried stories at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python compute_judge.py --config fulltext_beats3,fulltext_beats5,fulltext_beats10,idea_full --batch-size 128

echo "=== Finished at $(date) ==="
