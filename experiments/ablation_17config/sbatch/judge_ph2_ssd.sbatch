#!/bin/bash
#SBATCH --job-name=jdg-ph2
#SBATCH --partition=ssd-gpu
#SBATCH --account=ssd
#SBATCH --qos=ssd
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=35:00:00
#SBATCH --output=results/logs/judge_ph2_%j.out
#SBATCH --error=results/logs/judge_ph2_%j.err

# Judge Phase 2: fulltext_beats3, fulltext_beats5, fulltext_beats10

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID}
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/ /project/jevans/maxzhuyt/cache/torchinductor/ /project/jevans/maxzhuyt/cache/triton/

module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/vllm_env

cd /project/jevans/maxzhuyt/narrative_ablation
mkdir -p results/logs results/judge

echo "=== Starting judge Phase 2 at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python compute_judge.py --config fulltext_beats3,fulltext_beats5,fulltext_beats10 --batch-size 128

echo "=== Finished at $(date) ==="
