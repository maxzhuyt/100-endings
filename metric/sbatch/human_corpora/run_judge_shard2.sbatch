#!/bin/bash
#SBATCH --job-name=rej-sh2
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/results_endings/logs/judge_sh2_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/results_endings/logs/judge_sh2_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/vllm_env

cd /project/jevans/maxzhuyt/narrative_project/results_endings

echo "=== Shard 2: nyer_83 IDs 100-199 (Qwen3-8B) at $(date) ==="
python compute_judge.py --batch-size 512 --group nyer_83 --sid-min 100 --output ending_judge_sh2.json
echo "=== Done at $(date) ==="
