#!/bin/bash
#SBATCH --job-name=re-nyerx
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/results_endings/logs/nyer_expanded_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/results_endings/logs/nyer_expanded_%j.err

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

python run_endings.py \
    --data-dir /project/jevans/maxzhuyt/narrative_project/NYER_EXPANDED \
    --results-dir /project/jevans/maxzhuyt/narrative_project/results_endings/nyer_expanded \
    --no-id-filter
