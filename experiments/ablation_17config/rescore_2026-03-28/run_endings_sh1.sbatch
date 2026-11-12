#!/bin/bash
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
export PATH=/project/jevans/maxzhuyt/vllm_env/bin:$PATH

BASE=/project/jevans/maxzhuyt/narrative_project/narrative_ablation

#SBATCH --job-name=abl-e1
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/endings_sh1_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/endings_sh1_%j.err

for CONFIG in full_title no_warmup short_beats vanilla; do
    STORIES_DIR=$BASE/results/stories_txt/$CONFIG
    ENDINGS_DIR=$BASE/results/endings/$CONFIG
    mkdir -p $ENDINGS_DIR
    echo "=== $CONFIG ==="
    python $BASE/generate_endings.py --data-dir $STORIES_DIR --results-dir $ENDINGS_DIR --no-id-filter
done
