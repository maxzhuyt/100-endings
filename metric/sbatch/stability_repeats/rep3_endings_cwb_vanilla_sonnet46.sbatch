#!/bin/bash
#SBATCH --job-name=r3-e-van
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/rep3_endings_cwb_vanilla_sonnet46_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/rep3_endings_cwb_vanilla_sonnet46_%j.err

unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/
module purge
module load cuda

echo "=== Rep 3: cwb_vanilla_sonnet46 endings (T=1.2, iter1 only) at $(date) ==="
/project/jevans/maxzhuyt/vllm_env/bin/python /project/jevans/maxzhuyt/narrative_project/narrative_ablation/generate_endings.py \
    --data-dir /project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/stories_txt/cwb_vanilla_sonnet46_iter1 \
    --results-dir /project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/endings_rep3/cwb_vanilla_sonnet46 \
    --no-id-filter
echo "=== Done at $(date) ==="
