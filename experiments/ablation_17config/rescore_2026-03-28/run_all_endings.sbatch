#!/bin/bash
#SBATCH --job-name=abl-ends
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/endings_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/endings_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
export PATH=/project/jevans/maxzhuyt/vllm_env/bin:$PATH

BASE=/project/jevans/maxzhuyt/narrative_project/narrative_ablation
mkdir -p $BASE/results/logs

echo "=== Ablation Endings (describe pipeline) at $(date) ==="

for CONFIG in $(ls $BASE/results/stories_txt/); do
    STORIES_DIR=$BASE/results/stories_txt/$CONFIG
    ENDINGS_DIR=$BASE/results/endings/$CONFIG
    N_STORIES=$(ls $STORIES_DIR/*.txt 2>/dev/null | wc -l)
    N_DONE=$(ls $ENDINGS_DIR/*_endings.json 2>/dev/null | wc -l)

    if [ "$N_STORIES" -eq 0 ]; then
        echo "  SKIP $CONFIG: no stories"
        continue
    fi

    if [ "$N_DONE" -ge "$N_STORIES" ]; then
        echo "  SKIP $CONFIG: already done ($N_DONE/$N_STORIES)"
        continue
    fi

    mkdir -p $ENDINGS_DIR
    echo ""
    echo "=== $CONFIG ($N_STORIES stories, $N_DONE done) ==="

    python $BASE/generate_endings.py \
        --data-dir $STORIES_DIR \
        --results-dir $ENDINGS_DIR \
        --no-id-filter
done

echo ""
echo "=== All configs done at $(date) ==="
