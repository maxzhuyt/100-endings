#!/bin/bash
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
export PATH=/project/jevans/maxzhuyt/vllm_env/bin:$PATH

#SBATCH --job-name=abl-j2
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/judge_sh2_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/judge_sh2_%j.err

cd /project/jevans/maxzhuyt/narrative_project/narrative_ablation
for CONFIG in fulltext_beats3 fulltext_beats5 fulltext_beats10 instruction_vanilla idea_instruction_vanilla; do
    echo "=== Judge: $CONFIG ==="
    python sbatch/run_judge_inline.py $CONFIG
done
