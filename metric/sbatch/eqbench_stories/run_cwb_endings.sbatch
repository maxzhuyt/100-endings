#!/bin/bash
#SBATCH --job-name=cwb-endings
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/creative-writing-bench/results/endings/slurm_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/creative-writing-bench/results/endings/slurm_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
export PATH=/project/jevans/maxzhuyt/vllm_env/bin:$PATH

TOP10="claude-sonnet-4-6,claude-opus-4-6,openai__gpt-5.2,claude-opus-4-5-20251101,anthropic__claude-sonnet-4.5,o3,moonshotai__kimi-k2,openrouter__pony-alpha,kimi-k2-thinking,openrouter__horizon-alpha,narrative_sonnet46__anthropic_claude-sonnet-4-6"

echo "=== CWB Endings (Top 10) at $(date) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python /project/jevans/maxzhuyt/creative-writing-bench/ending_results_analysis/scripts/run_cwb_endings.py \
    --model-slugs "$TOP10" \
    --batch-size 64

echo "=== Finished at $(date) ==="
