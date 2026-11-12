#!/bin/bash
#SBATCH --job-name=cwb-judge
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --qos=jevans-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=H200
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/creative-writing-bench/results/judge/slurm_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/creative-writing-bench/results/judge/slurm_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
export PATH=/project/jevans/maxzhuyt/vllm_env/bin:$PATH

echo "=== CWB Judge at $(date) ==="
python /project/jevans/maxzhuyt/creative-writing-bench/ending_results_analysis/scripts/compute_cwb_judge.py --batch-size 512
echo "=== Finished at $(date) ==="
