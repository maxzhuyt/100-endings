#!/bin/bash
#SBATCH --job-name=narr-stories
#SBATCH --partition=jevans
#SBATCH --account=pi-jevans
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=results/logs/stories_%j.out
#SBATCH --error=results/logs/stories_%j.err

# Stage 1: Generate stories via API (CPU only, no GPU needed)

export PYTHONUNBUFFERED=1

module load python
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

cd /project/jevans/maxzhuyt/narrative_ablation
mkdir -p results/logs

echo "=== Starting story generation at $(date) ==="
echo "Node: $(hostname)"

python generate_stories.py \
    --model "anthropic/claude-sonnet-4-6" \
    --iterations 3 \
    --threads 4

echo "=== Finished at $(date) ==="
