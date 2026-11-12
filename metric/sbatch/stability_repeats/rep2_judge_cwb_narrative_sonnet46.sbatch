#!/bin/bash
#SBATCH --job-name=r2-j-nar
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/rep2_judge_cwb_narrative_sonnet46_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/rep2_judge_cwb_narrative_sonnet46_%j.err

unset PYTHONPATH
unset PYTHONHOME
unset CONDA_PREFIX
unset VIRTUAL_ENV
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module purge
source /project/jevans/maxzhuyt/vllm_env/bin/activate
module load cuda

export ENDINGS_DIR=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/endings_rep2/cwb_narrative_sonnet46
export OUTPUT_FILE=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/judge_rep2/cwb_narrative_sonnet46_judge.json

echo "=== Rep 2: cwb_narrative_sonnet46 judge at $(date) ==="
/project/jevans/maxzhuyt/vllm_env/bin/python /project/jevans/maxzhuyt/narrative_project/narrative_ablation/sbatch/run_judge_rep.py cwb_narrative_sonnet46
echo "=== Done at $(date) ==="
