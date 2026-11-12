#!/bin/bash
#SBATCH --job-name=re-tmas
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/results_endings/logs/tmas_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/results_endings/logs/tmas_%j.err

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

# Extract TMAS stories to txt files
python -c "
import json, os
os.makedirs('/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/tell_me_a_story/txt', exist_ok=True)
with open('/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/tell_me_a_story/all_stories.jsonl') as f:
    for line in f:
        r = json.loads(line)
        sid = f'tmas_{r[\"story_id\"]:04d}'
        with open(f'/project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/tell_me_a_story/txt/{sid}.txt', 'w') as out:
            out.write(r['text'])
print('Extracted TMAS stories')
"

python run_endings.py \
    --data-dir /project/jevans/maxzhuyt/narrative_project/BASELINE_CORPUS/tell_me_a_story/txt \
    --results-dir /project/jevans/maxzhuyt/narrative_project/results_endings/tell_me_a_story \
    --no-id-filter
