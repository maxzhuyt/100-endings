#!/bin/bash
#SBATCH --job-name=eval-n-opus46
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/creative-writing-bench/results/logs/eval_narrative_opus46_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/creative-writing-bench/results/logs/eval_narrative_opus46_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
export PATH=/project/jevans/maxzhuyt/vllm_env/bin:$PATH

cd /project/jevans/maxzhuyt/creative-writing-bench

# ── Step 1: Extract stories from creative_bench_runs.json ─────────────────
echo "=== Extracting stories for narrative_opus46__anthropic_claude-opus-4-6 at $(date) ==="
python3 << 'PYEOF'
import json, os

CWB = "/project/jevans/maxzhuyt/creative-writing-bench"
data = json.load(open(f"{CWB}/creative_bench_runs.json"))

RUN_KEY = "narrative_opus46__anthropic_claude-opus-4-6"
SLUG = "narrative_opus46__anthropic_claude-opus-4-6"

if RUN_KEY not in data:
    print(f"ERROR: {RUN_KEY} not found in creative_bench_runs.json")
    import sys; sys.exit(1)

v = data[RUN_KEY]
tasks = v.get("creative_tasks", {})
out_dir = os.path.join(CWB, "stories", SLUG)
os.makedirs(out_dir, exist_ok=True)

total = 0
for iter_key in sorted(tasks.keys(), key=int):
    if int(iter_key) > 1:
        continue  # iter 1 only
    for pid in sorted(tasks[iter_key].keys(), key=int):
        entry = tasks[iter_key][pid]
        for mod, result in entry.get("results_by_modifier", {}).items():
            text = result.get("model_response", "")
            if not text or result.get("generation_failed"):
                continue
            fname = f"i{int(iter_key)}_p{int(pid):02d}.txt"
            with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                f.write(text.strip() + "\n")
            total += 1

print(f"Extracted {total} stories to {out_dir}/")
PYEOF

# ── Step 2: Generate endings ──────────────────────────────────────────────
echo "=== Generating endings for narrative_opus46__anthropic_claude-opus-4-6 at $(date) ==="
python /project/jevans/maxzhuyt/creative-writing-bench/ending_results_analysis/scripts/run_cwb_endings_rep.py \
    --model-slugs "narrative_opus46__anthropic_claude-opus-4-6" \
    --results-base /project/jevans/maxzhuyt/creative-writing-bench/results/endings

# ── Step 3: Judge endings ─────────────────────────────────────────────────
echo "=== Judging endings for narrative_opus46__anthropic_claude-opus-4-6 at $(date) ==="
python /project/jevans/maxzhuyt/creative-writing-bench/ending_results_analysis/scripts/compute_cwb_judge_rep.py \
    --endings-base /project/jevans/maxzhuyt/creative-writing-bench/results/endings \
    --output /project/jevans/maxzhuyt/creative-writing-bench/results/judge/ending_judge_opus46.json \
    --model-slugs "narrative_opus46__anthropic_claude-opus-4-6"

echo "=== Done at $(date) ==="
