#!/bin/bash
#SBATCH --job-name=abl-judge
#SBATCH --partition=jevans-gpu
#SBATCH --account=pi-jevans
#SBATCH --constraint=H200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/judge_%j.out
#SBATCH --error=/project/jevans/maxzhuyt/narrative_project/narrative_ablation/results/logs/judge_%j.err

export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/project/jevans/maxzhuyt/cache/torchinductor
export TRITON_CACHE_DIR=/project/jevans/maxzhuyt/cache/triton
mkdir -p $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
rm -rf ~/.cache/vllm/torch_compile_cache/

module load cuda
export PATH=/project/jevans/maxzhuyt/vllm_env/bin:$PATH

BASE=/project/jevans/maxzhuyt/narrative_project/narrative_ablation

echo "=== Ablation Judge (Qwen3-8B, lenient) at $(date) ==="

# The compute_judge.py expects a CATEGORIES dict. We need a wrapper
# that creates per-config judge calls.

python3 << 'PYEOF'
import json, os, sys, time, gc, logging

# Patch for vLLM + transformers 5.x compatibility
from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE = "/project/jevans/maxzhuyt/narrative_project/narrative_ablation"
MODEL_PATH = "/project/jevans/maxzhuyt/models/Qwen3-8B"
BATCH_SIZE = 512
PCT_MIN, PCT_MAX = 10, 99
MAX_CONTINUATION_WORDS = 2000

import re

def split_sentences(text):
    boundaries = [0]
    for m in re.finditer(r'[.!?]["\u201d\u2019)]*\s+(?=[A-Z\u201c"(\[])', text):
        boundaries.append(m.end())
    parts = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts

def truncate_words(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [...]"

SYSTEM_PROMPT = """You are a literary judge. You determine whether a predicted story ending captures the general direction of the true story continuation. Answer Y if the prediction gets the broad plot trajectory right — the same characters are involved, similar types of events unfold, or the emotional arc matches. Minor differences in specific details, wording, or sequence are acceptable. Answer N only if the prediction describes a fundamentally different outcome from what actually happens."""

def build_judge_prompt(preceding, continuation, ending):
    return f"""Here is the actual continuation of a story:

---BEGIN CONTINUATION---
{preceding} {continuation}
---END CONTINUATION---

Here is a predicted ending:

---BEGIN PREDICTED---
{preceding} {ending}
---END PREDICTED---

Does the predicted ending capture the general direction of the true continuation — similar characters involved, similar types of events, or a matching emotional arc? Answer Y if the broad trajectory is right, even if specific details differ. Answer N only if the prediction is fundamentally wrong about what happens. Answer ONLY Y or N.

Judgment:"""

from vllm import LLM, SamplingParams

log.info(f"Loading model {MODEL_PATH}...")
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.92,
    max_model_len=16384,
    trust_remote_code=True,
    dtype="bfloat16",
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

configs = sorted(d for d in os.listdir(f"{BASE}/results/endings")
                 if os.path.isdir(f"{BASE}/results/endings/{d}"))

t_global = time.time()

for config in configs:
    endings_dir = f"{BASE}/results/endings/{config}"
    stories_dir = f"{BASE}/results/stories_txt/{config}"
    output_file = f"{BASE}/results/judge/{config}_judge.json"

    if os.path.exists(output_file):
        existing = json.load(open(output_file))
        log.info(f"  {config}: {len(existing)} already judged, resuming")
    else:
        existing = {}

    endings_files = sorted(f for f in os.listdir(endings_dir) if f.endswith("_endings.json"))
    if not endings_files:
        continue

    log.info(f"\n=== {config}: {len(endings_files)} stories ===")
    done = 0

    for ef in endings_files:
        sid = ef.replace("_endings.json", "")
        if sid in existing:
            done += 1
            continue

        ej = json.load(open(os.path.join(endings_dir, ef)))
        story_path = os.path.join(stories_dir, f"{sid}.txt")
        if not os.path.exists(story_path):
            continue
        text = open(story_path).read().strip()
        sentences = split_sentences(text)
        if len(sentences) < 3:
            continue

        existing[sid] = {"group": config, "positions": []}

        for pos_data in ej["positions"]:
            pos_idx = pos_data["position"]
            pct = pos_data.get("pct_story_revealed", 0)
            if pct < PCT_MIN or pct > PCT_MAX:
                continue

            continuation = " ".join(sentences[pos_idx:])
            if not continuation.strip():
                continue
            continuation = truncate_words(continuation, MAX_CONTINUATION_WORDS)
            preceding = sentences[pos_idx - 1] if pos_idx > 0 else ""
            endings = pos_data["endings"]

            conversations = []
            for ending in endings:
                conversations.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_judge_prompt(preceding, continuation, ending)},
                ])

            judgments = []
            for b_start in range(0, len(endings), BATCH_SIZE):
                b_end = min(b_start + BATCH_SIZE, len(endings))
                outputs = llm.chat(
                    conversations[b_start:b_end],
                    sampling_params,
                    chat_template_kwargs={"enable_thinking": False},
                )
                for output in outputs:
                    resp = output.outputs[0].text.strip().upper()
                    judgments.append(1 if resp.startswith("Y") else 0)

            yes_rate = sum(judgments) / len(judgments) if judgments else 0
            existing[sid]["positions"].append({
                "position": pos_idx,
                "pct": pct,
                "yes_rate": yes_rate,
                "judgments": judgments,
            })

        done += 1
        if done % 5 == 0:
            with open(output_file, "w") as f:
                json.dump(existing, f)

        gc.collect()

    with open(output_file, "w") as f:
        json.dump(existing, f)
    log.info(f"  {config}: {done} stories judged → {output_file}")

elapsed = time.time() - t_global
log.info(f"\nAll configs done in {elapsed/3600:.1f}h")
PYEOF

echo "=== Finished at $(date) ==="
