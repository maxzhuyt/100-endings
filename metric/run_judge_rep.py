#!/usr/bin/env python3
"""Judge wrapper that respects ENDINGS_DIR and OUTPUT_FILE env vars."""
import os, sys

# Override paths via env vars before importing the main logic
config = sys.argv[1]
BASE = "/project/jevans/maxzhuyt/narrative_project/narrative_ablation"
endings_dir = os.environ.get("ENDINGS_DIR", f"{BASE}/results/endings/{config}")
output_file = os.environ.get("OUTPUT_FILE", f"{BASE}/results/judge/{config}_judge.json")
stories_dir = f"{BASE}/results/stories_txt/{config}"

# Now run the judge logic inline (copy from run_judge_inline.py with overridden paths)
import json, re, time, gc
import numpy as np

from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

MODEL_PATH = os.environ.get("ENDINGS_JUDGE_MODEL", "/project/jevans/maxzhuyt/models/Qwen3-8B")
BATCH_SIZE = 512
PCT_MIN, PCT_MAX = 10, 99
MAX_CONTINUATION_WORDS = 2000

def split_sentences(text):
    boundaries = [0]
    for m in re.finditer(r'[.!?]["\u201d\u2019)]*\s+(?=[A-Z\u201c"(\[])', text):
        boundaries.append(m.end())
    parts = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end].strip()
        if chunk: parts.append(chunk)
    return parts

def truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text

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

if not os.path.isdir(endings_dir):
    print(f"No endings for {config} at {endings_dir}, skipping")
    sys.exit(0)

if os.path.exists(output_file):
    existing = json.load(open(output_file))
    print(f"Resuming {config}: {len(existing)} already done")
else:
    existing = {}

endings_files = sorted(f for f in os.listdir(endings_dir) if f.endswith("_endings.json"))
todo = [f for f in endings_files if f.replace("_endings.json", "") not in existing]
if not todo:
    print(f"{config}: all {len(endings_files)} stories already judged")
    sys.exit(0)

print(f"{config}: {len(todo)} stories to judge ({len(existing)} done)")
print(f"  endings_dir: {endings_dir}")
print(f"  output_file: {output_file}")

from vllm import LLM, SamplingParams
llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, gpu_memory_utilization=0.92,
          max_model_len=16384, trust_remote_code=True, dtype="bfloat16")
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

t0 = time.time()
done = 0

for ef in todo:
    sid = ef.replace("_endings.json", "")
    ej = json.load(open(os.path.join(endings_dir, ef)))
    story_path = os.path.join(stories_dir, f"{sid}.txt")
    if not os.path.exists(story_path): continue
    text = open(story_path).read().strip()
    sentences = split_sentences(text)
    if len(sentences) < 3: continue

    existing[sid] = {"group": config, "positions": []}
    for pos_data in ej["positions"]:
        pos_idx = pos_data["position"]
        pct = pos_data.get("pct_story_revealed", 0)
        if pct < PCT_MIN or pct > PCT_MAX: continue
        continuation = " ".join(sentences[pos_idx:])
        if not continuation.strip(): continue
        continuation = truncate_words(continuation, MAX_CONTINUATION_WORDS)
        preceding = sentences[pos_idx - 1] if pos_idx > 0 else ""
        endings = pos_data["endings"]

        conversations = [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_prompt(preceding, continuation, e)},
        ] for e in endings]

        judgments = []
        for b_start in range(0, len(endings), BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, len(endings))
            outputs = llm.chat(conversations[b_start:b_end], sampling_params,
                              chat_template_kwargs={"enable_thinking": False})
            for output in outputs:
                resp = output.outputs[0].text.strip().upper()
                judgments.append(1 if resp.startswith("Y") else 0)

        yes_rate = sum(judgments) / len(judgments) if judgments else 0
        existing[sid]["positions"].append({
            "position": pos_idx, "pct": pct,
            "yes_rate": yes_rate, "judgments": judgments,
        })

    done += 1
    if done % 5 == 0:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(existing, f)
    gc.collect()

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    json.dump(existing, f)
print(f"{config}: {done} stories judged in {(time.time()-t0)/60:.1f}min → {output_file}")
