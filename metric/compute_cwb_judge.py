#!/usr/bin/env python3
"""
LLM-as-judge for creative-writing-bench endings (individual judging, lenient).

Each of the 100 predicted endings is judged independently against the
actual story continuation. Uses Qwen3-8B as judge with lenient matching.

Output: results/judge/ending_judge.json
"""

import os, json, re, gc, sys, time, argparse
import logging

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

BASE = "/project/jevans/maxzhuyt/creative-writing-bench"
STORIES_DIR = f"{BASE}/stories"
ENDINGS_DIR = f"{BASE}/results/endings"
OUTPUT_FILE = f"{BASE}/results/judge/ending_judge.json"
MODEL_PATH = os.environ.get("ENDINGS_JUDGE_MODEL", "/project/jevans/maxzhuyt/models/Qwen3-8B")
PCT_MIN, PCT_MAX = 10, 99
MAX_CONTINUATION_WORDS = 2000

SYSTEM_PROMPT = """You are a literary judge. You determine whether a predicted story ending captures the general direction of the true story continuation. Answer Y if the prediction gets the broad plot trajectory right — the same characters are involved, similar types of events unfold, or the emotional arc matches. Minor differences in specific details, wording, or sequence are acceptable. Answer N only if the prediction describes a fundamentally different outcome from what actually happens."""


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


def build_judge_prompt(preceding_sentence, continuation, ending):
    return f"""Here is the actual continuation of a story:

---BEGIN CONTINUATION---
{preceding_sentence} {continuation}
---END CONTINUATION---

Here is a predicted ending:

---BEGIN PREDICTED---
{preceding_sentence} {ending}
---END PREDICTED---

Does the predicted ending capture the general direction of the true continuation — similar characters involved, similar types of events, or a matching emotional arc? Answer Y if the broad trajectory is right, even if specific details differ. Answer N only if the prediction is fundamentally wrong about what happens. Answer ONLY Y or N.

Judgment:"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-slug", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-continuation-words", type=int,
                        default=MAX_CONTINUATION_WORDS)
    args = parser.parse_args()

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

    # Load existing results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        log.info(f"Resumed: {len(results)} stories already done")
    else:
        results = {}

    # Discover model directories
    if args.model_slug:
        model_slugs = [args.model_slug]
    else:
        model_slugs = sorted(d for d in os.listdir(ENDINGS_DIR)
                             if os.path.isdir(os.path.join(ENDINGS_DIR, d)))

    # Collect work: story-level
    stories_work = []
    for slug in model_slugs:
        endings_dir = os.path.join(ENDINGS_DIR, slug)
        story_dir = os.path.join(STORIES_DIR, slug)

        for ef in sorted(os.listdir(endings_dir)):
            if not ef.endswith("_endings.json"):
                continue

            sid = ef.replace("_endings.json", "")
            key = f"{slug}/{sid}"
            if key in results:
                continue

            ej = json.load(open(os.path.join(endings_dir, ef)))

            story_path = os.path.join(story_dir, f"{sid}.txt")
            if not os.path.exists(story_path):
                continue
            text = open(story_path, encoding="utf-8").read().strip()
            sentences = split_sentences(text)
            if len(sentences) < 3:
                continue

            story_positions = []
            for pos_data in ej["positions"]:
                pos_idx = pos_data["position"]
                pct = pos_data.get("pct_story_revealed", 0)
                if pct < PCT_MIN or pct > PCT_MAX:
                    continue

                continuation = " ".join(sentences[pos_idx:])
                if not continuation.strip():
                    continue
                continuation = truncate_words(continuation,
                                              args.max_continuation_words)
                preceding = sentences[pos_idx - 1] if pos_idx > 0 else ""

                story_positions.append({
                    "position": pos_idx,
                    "pct": pct,
                    "preceding": preceding,
                    "continuation": continuation,
                    "endings": pos_data["endings"],
                })

            if story_positions:
                stories_work.append({
                    "key": key,
                    "slug": slug,
                    "sid": sid,
                    "positions": story_positions,
                })

    total_calls = sum(len(p["endings"]) for s in stories_work for p in s["positions"])
    log.info(f"Stories: {len(stories_work)}, judge calls: {total_calls:,}")

    t_start = time.time()
    stories_done = 0

    for story in stories_work:
        key = story["key"]
        if key in results:
            continue

        results[key] = {
            "model_slug": story["slug"],
            "story_id": story["sid"],
            "positions": [],
        }

        for pos in story["positions"]:
            continuation = pos["continuation"]
            endings = pos["endings"]
            preceding = pos["preceding"]

            conversations = []
            for ending in endings:
                conversations.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_judge_prompt(
                        preceding, continuation, ending)},
                ])

            judgments = []
            for b_start in range(0, len(endings), args.batch_size):
                b_end = min(b_start + args.batch_size, len(endings))
                outputs = llm.chat(
                    conversations[b_start:b_end],
                    sampling_params,
                    chat_template_kwargs={"enable_thinking": False},
                )
                for output in outputs:
                    resp = output.outputs[0].text.strip().upper()
                    judgments.append(1 if resp.startswith("Y") else 0)

            yes_rate = sum(judgments) / len(judgments) if judgments else 0
            results[key]["positions"].append({
                "position": pos["position"],
                "pct": pos["pct"],
                "yes_rate": yes_rate,
                "judgments": judgments,
            })

        stories_done += 1
        elapsed = time.time() - t_start
        rate = stories_done / elapsed * 3600
        log.info(f"  [{stories_done}/{len(stories_work)}] {key}: "
                 f"{len(story['positions'])} pos ({rate:.0f} stories/hr)")

        if stories_done % 5 == 0:
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f)
            log.info(f"  Saved {len(results)} stories")

        gc.collect()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    log.info(f"DONE. {len(results)} stories. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
