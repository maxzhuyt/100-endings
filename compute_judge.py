#!/usr/bin/env python3
"""
Stage 4: LLM-as-judge evaluation of generated endings.

For each story position, asks Qwen3-8B whether each of the 100 generated
endings describes an event/idea that actually occurs in the true continuation.

Output: results/judge/{config_id}_judge.json

Usage:
    python compute_judge.py                         # all configs
    python compute_judge.py --config full,vanilla   # specific configs
    python compute_judge.py --batch-size 128
"""

import os
import sys
import re
import json
import gc
import time
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = "/project/jevans/maxzhuyt/narrative_ablation"
STORIES_DIR = os.path.join(BASE_DIR, "results", "stories")
ENDINGS_DIR = os.path.join(BASE_DIR, "results", "endings")
JUDGE_DIR = os.path.join(BASE_DIR, "results", "judge")

MODEL_PATH = "/project/jevans/maxzhuyt/models/Qwen3-8B"
PCT_MIN, PCT_MAX = 10, 99
MAX_CONTINUATION_WORDS = 2000

SYSTEM_PROMPT = """You are a precise literary judge. You determine whether a generated story ending describes an event or idea that actually occurs in the true story continuation. Answer Y if the core event/action in the generated ending matches something in the continuation, even if wording differs. Answer N if the event does not occur. Be strict: vague thematic similarity is not enough; the specific event or action must occur."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def build_judge_prompt(continuation, endings):
    endings_block = "\n".join(f"[{i+1}] {e}" for i, e in enumerate(endings))
    return f"""Here is the actual continuation of a story:

---BEGIN CONTINUATION---
{continuation}
---END CONTINUATION---

Below are {len(endings)} generated endings (each one sentence). For each, answer whether the event, action, or idea described in the generated ending actually occurs in the actual continuation above.

Answer ONLY with a comma-separated list of Y or N, one per ending, in order. Example for 5 endings: Y,N,N,Y,N

{endings_block}

Judgments:"""


def parse_judgments(response, expected_n):
    response = response.strip().upper()
    parts = re.split(r'[,\s]+', response)
    judgments = []
    for p in parts:
        p = p.strip().strip('.')
        if p in ('Y', 'YES', '1'):
            judgments.append(1)
        elif p in ('N', 'NO', '0'):
            judgments.append(0)
        elif ':' in p:
            val = p.split(':')[-1].strip()
            if val in ('Y', 'YES', '1'):
                judgments.append(1)
            elif val in ('N', 'NO', '0'):
                judgments.append(0)

    if len(judgments) < expected_n:
        log.warning(f"  Parse: got {len(judgments)}/{expected_n} judgments. "
                    f"Response: '{response[:100]}...'")
        judgments.extend([-1] * (expected_n - len(judgments)))
    elif len(judgments) > expected_n:
        judgments = judgments[:expected_n]

    return judgments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-continuation-words", type=int,
                        default=MAX_CONTINUATION_WORDS)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    os.makedirs(JUDGE_DIR, exist_ok=True)

    # Discover configs
    if args.config:
        config_ids = [c.strip() for c in args.config.split(",")]
    else:
        config_ids = sorted(d for d in os.listdir(ENDINGS_DIR)
                            if os.path.isdir(os.path.join(ENDINGS_DIR, d)))

    log.info(f"Configs: {config_ids}")

    # Load model
    log.info(f"Loading judge model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        max_model_len=16384,
        trust_remote_code=True,
        dtype="float16",
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        stop=["\n\n"],
    )

    for config_id in config_ids:
        endings_dir = os.path.join(ENDINGS_DIR, config_id)
        if not os.path.isdir(endings_dir):
            log.warning(f"[{config_id}] No endings dir, skipping")
            continue

        output_path = os.path.join(JUDGE_DIR, f"{config_id}_judge.json")

        # Load checkpoint
        if os.path.exists(output_path):
            results = json.load(open(output_path))
            log.info(f"[{config_id}] Resuming from {len(results)} entries")
        else:
            results = {}

        # Collect work items
        work_items = []
        endings_files = sorted(f for f in os.listdir(endings_dir)
                               if f.endswith("_endings.json"))

        for ef in endings_files:
            story_id = ef.replace("_endings.json", "")
            if story_id in results:
                continue

            ej = json.load(open(os.path.join(endings_dir, ef)))

            # Load story text
            story_path = os.path.join(STORIES_DIR, config_id, f"{story_id}.txt")
            if not os.path.exists(story_path):
                log.warning(f"  [{story_id}] story text missing, skipping")
                continue

            story_text = open(story_path, encoding="utf-8").read().strip()
            sentences = split_sentences(story_text)
            if len(sentences) < 3:
                continue

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

                work_items.append({
                    "sid": story_id,
                    "position": pos_idx,
                    "pct": pct,
                    "continuation": continuation,
                    "endings": pos_data["endings"],
                })

        log.info(f"[{config_id}] {len(work_items)} positions to judge")

        if not work_items:
            continue

        # Process in batches
        batch_size = args.batch_size
        n_batches = (len(work_items) + batch_size - 1) // batch_size
        save_interval = 10

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(work_items))
            batch = work_items[start:end]

            # Build conversations
            conversations = []
            for item in batch:
                user_msg = build_judge_prompt(
                    item["continuation"], item["endings"])
                conversations.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ])

            t0 = time.time()
            outputs = llm.chat(
                conversations,
                sampling_params,
                chat_template_kwargs={"enable_thinking": False},
            )
            dt = time.time() - t0

            # Parse results
            for item, output in zip(batch, outputs):
                response = output.outputs[0].text.strip()
                judgments = parse_judgments(response, len(item["endings"]))

                sid = item["sid"]
                if sid not in results:
                    results[sid] = {"config_id": config_id, "positions": []}

                results[sid]["positions"].append({
                    "position": item["position"],
                    "pct": item["pct"],
                    "yes_rate": (sum(1 for j in judgments if j == 1)
                                 / len(judgments) if judgments else 0),
                    "judgments": judgments,
                })

            log.info(f"  Batch {batch_idx+1}/{n_batches}: {len(batch)} pos "
                     f"in {dt:.1f}s ({len(batch)/dt:.1f} pos/s)")

            if (batch_idx + 1) % save_interval == 0 or batch_idx == n_batches - 1:
                with open(output_path, "w") as f:
                    json.dump(results, f)
                log.info(f"  Saved {len(results)} stories to {output_path}")

        # Final save
        with open(output_path, "w") as f:
            json.dump(results, f)
        log.info(f"[{config_id}] Done. {len(results)} stories → {output_path}")

    log.info("\nAll judge evaluation complete.")


if __name__ == "__main__":
    main()
