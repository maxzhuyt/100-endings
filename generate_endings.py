#!/usr/bin/env python3
"""
Stage 2: Generate 100 endings per position for pipeline-generated stories.

Reads story text files from results/stories/{config_id}/*.txt,
splits into sentences, and at each position generates 100 diverse endings
using vLLM (Qwen3-32B, T=1.2).

Output: results/endings/{config_id}/{prompt_id}_iter{N}_endings.json

Usage:
    python generate_endings.py                      # all configs
    python generate_endings.py --config full,vanilla # specific configs
    python generate_endings.py --batch-size 32       # smaller batches
"""

import os
import sys
import re
import json
import time
import gc
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

MODEL_PATH = "/project/jevans/maxzhuyt/models/Qwen3-32B"
N_ENDINGS = 100
TEMPERATURE = 1.2
TOP_P = 0.95
MAX_TOKENS = 100
MIN_SENT_WORDS = 10

SYSTEM_PROMPT = "You are a fiction writer. Write only story text."
COMPLETION_TEMPLATE = """\
Read the following incomplete story. The ending has been removed.

Write one sentence that could end this story.

Write ONLY that sentence. No commentary, no explanation, no preamble.

STORY SO FAR:
{story_so_far}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_sentences(text):
    """Split text into sentences at natural boundaries."""
    boundaries = [0]
    for m in re.finditer(
        r'[.!?]["\u201d\u2019)]*\s+(?=[A-Z\u201c"(\[])', text
    ):
        boundaries.append(m.end())
    parts = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def active_positions(sentences, last_context_idx):
    """Positions to evaluate. Skip if sentence too short."""
    positions = []
    for i in range(1, last_context_idx + 1):
        if len(sentences[i - 1].split()) >= MIN_SENT_WORDS or i == last_context_idx:
            positions.append(i)
    return positions


def discover_stories(config_id):
    """Find all .txt story files for a given config."""
    story_dir = os.path.join(STORIES_DIR, config_id)
    if not os.path.isdir(story_dir):
        return []
    stories = []
    for fname in sorted(os.listdir(story_dir)):
        if fname.endswith(".txt") and not fname.endswith("_meta.json"):
            # Parse prompt_id and iteration from filename: p01_iter1.txt
            base = fname.replace(".txt", "")
            stories.append({
                "id": base,
                "path": os.path.join(story_dir, fname),
                "config_id": config_id,
            })
    return stories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Comma-separated config IDs (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    # Discover configs
    if args.config:
        config_ids = [c.strip() for c in args.config.split(",")]
    else:
        config_ids = sorted(d for d in os.listdir(STORIES_DIR)
                            if os.path.isdir(os.path.join(STORIES_DIR, d)))

    log.info(f"Configs to process: {config_ids}")

    # Collect all stories
    all_stories = []
    for cid in config_ids:
        stories = discover_stories(cid)
        all_stories.extend(stories)
    log.info(f"Total stories: {len(all_stories)}")

    if not all_stories:
        log.warning("No stories found. Run generate_stories.py first.")
        return

    # Load model
    log.info(f"Loading model: {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.92,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        n=N_ENDINGS,
    )

    # Process each story
    for si, story_info in enumerate(all_stories):
        config_id = story_info["config_id"]
        story_id = story_info["id"]
        story_path = story_info["path"]

        out_dir = os.path.join(ENDINGS_DIR, config_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{story_id}_endings.json")

        if os.path.exists(out_path):
            continue  # resume

        # Load and split
        with open(story_path, encoding="utf-8") as f:
            text = f.read().strip()

        sentences = split_sentences(text)
        n_sents = len(sentences)

        if n_sents < 3:
            log.warning(f"[{config_id}/{story_id}] Only {n_sents} sentences, skipping")
            continue

        # Ending = last sentence; context positions = 1..N-1
        actual_ending = sentences[-1]
        last_context_idx = n_sents - 1
        positions = active_positions(sentences, last_context_idx)
        skipped = last_context_idx - len(positions)

        log.info(f"\n[{si+1}/{len(all_stories)}] {config_id}/{story_id}: "
                 f"{n_sents} sents, {len(positions)} positions "
                 f"({skipped} skipped) × {N_ENDINGS} endings")

        # Build prompts
        prompts = []
        for i in positions:
            context = " ".join(sentences[:i])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": COMPLETION_TEMPLATE.format(
                    story_so_far=context)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(prompt)

        # Generate in batches
        t0 = time.time()
        all_results = []
        BATCH = args.batch_size

        for batch_start in range(0, len(prompts), BATCH):
            batch_end = min(batch_start + BATCH, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_positions = positions[batch_start:batch_end]

            outputs = llm.generate(batch_prompts, sampling_params)

            for pos, output in zip(batch_positions, outputs):
                endings = [o.text.strip() for o in output.outputs]
                pct = pos / n_sents * 100
                all_results.append({
                    "position": pos,
                    "n_context_sentences": pos,
                    "pct_story_revealed": round(pct, 1),
                    "endings": endings,
                })

            elapsed = time.time() - t0
            log.info(f"    Batch {batch_start // BATCH + 1}: "
                     f"positions {batch_positions[0]}-{batch_positions[-1]} "
                     f"({elapsed:.0f}s)")

        # Save
        result = {
            "story_id": story_id,
            "config_id": config_id,
            "n_sentences": n_sents,
            "n_ending_sentences": 1,
            "n_positions_evaluated": len(positions),
            "n_positions_skipped": skipped,
            "actual_ending": actual_ending,
            "n_endings_per_position": N_ENDINGS,
            "temperature": TEMPERATURE,
            "model": MODEL_PATH,
            "positions": all_results,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

        elapsed = time.time() - t0
        log.info(f"  [{config_id}/{story_id}] Done in {elapsed:.0f}s")
        gc.collect()

    log.info(f"\nAll done. Endings saved to: {ENDINGS_DIR}/")


if __name__ == "__main__":
    main()
