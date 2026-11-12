#!/usr/bin/env python3
"""
Generate 100 endings per position for creative-writing-bench stories.

Iterates over stories/{model_slug}/ directories, generates endings for each
story, saves to results/endings/{model_slug}/.

Supports sharding via --shard-id and --n-shards for multi-GPU parallelism.
Resume-safe: skips stories with existing _endings.json.

Usage:
    python run_cwb_endings.py --shard-id 0 --n-shards 4
"""

import os, sys, re, json, time, gc, argparse
import numpy as np

# Patch for vLLM + transformers 5.x compatibility
from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

BASE = "/project/jevans/maxzhuyt/creative-writing-bench"
STORIES_DIR = f"{BASE}/stories"
RESULTS_DIR = f"{BASE}/results/endings"

MODEL_PATH = os.environ.get("ENDINGS_GEN_MODEL", "/project/jevans/maxzhuyt/models/Qwen3-32B")
N_ENDINGS = 100
TEMPERATURE = 1.2
TOP_P = 0.95
MAX_TOKENS = 150
MAX_WORD_COUNT = 5000
MIN_SENT_WORDS = 10

SYSTEM_PROMPT = "You are a fiction writer. Write only story text."

COMPLETION_TEMPLATE = """\
In 2-3 sentences, describe how you think this story ends. No commentary, no explanation, no preamble.

STORY SO FAR:
{story_so_far}"""


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


def active_positions(sentences, last_context_idx):
    positions = []
    for i in range(1, last_context_idx + 1):
        if len(sentences[i - 1].split()) >= MIN_SENT_WORDS or i == last_context_idx:
            positions.append(i)
    return positions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-slugs", type=str, default=None,
                        help="Comma-separated model slugs to process (default: all)")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    # Discover model directories
    if args.model_slugs:
        model_slugs = [s.strip() for s in args.model_slugs.split(",")]
        # Validate
        model_slugs = [s for s in model_slugs
                       if os.path.isdir(os.path.join(STORIES_DIR, s))]
    else:
        model_slugs = sorted(d for d in os.listdir(STORIES_DIR)
                             if os.path.isdir(os.path.join(STORIES_DIR, d)))

    # Shard by model
    my_slugs = model_slugs[args.shard_id::args.n_shards]
    print(f"Shard {args.shard_id}/{args.n_shards}: {len(my_slugs)} models "
          f"(of {len(model_slugs)} total)")
    print(f"Models: {my_slugs[:5]}{'...' if len(my_slugs) > 5 else ''}")

    # Count total stories
    total_stories = 0
    for slug in my_slugs:
        story_dir = os.path.join(STORIES_DIR, slug)
        total_stories += len([f for f in os.listdir(story_dir) if f.endswith(".txt")])
    print(f"Total stories in shard: {total_stories}")

    # Load model (identical to run_endings_scaled.py)
    print(f"Loading model: {MODEL_PATH}")
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

    processed = 0
    skipped = 0
    t_start = time.time()
    BATCH = args.batch_size

    for slug_idx, slug in enumerate(my_slugs):
        story_dir = os.path.join(STORIES_DIR, slug)
        out_dir = os.path.join(RESULTS_DIR, slug)
        os.makedirs(out_dir, exist_ok=True)

        story_files = sorted(f for f in os.listdir(story_dir) if f.endswith(".txt"))
        print(f"\n{'='*70}")
        print(f"[{slug_idx+1}/{len(my_slugs)}] {slug}: {len(story_files)} stories")

        for sf in story_files:
            sid = sf.replace(".txt", "")
            out_path = os.path.join(out_dir, f"{sid}_endings.json")

            # Resume check
            if os.path.exists(out_path):
                skipped += 1
                continue

            text = open(os.path.join(story_dir, sf), encoding="utf-8").read().strip()
            if len(text.split()) > MAX_WORD_COUNT:
                continue

            sentences = split_sentences(text)
            if len(sentences) < 3:
                continue

            actual_ending = [sentences[-1]]
            last_ctx = len(sentences) - 1
            positions = active_positions(sentences, last_ctx)
            n_skipped = last_ctx - len(positions)

            print(f"\n[{sid}]")
            print(f"  {len(sentences)} sentences, {len(text)} chars")
            print(f"  Actual ending ({len(actual_ending)} sents): "
                  f"{actual_ending[0][:60]}...")
            print(f"  Evaluating {len(positions)}/{last_ctx} positions "
                  f"({n_skipped} skipped, short sentences) x {N_ENDINGS} endings")

            # Build all prompts with tokenizer (matching run_endings_scaled.py)
            prompts = []
            for pos in positions:
                context = " ".join(sentences[:pos])
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": COMPLETION_TEMPLATE.format(
                        story_so_far=context)},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                prompts.append(prompt)

            # Generate in batches (matching run_endings_scaled.py)
            t0 = time.time()
            position_results = []

            for batch_start in range(0, len(prompts), BATCH):
                batch_end = min(batch_start + BATCH, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                batch_positions = positions[batch_start:batch_end]

                outputs = llm.generate(batch_prompts, sampling_params)

                for pos, output in zip(batch_positions, outputs):
                    endings = [o.text.strip() for o in output.outputs]
                    pct = round(pos / len(sentences) * 100, 1)
                    position_results.append({
                        "position": pos,
                        "n_context_sentences": pos,
                        "pct_story_revealed": pct,
                        "endings": endings,
                    })

                elapsed_batch = time.time() - t0
                print(f"    Batch {batch_start // BATCH + 1}: "
                      f"positions {batch_positions[0]}-{batch_positions[-1]} "
                      f"({elapsed_batch:.0f}s)")

            result = {
                "story_id": sid,
                "model_slug": slug,
                "n_sentences": len(sentences),
                "n_ending_sentences": len(actual_ending),
                "n_positions_evaluated": len(positions),
                "n_positions_skipped": n_skipped,
                "actual_ending": " ".join(actual_ending),
                "n_endings_per_position": N_ENDINGS,
                "temperature": TEMPERATURE,
                "model": MODEL_PATH,
                "positions": position_results,
            }

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

            processed += 1
            elapsed = time.time() - t_start
            rate = processed / elapsed * 60
            print(f"  Saved. Total: {processed} done, {skipped} skipped, "
                  f"{rate:.1f} stories/min")

            gc.collect()

    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"DONE. Processed: {processed}, Skipped: {skipped}, "
          f"Time: {elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
