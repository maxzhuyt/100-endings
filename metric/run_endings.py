#!/usr/bin/env python3
"""
Ending prediction: high-level plot guesses, 100 per position.

At each sentence position, asks the model to describe in 2-3 sentences
how it thinks the story ends — a plot-level prediction, not creative prose.

Uses vLLM offline batch inference with Qwen3-32B.
"""

import os, sys, re, json, time, gc, argparse
import numpy as np

# Patch for vLLM + transformers 5.x compatibility
from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH = os.environ.get("ENDINGS_GEN_MODEL", "/project/jevans/maxzhuyt/models/Qwen3-32B")
DATA_DIR = "/project/jevans/maxzhuyt/narrative_project/NEWCORPUS_CLEANED"
RESULTS_DIR = "/project/jevans/maxzhuyt/narrative_project/results_endings"

N_ENDINGS = 100
TEMPERATURE = 1.2
TOP_P = 0.95
MAX_TOKENS = 150

MAX_STORY_ID = 200
MAX_WORD_COUNT = 5000
MIN_SENT_WORDS = 10

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a fiction writer. Write only story text."

COMPLETION_TEMPLATE = """\
In 2-3 sentences, describe how you think this story ends. No commentary, no explanation, no preamble.

STORY SO FAR:
{story_so_far}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_stories(data_dir, max_id=MAX_STORY_ID, max_words=MAX_WORD_COUNT,
                     no_id_filter=False, min_id=0):
    stories = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".txt"):
            continue
        story_id = fname.replace(".txt", "")
        if not no_id_filter:
            try:
                id_num = int(story_id)
            except ValueError:
                continue
            if id_num < min_id or id_num >= max_id:
                continue
        fpath = os.path.join(data_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        if len(text.split()) > max_words:
            continue
        stories[story_id] = fpath
    return stories


def split_sentences(text):
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
    positions = []
    for i in range(1, last_context_idx + 1):
        if len(sentences[i - 1].split()) >= MIN_SENT_WORDS or i == last_context_idx:
            positions.append(i)
    return positions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--no-id-filter", action="store_true")
    parser.add_argument("--min-id", type=int, default=0)
    parser.add_argument("--max-id", type=int, default=MAX_STORY_ID)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    data_dir = args.data_dir
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    stories = discover_stories(data_dir, max_id=args.max_id,
                               no_id_filter=args.no_id_filter, min_id=args.min_id)
    print(f"Discovered {len(stories)} stories in {data_dir} "
          f"(<= {MAX_WORD_COUNT} words)")
    if args.stories:
        ids = [s.strip().zfill(5) for s in args.stories.split(",")]
        stories = {k: v for k, v in stories.items() if k in ids}

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

    for story_id, story_path in stories.items():
        output_path = os.path.join(results_dir, f"{story_id}_endings.json")
        if os.path.exists(output_path):
            continue

        with open(story_path, "r", encoding="utf-8") as f:
            full_text = f.read().strip()

        sentences = split_sentences(full_text)
        n_sents = len(sentences)

        ending_sents = [sentences[-1]]
        actual_ending = " ".join(ending_sents)
        last_context_idx = n_sents - len(ending_sents)

        positions = active_positions(sentences, last_context_idx)
        skipped = last_context_idx - len(positions)

        print(f"\n{'='*70}")
        print(f"[{story_id}]")
        print(f"  {n_sents} sentences, {len(full_text)} chars")
        print(f"  Actual ending: {actual_ending[:120]}...")
        print(f"  Evaluating {len(positions)}/{last_context_idx} positions "
              f"({skipped} skipped) × {N_ENDINGS} endings")

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
            print(f"    Batch {batch_start // BATCH + 1}: "
                  f"positions {batch_positions[0]}-{batch_positions[-1]} "
                  f"({elapsed:.0f}s)")

        result = {
            "story_id": story_id,
            "n_sentences": n_sents,
            "n_ending_sentences": len(ending_sents),
            "n_positions_evaluated": len(positions),
            "n_positions_skipped": skipped,
            "actual_ending": actual_ending,
            "n_endings_per_position": N_ENDINGS,
            "temperature": TEMPERATURE,
            "model": MODEL_PATH,
            "positions": all_results,
        }
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        elapsed = time.time() - t0
        print(f"  [{story_id}] Done in {elapsed:.0f}s")
        gc.collect()

    print(f"\n{'='*70}")
    print("Generation complete.")


if __name__ == "__main__":
    main()
