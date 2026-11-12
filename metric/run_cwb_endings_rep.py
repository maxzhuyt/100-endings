#!/usr/bin/env python3
"""
CWB endings generation for repeat runs.
Usage: python run_cwb_endings_rep.py --rep 2 --model-slugs "slug1,slug2,..."
       --results-base /path/to/endings_rep2

Same as run_cwb_endings.py but writes to a separate rep directory.
"""

import os, sys, re, json, time, gc, argparse
import numpy as np

from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

BASE = "/project/jevans/maxzhuyt/creative-writing-bench"
STORIES_DIR = f"{BASE}/stories"

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
    parser.add_argument("--model-slugs", type=str, required=True)
    parser.add_argument("--results-base", type=str, required=True,
                        help="Base dir for this rep, e.g., results/endings_rep2")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_slugs = [s.strip() for s in args.model_slugs.split(",")]
    model_slugs = [s for s in model_slugs if os.path.isdir(os.path.join(STORIES_DIR, s))]

    print(f"Models: {len(model_slugs)}")
    total = sum(len([f for f in os.listdir(os.path.join(STORIES_DIR, s)) if f.endswith('.txt') and f.startswith('i1_')])
                for s in model_slugs)
    print(f"Total stories: {total}")

    llm = LLM(model=MODEL_PATH, dtype="bfloat16", tensor_parallel_size=1,
              max_model_len=8192, enable_prefix_caching=True,
              gpu_memory_utilization=0.92, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=TOP_P,
                                     max_tokens=MAX_TOKENS, n=N_ENDINGS)

    processed = 0
    t_start = time.time()
    BATCH = args.batch_size

    for slug in model_slugs:
        story_dir = os.path.join(STORIES_DIR, slug)
        out_dir = os.path.join(args.results_base, slug)
        os.makedirs(out_dir, exist_ok=True)

        story_files = sorted(f for f in os.listdir(story_dir)
                             if f.endswith(".txt") and f.startswith("i1_"))
        print(f"\n{'='*70}")
        print(f"{slug}: {len(story_files)} stories → {out_dir}")

        for sf in story_files:
            sid = sf.replace(".txt", "")
            out_path = os.path.join(out_dir, f"{sid}_endings.json")
            if os.path.exists(out_path):
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

            prompts = []
            for pos in positions:
                context = " ".join(sentences[:pos])
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": COMPLETION_TEMPLATE.format(story_so_far=context)},
                ]
                prompts.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))

            t0 = time.time()
            position_results = []
            for batch_start in range(0, len(prompts), BATCH):
                batch_end = min(batch_start + BATCH, len(prompts))
                outputs = llm.generate(prompts[batch_start:batch_end], sampling_params)
                for pos, output in zip(positions[batch_start:batch_end], outputs):
                    endings = [o.text.strip() for o in output.outputs]
                    pct = round(pos / len(sentences) * 100, 1)
                    position_results.append({
                        "position": pos, "n_context_sentences": pos,
                        "pct_story_revealed": pct, "endings": endings,
                    })

            result = {
                "story_id": sid, "model_slug": slug,
                "n_sentences": len(sentences),
                "n_ending_sentences": len(actual_ending),
                "n_positions_evaluated": len(positions),
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
            print(f"  [{processed}] {sid} ({len(positions)} pos, {time.time()-t0:.0f}s)")
            gc.collect()

    print(f"\nDone. {processed} stories in {(time.time()-t_start)/3600:.1f}h")


if __name__ == "__main__":
    main()
