#!/usr/bin/env python3
"""
CWB judge for repeat runs.
Usage: python compute_cwb_judge_rep.py --endings-base /path/to/endings_rep2
       --output /path/to/judge_rep2/ending_judge.json
       [--model-slugs "slug1,slug2,..."]
"""

import os, json, re, gc, sys, time, argparse
import logging

from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: self.all_special_tokens)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

BASE = "/project/jevans/maxzhuyt/creative-writing-bench"
STORIES_DIR = f"{BASE}/stories"
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
        if chunk: parts.append(chunk)
    return parts


def truncate_words(text, max_words):
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endings-base", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-slugs", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    log.info(f"Loading model {MODEL_PATH}...")
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=1, gpu_memory_utilization=0.92,
              max_model_len=16384, trust_remote_code=True, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    # Resume
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if os.path.exists(args.output):
        results = json.load(open(args.output))
        log.info(f"Resumed: {len(results)} stories")
    else:
        results = {}

    # Discover models
    if args.model_slugs:
        model_slugs = [s.strip() for s in args.model_slugs.split(",")]
    else:
        model_slugs = sorted(d for d in os.listdir(args.endings_base)
                             if os.path.isdir(os.path.join(args.endings_base, d)))

    # Collect work
    stories_work = []
    for slug in model_slugs:
        endings_dir = os.path.join(args.endings_base, slug)
        story_dir = os.path.join(STORIES_DIR, slug)
        if not os.path.isdir(endings_dir):
            continue

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
                continuation = truncate_words(" ".join(sentences[pos_idx:]), MAX_CONTINUATION_WORDS)
                if not continuation.strip():
                    continue
                preceding = sentences[pos_idx - 1] if pos_idx > 0 else ""
                story_positions.append({
                    "position": pos_idx, "pct": pct,
                    "preceding": preceding, "continuation": continuation,
                    "endings": pos_data["endings"],
                })

            if story_positions:
                stories_work.append({"key": key, "slug": slug, "sid": sid,
                                     "positions": story_positions})

    log.info(f"Stories to judge: {len(stories_work)}")
    t_start = time.time()
    done = 0

    for story in stories_work:
        key = story["key"]
        results[key] = {"model_slug": story["slug"], "story_id": story["sid"], "positions": []}

        for pos in story["positions"]:
            conversations = [[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(
                    pos["preceding"], pos["continuation"], e)},
            ] for e in pos["endings"]]

            judgments = []
            for b_start in range(0, len(pos["endings"]), args.batch_size):
                b_end = min(b_start + args.batch_size, len(pos["endings"]))
                outputs = llm.chat(conversations[b_start:b_end], sampling_params,
                                    chat_template_kwargs={"enable_thinking": False})
                for o in outputs:
                    judgments.append(1 if o.outputs[0].text.strip().upper().startswith("Y") else 0)

            results[key]["positions"].append({
                "position": pos["position"], "pct": pos["pct"],
                "yes_rate": sum(judgments) / len(judgments) if judgments else 0,
                "judgments": judgments,
            })

        done += 1
        if done % 5 == 0:
            with open(args.output, "w") as f:
                json.dump(results, f)
            log.info(f"  [{done}/{len(stories_work)}] ({(time.time()-t_start)/60:.0f}min)")
        gc.collect()

    with open(args.output, "w") as f:
        json.dump(results, f)
    log.info(f"DONE. {done} stories → {args.output}")


if __name__ == "__main__":
    main()
