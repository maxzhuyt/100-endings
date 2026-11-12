#!/usr/bin/env python3
"""
LLM-as-judge for ending predictions (individual judging).

Each of the 100 predicted endings is judged independently against the
actual story continuation. Uses Qwen3-8B as judge (MODEL_PATH below).

Output: results_endings/ending_judge.json
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

BASE = "/project/jevans/maxzhuyt/narrative_project"

CATEGORIES = {
    "nyer_83": {
        "endings": f"{BASE}/results_endings/nyer_83",
        "corpus":  f"{BASE}/NYER_83",
    },
    "nyer_expanded": {
        "endings": f"{BASE}/results_endings/nyer_expanded",
        "corpus":  f"{BASE}/NYER_EXPANDED",
    },
    "gutenberg": {
        "endings": f"{BASE}/results_endings/gutenberg",
        "corpus":  f"{BASE}/BASELINE_CORPUS/gutenberg",
    },
    "storystar": {
        "endings": f"{BASE}/results_endings/storystar",
        "corpus":  f"{BASE}/BASELINE_CORPUS/storystar/sampled",
    },
    "wp_human": {
        "endings": f"{BASE}/results_endings/wp_human",
        "corpus":  f"{BASE}/BASELINE_CORPUS/ghostbuster-data/wp/human",
    },
    "tell_me_a_story": {
        "endings": f"{BASE}/results_endings/tell_me_a_story",
        "corpus":  None,
    },
}

TMAS_STORIES = f"{BASE}/BASELINE_CORPUS/tell_me_a_story/all_stories.jsonl"
MODEL_PATH   = os.environ.get("ENDINGS_JUDGE_MODEL", "/project/jevans/maxzhuyt/models/Qwen3-8B")
OUTPUT_FILE  = f"{BASE}/results_endings/ending_judge.json"
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
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--max-continuation-words", type=int, default=MAX_CONTINUATION_WORDS)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=str, default=OUTPUT_FILE,
                        help="Output JSON path (for sharded runs)")
    parser.add_argument("--sid-min", type=int, default=None,
                        help="Min numeric story ID (inclusive)")
    parser.add_argument("--sid-max", type=int, default=None,
                        help="Max numeric story ID (exclusive)")
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

    # Load TMAS texts
    tmas_texts = {}
    if os.path.exists(TMAS_STORIES):
        with open(TMAS_STORIES) as f:
            for line in f:
                r = json.loads(line)
                tmas_texts[r['story_id']] = r['text']

    # Resume
    output_file = args.output
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(output_file):
        with open(output_file) as f:
            results = json.load(f)
        log.info(f"Resumed: {len(results)} stories already done")
    else:
        results = {}

    # Collect work
    stories_work = []

    for group_name, cfg in CATEGORIES.items():
        if args.group and group_name != args.group:
            continue
        endings_dir = cfg["endings"]
        corpus_dir = cfg["corpus"]
        if not os.path.exists(endings_dir):
            log.warning(f"  {group_name}: endings dir missing, skipping")
            continue

        endings_files = sorted(f for f in os.listdir(endings_dir)
                               if f.endswith("_endings.json"))
        log.info(f"{group_name}: {len(endings_files)} stories")

        for ef in endings_files:
            sid = ef.replace("_endings.json", "")
            if sid in results:
                continue
            # Optional numeric ID filtering
            if args.sid_min is not None or args.sid_max is not None:
                try:
                    sid_num = int(sid)
                    if args.sid_min is not None and sid_num < args.sid_min:
                        continue
                    if args.sid_max is not None and sid_num >= args.sid_max:
                        continue
                except ValueError:
                    pass

            ej = json.load(open(os.path.join(endings_dir, ef)))

            # Load story text
            story_text = None
            if group_name == "tell_me_a_story":
                source_id = ej.get("source_id")
                if source_id is not None:
                    story_text = tmas_texts.get(source_id, "")
                else:
                    sid_num = sid.replace("tmas_", "")
                    try:
                        story_text = tmas_texts.get(int(sid_num), "")
                    except ValueError:
                        pass
            elif group_name in ("wp_human", "wp_gpt"):
                source_file = ej.get("source_file", "")
                if source_file and corpus_dir:
                    spath = os.path.join(corpus_dir, source_file)
                    if os.path.exists(spath):
                        story_text = open(spath).read().strip()
                if not story_text and corpus_dir:
                    spath = os.path.join(corpus_dir, f"{sid}.txt")
                    if os.path.exists(spath):
                        story_text = open(spath).read().strip()
            else:
                if corpus_dir:
                    # Try with and without leading zeros
                    for fname in [f"{sid}.txt", sid + ".txt"]:
                        path = os.path.join(corpus_dir, fname)
                        if os.path.exists(path):
                            story_text = open(path, encoding="utf-8").read().strip()
                            break

            if not story_text:
                log.warning(f"  {sid}: no story text, skipping")
                continue

            sentences = split_sentences(story_text)
            if len(sentences) < 3:
                continue

            story_positions = []
            for pos_data in ej['positions']:
                pos_idx = pos_data['position']
                pct = pos_data.get('pct_story_revealed', 0)
                if pct < PCT_MIN or pct > PCT_MAX:
                    continue

                continuation = " ".join(sentences[pos_idx:])
                if not continuation.strip():
                    continue
                continuation = truncate_words(continuation, args.max_continuation_words)
                preceding = sentences[pos_idx - 1] if pos_idx > 0 else ""

                story_positions.append({
                    "position": pos_idx,
                    "pct": pct,
                    "preceding": preceding,
                    "continuation": continuation,
                    "endings": pos_data['endings'],
                })

            if story_positions:
                stories_work.append({
                    "sid": sid,
                    "group": group_name,
                    "positions": story_positions,
                })

    total_calls = sum(len(p["endings"]) for s in stories_work for p in s["positions"])
    log.info(f"Stories: {len(stories_work)}, judge calls: {total_calls:,}")

    t_start = time.time()
    stories_done = 0

    for si, story in enumerate(stories_work):
        sid = story["sid"]
        group = story["group"]
        if sid in results:
            continue

        results[sid] = {"group": group, "positions": []}

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
            results[sid]["positions"].append({
                "position": pos["position"],
                "pct": pos["pct"],
                "yes_rate": yes_rate,
                "judgments": judgments,
            })

        stories_done += 1
        elapsed = time.time() - t_start
        rate = stories_done / elapsed * 3600
        log.info(f"  [{stories_done}/{len(stories_work)}] {sid} ({group}): "
                 f"{len(story['positions'])} pos ({rate:.0f} stories/hr)")

        if stories_done % 5 == 0:
            with open(output_file, "w") as f:
                json.dump(results, f)
            log.info(f"  Saved {len(results)} stories")

        gc.collect()

    with open(output_file, "w") as f:
        json.dump(results, f)
    log.info(f"DONE. {len(results)} stories. Output: {output_file}")


if __name__ == "__main__":
    main()
