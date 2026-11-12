# 100-Endings metric: the code that produced the paper's numbers

These are the exact scripts behind every no-rate in the paper. Only two things were changed:
the model paths can now be overridden with environment variables, and one docstring was corrected.
Default paths point to the authors' cluster.

| Stage | Human corpora (Tables 1, 6) | EQ-Bench stories (Tables 2, 7, 29) |
|---|---|---|
| 1. Ending generation, Qwen3-32B | `run_endings.py` | `run_cwb_endings.py` (repeats: `run_cwb_endings_rep.py`) |
| 2. Judge, Qwen3-8B | `compute_judge.py` | `compute_cwb_judge.py` / `compute_cwb_judge_rep.py` (repeats: `run_judge_rep.py`) |
| 3. Curve statistics | `curve_metrics.py` | `curve_metrics.py` |

Each corpus has a pair of scripts. Within a pair, the prompts, sampling parameters, sentence splitter and
position filter are identical.

## Settings (Appendix C)

- **Sentence splitting.** Regex: `[.!?]["”’)]*\s+(?=[A-Z“"(\[])`.
- **Positions evaluated.** A position is a sentence boundary: the model sees sentences 1..i, and
  pct = i / (number of sentences) × 100. The judge keeps positions with 10 ≤ pct ≤ 99.
  - A position is skipped when sentence i has fewer than 10 words, except the last context position.
  - Human-corpus stories longer than 5,000 words are excluded.
- **Generator.** Qwen3-32B in vLLM: `max_model_len=8192`, prefix caching on, thinking off.
  Sampling: T = 1.2, top-p = 0.95, `max_tokens=150`, n = 100 per position.
  - System prompt: "You are a fiction writer. Write only story text."
  - User prompt: "In 2-3 sentences, describe how you think this story ends. No commentary, no
    explanation, no preamble.\n\nSTORY SO FAR:\n{prefix}".
- **Judge.** Qwen3-8B, T = 0, `max_tokens=10`, `max_model_len=16384`, thinking off.
  - It judges each of the 100 endings in its own prompt, using the lenient "general direction" criterion
    (system prompt in `compute_judge.py`).
  - The actual continuation and the predicted ending are both prefixed with the preceding sentence.
  - The continuation is cut to its **first** 2,000 words. At early positions of stories longer than that,
    the true ending is therefore not shown to the judge (see `../ERRATA.md` A2).
  - The verdict is Y if the output starts with "Y".
- **Statistics** (`curve_metrics.py`).
  - no-rate(i) = 1 − (Y verdicts)/100.
  - Mean NR: averaged over all evaluated positions. Late NR: averaged over positions with pct ≥ 80.
  - Retain: post-spike retention on the raw curve.
  - IR_α: vertex-angle inflection rate on the moving-average (w = 3) curve, rescaled to the unit square.

## Running on your own stories

```bash
export ENDINGS_GEN_MODEL=/path/to/Qwen3-32B ENDINGS_JUDGE_MODEL=/path/to/Qwen3-8B
# stories: one .txt per story in a folder
python metric/run_endings.py --data-dir my_stories/ --results-dir out/endings --no-id-filter
```

The judge scripts read endings from fixed per-corpus folders. Edit `CATEGORIES` in `compute_judge.py`, or
`BASE` in `compute_cwb_judge.py` (which reads `results/endings/{model_slug}/`), to point at your output.
`sbatch/` has the SLURM files that ran each stage, with one H200 GPU per job.

## Files this does not include

- `archive-2026-03-20` (git tag): the earlier version of the metric, with one-sentence endings and a strict
  judge that took all 100 endings in a single call. It produced the March ablation numbers. No number in the
  published paper uses it.
