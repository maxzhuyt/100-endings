# Narrative generation pipeline (Section 4, Appendix E)

The pipeline has three steps: Step 0 warmup, Step 1 beat sheet ("to-do list"), and Step 2 story.
It is implemented in `core/narrative_pipeline.py`; `core/idea_generation.py` is the optional idea generator.

## How the paper's stories were generated

The Table 2 and Appendix J stories were generated through the EQ-Bench harness in our public fork,
[maxzhuyt/creative-writing-bench @ 68ba9d3](https://github.com/maxzhuyt/creative-writing-bench/tree/68ba9d3).
That fork's `core/narrative_pipeline.py` is the file in this folder, minus the optional instruction
wrapper on `VanillaStep`, which was added later for the ablation.

| Condition | Command (run inside the fork) |
|---|---|
| Opus 4.6 + pipeline | `python creative_writing_bench.py --test-model anthropic/claude-opus-4-6 --judge-model anthropic/claude-sonnet-4 --step0 precomputed --step0-templates data/narrative_step0_templates.json --step1 default --iterations 1 --threads 6 --run-id narrative_opus46 --no-elo` |
| GPT-5.2 + pipeline | the same command with `--test-model openai/gpt-5.2 --run-id narrative_gpt52` |
| Sonnet 4.6 + pipeline | the same command with `--test-model anthropic/claude-sonnet-4-6 --run-id narrative_sonnet46` |
| Sonnet 4.6, no Step 0 | `--step0 none --step1 default --run-id narrative_sonnet46_nostep0` |
| Vanilla (zero-shot) | runs `run1__claude-sonnet-4-6`, `run1__claude-opus-4-6`, `run1__openai_gpt-5_2` of the same harness (EQ-Bench prompt sent as is) |

- **Step 0.** One of six precomputed beat sheets is drawn at random for each story (`data/narrative_step0_templates.json`):
  - A1, A2 and A4 are title-mode analyses of *Omelas*, *The Lottery* and *The Yellow Wallpaper*.
  - B1, B2 and B3 are full-text analyses of the New Yorker stories *The Fellow*, *Poor Girl* and *The Dog*.
  - Each story's draw is recorded in `data/stories/<condition>/pipeline_steps.jsonl.gz`, together with its Step 0 and Step 1 outputs.
- **Sonnet 4.6 + pipeline commit.** This run (18 March 2026) predates the first pipeline commit in the fork (19 March, f4f705e).
  - Its run log shows the same six precomputed templates.
  - It stored its outputs under the field names `narrative_step0` and `narrative_step1`.
- **Shared generation settings** (Appendix E.2):
  - T = 0.7, min_p = 0.1, `max_tokens=4000` per call.
  - Reasoning is disabled, and no system prompt is sent.

## Limitations of this code as used in the paper

See [`../ERRATA.md`](../ERRATA.md) for the measured consequences of each point.

1. **The story writer never sees the original prompt.** Step 2's prompt contains only the Step 1 beat sheet:
   "Here is a structural to-do list for a short story: --- {beats} --- Write the story now. Follow the structural beats closely. Write only the story — no commentary."
   - It never sees the EQ-Bench prompt or its 1,000-word target.
   - Pipeline stories are therefore much longer than the vanilla ones.
2. **Stories can be cut off without notice.** The stop reason is not recorded or checked. A story that reaches the 4,000-token cap ends mid-sentence, and its "ending" is the truncation point.
3. **There is no tension system prompt.** Section 4 of the paper mentions "a system-prompt emphasis on sustaining tension", but no system prompt is sent at any step.

Anyone re-running the pipeline should raise `max_tokens`, record `finish_reason` / `stop_reason` and
regenerate when it reports the cap, and pass the prompt and word target to Step 2.

## Standalone runner (used for the March ablation)

`generate_stories.py` runs the same steps without the EQ-Bench harness, on `data/story_ideas.json` or with
idea generation (`--idea-gen`, Screenwriters Taxonomy + `data/tropes_filtered.jsonl`). It reads an OpenRouter key
from `.env` (`TEST_API_KEY`, `TEST_API_URL`). Run `python pipeline/generate_stories.py --help` for the options.
The configurations used in `experiments/ablation_17config/` are listed in that folder's README.

The full-text warmup mode (`--step0 fulltext`) needs New Yorker story texts in `data/nyer_sources/`. Those texts
cannot be redistributed and are not included in this repository.
