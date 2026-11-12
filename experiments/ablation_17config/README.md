# March 2026 pipeline ablation (17 configurations): withdrawn

This ablation ran in March 2026 with Claude Opus 4.6 on the standalone runner (`pipeline/generate_stories.py`).
It is **not part of the published paper**. Its write-up, [`ANALYSIS.md`](ANALYSIS.md), states four hypotheses
(H1–H4: idea generation, tropes, taxonomy, number of components) and a component ranking.

**Decision: every configuration is invalid for 100-Endings measurement, and all conclusions in `ANALYSIS.md`
are withdrawn.** The story texts are kept for reference only.

## Why

1. **Almost every story is cut off.**
   - Every configuration ran with `max_tokens=4000` and no stop-reason check (see `pipeline/README.md`).
   - Between 67% and 100% of the judged stories in each configuration end mid-sentence. This includes all three
     baselines: `vanilla` 43/45 judged, `instruction_vanilla` 15/15, `idea_vanilla` 14/16.
   - The 100-Endings judge treats the cut-off point as the story's ending, so late no-rate, and the comparisons
     between configurations, measure where each generation hit the cap.
2. **The length filter selected stories at the cap.**
   - `retry_unqualified.py` kept or regenerated stories until they fell in 2,000–3,000 words. That band is just
     under the 4,000-token cap, so the filter favored the stories that reached it.
   - The configuration medians are 2,604–2,853 words.
3. **The writer never saw a length target.** As in the paper's pipeline, Step 2 sees only the beat sheet.
   The ablation ideas also carry no word target, and the vanilla baselines send the raw idea with no target either.
4. **The numbers in `ANALYSIS.md` come from a judge the paper does not use.**
   - They were produced by the earlier strict judge (git tag `archive-2026-03-20`).
   - Re-judging with the paper's judge on 2026-03-29 (`rescore_2026-03-28/`) moved every configuration mean from
     0.75–0.89 to 0.45–0.74.
5. **Design defects.**
   - `idea_no_taxonomy_bugged` has the same flags as `idea_tropes_only`: it is a second run of that configuration.
   - The intended `idea_no_taxonomy` never ran (its directory is empty), so there is no taxonomy-only contrast.
   - Several configurations were judged on 15–18 of their 45–114 stories.
   - 10 outputs are planning text or fragments, not stories (`analysis/audit/non_story.csv`).

## Per-configuration audit

All columns are counted over **all story files** in the configuration, except "judged", which counts the
stories the paper's judge scored on 2026-03-29. Source: `analysis/audit/per_story.csv`.

| Configuration | stories | end mid-sentence | median words | judged | judged and complete | Verdict |
|---|---|---|---|---|---|---|
| `vanilla` | 45 | 43 | 2,625 | 45 | 2 | invalid |
| `instruction_vanilla` | 45 | 43 | 2,682 | 15 | 0 | invalid |
| `short_beats` | 45 | 40 | 2,853 | 45 | 5 | invalid |
| `full_title` | 45 | 40 | 2,769 | 45 | 5 | invalid |
| `full_fulltext` | 44 | 42 | 2,732 | 44 | 2 | invalid |
| `no_warmup` | 45 | 43 | 2,824 | 45 | 2 | invalid |
| `fulltext_beats3` | 45 | 39 | 2,829 | 15 | 3 | invalid |
| `fulltext_beats5` | 45 | 41 | 2,847 | 15 | 1 | invalid |
| `fulltext_beats10` | 45 | 39 | 2,782 | 15 | 4 | invalid |
| `idea_bare` | 16 | 15 | 2,840 | 16 | 1 | invalid |
| `idea_full` | 113 | 103 | 2,769 | 17 | 2 | invalid |
| `idea_no_priming` | 114 | 106 | 2,802 | 18 | 2 | invalid |
| `idea_no_tropes` | 112 | 97 | 2,781 | 16 | 1 | invalid |
| `idea_tax_only` | 110 | 101 | 2,793 | 18 | 2 | invalid |
| `idea_tropes_only` | 18 | 18 | 2,740 | 18 | 0 | invalid |
| `idea_no_taxonomy_bugged` | 15 | 14 | 2,823 | 15 | 1 | invalid (duplicate of `idea_tropes_only`) |
| `idea_vanilla` | 16 | 14 | 2,604 | 16 | 2 | invalid |
| `idea_instruction_vanilla` | 15 | 10 | 2,672 | 15 | 5 | invalid |
| `idea_no_taxonomy` | 0 | – | – | – | – | never ran |

No configuration has more than 5 judged complete stories. The data therefore cannot support a
complete-stories-only re-analysis.

## The paper's own generation runs

These runs are separate from the ablation. Full numbers are in [`../../ERRATA.md`](../../ERRATA.md) A1.

| Run | Used in | Verdict |
|---|---|---|
| Vanilla Sonnet 4.6, Opus 4.6, GPT-5.2 | Table 2, Table 7, human study | **Valid.** No story is cut off. |
| Sonnet 4.6 + pipeline | Table 2, Fig. 6, Table 7, human study | **Valid, with flags.** 8 of 32 stories are cut off, but the gain holds on the complete stories. Length differs from vanilla (2,343 vs 992 words). |
| Sonnet 4.6, no Step 0 | App. J | **Valid, with flags.** 11 of 32 stories are cut off; the gain holds on the complete stories. |
| Opus 4.6 + pipeline | Table 2, human study (`i1_p18`, cut off) | **Not established.** 18 of 32 stories are cut off and 5 more are at the ceiling. On the 9 clean prompts, no difference from vanilla is detectable. Re-generate. |
| GPT-5.2 + pipeline | Table 2 | **Not established.** 17 of 32 stories are cut off and 6 more are at the ceiling. Same as Opus. Re-generate. |
| Stability repeats 2–4 | Table 7 | **Valid.** They re-measure fixed texts; truncation does not affect a test of the metric's noise. |
| Kimi-K2 + pipeline | human study (`i1_p17` only) | **Valid for that use.** `i1_p17` is complete (946 words). |
| GPT-5 + pipeline / GPT-5 vanilla | not used | **Invalid, not released.** 28 of 32 pipeline stories are cut off, the vanilla run has no judge results on file, and 4 of its 96 responses are empty. |

## Re-running the ablation

To re-run it, use the fixes listed in `pipeline/README.md`:
- raise `max_tokens`;
- record the stop reason and regenerate when the cap is hit;
- give Step 2 the prompt and word target;
- drop the 2,000–3,000-word filter;
- run `idea_no_taxonomy`.

## Contents

- `stories/<config>/`: story texts and per-run configs.
- `sbatch/`, `rescore_2026-03-28/`: the SLURM files and the re-judging scripts.
- `retry_unqualified.py`, `run_instruction_vanilla.sh`: generation helpers.
- `notebooks/`: the March analysis notebooks. They read judge files on the authors' cluster and are kept for provenance only.
- `ANALYSIS.md`: the March write-up. Withdrawn; its figure files are in the git tag `archive-2026-03-20`.
