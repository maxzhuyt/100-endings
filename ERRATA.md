# Errata and known issues (arXiv:2604.09854v2)

We list every place where the published paper (v2, 3 Oct 2026) disagrees with the code or data
in this repository. The values come from `analysis/reproduce_tables.py` (`results/tables/TABLES.md`) and
from the truncation audit (`analysis/audit/`).

## A. Issues that affect conclusions

### A1. Many pipeline stories hit the 4,000-token cap and end mid-sentence
Every generation call uses `max_tokens=4000`, and no `stop_reason` / `finish_reason` was recorded
(`pipeline/core/narrative_pipeline.py`). A story that hits the cap ends mid-sentence. The 100-Endings judge
then treats that cut-off text as the story's ending.

| Stories (iteration 1) | n | end mid-sentence | complete, but at the model's length ceiling | median words |
|---|---|---|---|---|
| Sonnet 4.6 + pipeline (Table 2) | 32 | 8 | 2 | 2,343 |
| Opus 4.6 + pipeline (Table 2) | 32 | 18 | 5 | 2,648 |
| GPT-5.2 + pipeline (Table 2) | 32 | 17 | 6 | 2,895 |
| Sonnet 4.6, no Step 0 (App. J) | 32 | 11 | 7 | 2,810 |
| Sonnet 4.6 / Opus 4.6 / GPT-5.2 vanilla (Table 2) | 32 each | 0 / 0 / 0 | 0 / 0 / 1 | 992 / 998 / 2,134 |
| Top-10 zero-shot (Tables 1, 6) | 320 | 10 (9 horizon-alpha, 1 Sonnet 4.5) | not computed | 1,102 |
| Kimi-K2 + pipeline (human study only) | 32 | 0 | not computed | 557 |

These counts are lower bounds. A story cut off exactly at a sentence boundary cannot be detected without the
stop reason; the "length ceiling" column counts complete stories long enough for that to have happened.
- **Where the flags are.** Per-story flags are the `ends_mid_sentence` and `at_length_ceiling` columns of
  `results/scores/*.csv`.
- **How they were decided.** The detection rule, the 132 manual overrides and the per-model ceilings are in
  `analysis/audit/` (`audit.py`, `check_other_groups.py`).
- **Human study.** One of the six stories in the human study (Table 8), Opus 4.6 + pipeline `i1_p18`, is cut off.
  It ends: "You end up telling people 'I".

**Effect on Table 2.** The paired contrasts were recomputed on prompts whose pipeline story is complete
(`analysis/audit/sensitivity.md`; 95% bootstrap CIs over prompts).

| Pipeline − vanilla | prompts kept | Δ Mean NR | Δ Late NR |
|---|---|---|---|
| Sonnet 4.6, all prompts | 32 | +0.143 [+0.090, +0.199] | +0.203 [+0.138, +0.271] |
| Sonnet 4.6, complete and below ceiling | 22 | +0.149 [+0.079, +0.221] | +0.134 [+0.071, +0.200] |
| No Step 0, complete and below ceiling | 14 | +0.106 [+0.026, +0.178] | +0.166 [+0.049, +0.275] |
| Opus 4.6, all prompts | 32 | +0.086 [+0.027, +0.146] | +0.111 [+0.039, +0.185] |
| Opus 4.6, complete and below ceiling | 9 | +0.063 [−0.018, +0.152] | −0.034 [−0.145, +0.072] |
| GPT-5.2, all prompts | 32 | +0.045 [−0.004, +0.100] | +0.114 [+0.032, +0.188] |
| GPT-5.2, complete and below ceiling | 9 | −0.002 [−0.057, +0.054] | −0.034 [−0.230, +0.127] |

- **Sonnet 4.6 (with and without Step 0).** The gain holds in every subset.
- **Opus 4.6 and GPT-5.2.** The printed gains rest on truncated or near-cap stories. On the 9 prompts per model
  with a clean pipeline story, no no-rate difference is distinguishable from zero.
  - Those 9 stories are a selected subset: the Opus median is 1,141 words, against 2,648 for all 32.
  - This shows that the data cannot establish a pipeline effect for these two models. It does not show that
    the effect is absent.
- **EQ-Bench.** Truncation also lowers the pipeline's rubric scores. Opus 4.6's Δ EQ-Bench moves from −1.07 (all
  32 prompts) to +3.59 [+0.78, +6.04] (n = 9).
- **Table 1 / 6.** Not affected. Without its 10 cut-off stories, the Top-10 group's mean and late NR are
  0.633 and 0.212, against 0.630 and 0.215 with them.

### A2. Step 2 does not see the prompt, so pipeline stories are 2–3× longer
Step 2's prompt contains only the Step 1 beat sheet ("Here is a structural to-do list for a short story …").
The EQ-Bench prompt and its 1,000-word target reach Step 2 only where Step 1 copied a word count into the beats.
That happened in 1 of 32 Sonnet 4.6 beat sheets, 13 of 32 Opus 4.6, 23 of 32 GPT-5.2 and 5 of 32 no-Step-0 sheets.
Even GPT-5.2's pipeline stories have a median of 2,895 words.

Appendix E.4.3 prints this prompt correctly. Section 4 instead says Step 2 is "conditioned on the story idea and
plan from Step 1 … with a system-prompt emphasis on sustaining tension"; the code sends no system prompt at any
step.

The vanilla-versus-pipeline contrast in Table 2 is therefore also a contrast in length: median 2,343 vs 992 words
for Sonnet 4.6, 2,648 vs 998 for Opus 4.6, and 2,895 vs 2,134 for GPT-5.2. Appendix G checks length only across
corpora, not within Table 2.

**Length also changes what the judge sees.**
- **The cap.** The judge receives at most the first 2,000 words of the continuation (`truncate_words` in
  `metric/compute_judge.py` and `metric/compute_cwb_judge.py`). At early positions of longer stories, the true
  ending is therefore not in the judge's prompt.
- **How often.** This affects:
  - 37.0% of judged New Yorker positions and 25.4% of StoryStar positions;
  - 9–17% of pipeline positions;
  - at most 2.4% of vanilla positions.
- **Recomputed without them.** No-rate was recomputed on the positions where every story's ending is visible:
  pct ≥ 75 for the corpora and pct ≥ 45 for Table 2 (`analysis/audit/sensitivity.md`).
  - Corpora: New Yorker 0.648 > StoryStar 0.426 > Tell Me a Story 0.369 > Top-10 0.271. This is the same order
    as Late NR.
  - Table 2: the pipeline gains are +0.214 (Sonnet), +0.143 (Opus) and +0.092 (GPT-5.2), and every CI excludes zero.
- **Late NR.** Late NR (pct ≥ 80) is unaffected in every group.

### A3. Table 19's "paired" intervals are unpaired; paired intervals change the EQ-Bench conclusion
Table 19 and the † marks in Table 2 are described as a paired bootstrap over the 32 shared prompts. The
printed intervals are reproduced only by resampling the vanilla and the pipeline stories independently
(`paper_revision/scripts/01_table2_cis.py`, reproduced in `analysis/reproduce_tables.py`). A true paired
bootstrap, or a paired t-test, over the 32 prompts gives:

| Model | EQ-Bench Δ | paired 95% bootstrap CI | paired t-test p |
|---|---|---|---|
| Sonnet 4.6 | +2.99 | [+0.17, +5.22] | 0.030 |
| Opus 4.6 | −1.07 | [−3.91, +1.60] | 0.46 |
| GPT-5.2 | −2.56 | [−5.05, −0.22] | 0.051 |

Under pairing, Sonnet's EQ-Bench gain is significant. GPT-5.2's EQ-Bench drop is significant by the paired
bootstrap and borderline by the t-test. The statement "No EQ-Bench difference is significant in either
direction" (Section 5, Appendix F) therefore holds only for the unpaired test. The no-rate conclusions do not
change: every no-rate gain marked significant stays significant, and GPT-5.2's mean no-rate stays n.s. (full
paired table in `results/tables/table19_paired_bootstrap.csv`).

### A4. Human study: annotators used the strict criterion, the deployed judge uses the lenient one
Appendix D.1 says the Task 2 criterion was "deliberately worded to mirror the deployed judge's standard", and
quotes it: "Y if the core event … actually occurs … Be strict. Vague thematic similarity is not enough." That
wording is the criterion of the earlier, strict judge (git tag `archive-2026-03-20`). The judge that produced
every number in the paper is the lenient "general direction" judge printed in Appendix C.5. The judge's excess of
false positives over false negatives against human labels (88 vs 46, Table 14) is the direction this mismatch
predicts. Section 3.1 also still describes the strict, single-pass judge ("a strict standard … returns a binary
verdict for all 100 endings in a single pass"). Appendix C.5 describes the lenient judge with independent
per-ending prompts, which is the one actually used.

### A5. The pipeline story printed in Appendix K.3 is not a pipeline output
Appendix K.3 prints a pipeline story whose protagonist is "Elara". No run produced it: it is absent from every
story file and from the EQ-Bench runs JSON. The Sonnet 4.6 + pipeline story for prompt 3, the one whose curve
is shown in Figure 6 (right), is `data/stories/sonnet46_pipeline/i1_p03.txt` (protagonist Mara, 1,899 words).
The two texts share only the final sentence. Appendix K.1 prints a paraphrase with invented bullet points, not
the EQ-Bench prompt. The actual prompt is entry `"3"` of `pipeline/data/creative_writing_prompts_v3.json`
("Write one scene within a larger story set in Wales …"). The vanilla story in K.2 is genuine
(`data/stories/sonnet46_vanilla/i1_p03.txt`).

## B. Numbers that do not reproduce

| Where | Printed | Reproduced | Cause |
|---|---|---|---|
| Table 4, New Yorker Tension8 / ±CI | 78.40 / 4.33 | 78.49 / 2.85 | Printed values come from an older 30-story rubric run (overall 78.70). The printed Overall (78.61) and n = 53 come from the current run. |
| Table 4, StoryStar row | 45.81 / 44.17 / 4.23 / n = 31 | 44.25 / 43.56 / 2.25 / n = 100 | The only StoryStar rubric run on file has 100 stories, all in the 2,000–3,000-word band. No 31-story run was found. |
| Table 6, StoryStar IR120 | 0.232 | 0.233 | Minor. |
| Table 30, Δ mean NR (full − vanilla) | +14.4 pp | +14.35 pp | Rounding. |
| Table 20, StoryStar r(NR, sentences) | −0.009 | −0.0095 | Rounding. |
| Table 30, p-values | e.g. 4.7×10⁻⁷ | t = 5.04 with 31 df gives p = 1.9×10⁻⁵ | The printed p is the two-sided **normal** tail of the paired t statistic, not the t distribution with 31 df. The four significant p-values are 1–4 orders of magnitude too small. No significance decision changes. |
| Tables 2 / 18 / 19, bootstrap bounds | — | within Monte-Carlo error | The original random stream cannot be reproduced exactly. Bounds agree within 0.05 × half-width. |
| Table 18, Top-10 n | 319 | 320 | Table 2 prints 320. |
| Table 25 | 9 rows, "242 stories, 10 genres" | 10 genres with n ≥ 5 hold 202 stories | The row Disease/Disability (n = 6, 0.766) is missing from the printed table. 242 is the number of labelled stories. ANOVA (F = 0.61, p = 0.79, η² = 2.8%) reproduces. |
| Table 28 | "Per story (median)" | means | The per-story and per-position values are means. |

## C. Data that is not in this repository

- **Human-evaluation materials and labels (Section 3.3, Appendix D, Tables 9–16).** These were not on
  the authors' cluster when this release was assembled and will be added. Table 8's six stories are identified
  in `results/tables/TABLES.md`.
- **New Yorker and StoryStar texts and their generated endings.** New Yorker texts are under copyright (see the
  Ethics Statement) and StoryStar texts were scraped, so neither is redistributed. Per-story scores and full
  no-rate curves are included. The Tell Me a Story sample (CC-BY 4.0) is included in full.
- **Top-10 story texts.** These are the iteration-1 outputs in EQ-Bench's public `creative_bench_runs.zip`.
  Our endings and verdicts for them are included.
