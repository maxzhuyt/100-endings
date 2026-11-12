# Truncation audit and sensitivity analysis (ERRATA A1, A2)

| File | What it is |
|---|---|
| `audit.py` | Checks every generation run, both the paper's and the March ablation's, for stories that end mid-sentence, are at their model's length ceiling, or are not stories. It also checks judge coverage. Reads the authors' cluster; writes the CSVs below and `SUMMARY.md`. |
| `overrides.csv` | The 132 hand decisions that override the rule, each with its reason. |
| `per_story.csv`, `per_run.csv`, `model_ceilings.csv`, `non_story.csv`, `SUMMARY.md` | `audit.py` outputs. |
| `check_other_groups.py`, `other_groups.csv` | The same rule applied to the Top-10 and Kimi-K2 stories, with hand overrides inside the script. |
| `add_flags.py` | Writes `ends_mid_sentence` and `at_length_ceiling` into `results/scores/*.csv`. |
| `sensitivity.py`, `sensitivity.md`, `table2_sensitivity.csv`, `positions_hidden_ending.csv.gz` | Two analyses: Table 2 recomputed without truncated stories, and Tables 1–2 recomputed on positions where the judge sees the true ending. Runs from the released files. |

## Rule

A story **ends mid-sentence** if the following holds after trailing decoration (`---`, `***`, markdown) is stripped:

- its last character is not one of `. ! ? … " ” ’ ' ) ] — : ;`, and
- its last line has more than five words.

Every rule-positive case was then read by hand, and so were all short last lines, dash/colon endings and a random
sample of 97 rule-complete endings. The overrides record each decision.

## Length ceiling

A complete story is **at the length ceiling** when it is at least as long as the 10th percentile of its model's
mid-sentence stories. This applies only to a model whose mid-sentence stories pile up at the top of its length
range. Such a story may have been cut exactly at a sentence boundary.
