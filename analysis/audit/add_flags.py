#!/usr/bin/env python3
"""Add truncation flags to results/scores/<group>.csv for the model-generated groups.

ends_mid_sentence  story text stops mid-sentence (rule + manual overrides; per_story.csv / other_groups.csv)
at_length_ceiling  story ends cleanly but is at or above its model's truncation length (model_ceilings.csv);
                   such a story may have been cut exactly at a sentence boundary
Human corpora get empty values. Run after audit.py and check_other_groups.py."""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(HERE, "..", "..", "results", "scores")
RUN = {
    "sonnet46_vanilla": "run1__claude-sonnet-4-6",
    "opus46_vanilla": "run1__claude-opus-4-6",
    "gpt52_vanilla": "run1__openai_gpt-5_2",
    "sonnet46_pipeline": "narrative_sonnet46__anthropic_claude-sonnet-4-6",
    "opus46_pipeline": "narrative_opus46__anthropic_claude-opus-4-6",
    "gpt52_pipeline": "narrative_gpt52__openai_gpt-5_2",
    "sonnet46_pipeline_nostep0": "narrative_sonnet46_nostep0__anthropic_claude-sonnet-4-6",
}
ps = pd.read_csv(os.path.join(HERE, "per_story.csv"))
og = pd.read_csv(os.path.join(HERE, "other_groups.csv"))
for f in sorted(os.listdir(SCORES)):
    g = f[:-4]
    d = pd.read_csv(os.path.join(SCORES, f), dtype={"story_id": str})
    d = d.drop(columns=[c for c in ("ends_mid_sentence", "at_length_ceiling") if c in d])
    if g in RUN:
        a = ps[ps.run == RUN[g]].set_index("story_id")
        d["ends_mid_sentence"] = d.story_id.map(a.ends_mid_sentence)
        d["at_length_ceiling"] = d.story_id.map(a.at_length_ceiling)
    elif g in ("top10_zeroshot", "kimik2_pipeline"):
        a = og[og.group == g].set_index(["model_slug", "story_id"]).ends_mid_sentence
        d["ends_mid_sentence"] = [a.get((m, s)) for m, s in zip(d.model_slug, d.story_id)]
        d["at_length_ceiling"] = pd.NA
    else:
        d["ends_mid_sentence"] = pd.NA
        d["at_length_ceiling"] = pd.NA
    if g in RUN or g in ("top10_zeroshot", "kimik2_pipeline"):
        assert d.ends_mid_sentence.notna().all(), (g, d[d.ends_mid_sentence.isna()].story_id.tolist())
    d.to_csv(os.path.join(SCORES, f), index=False)
    if g not in ("new_yorker", "storystar", "tell_me_a_story"):
        print(f"{g:28s} n={len(d):3d} mid={int(d.ends_mid_sentence.astype(bool).sum()):3d} "
              f"ceiling_complete={int((d.at_length_ceiling.fillna(False).astype(bool) & ~d.ends_mid_sentence.astype(bool)).sum())}")
