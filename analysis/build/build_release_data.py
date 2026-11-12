#!/usr/bin/env python3
"""Assemble the released per-story data (results/, data/stories/) from the authors'
cluster files. Kept for provenance: it documents exactly which raw file each released
number comes from. It only runs on the authors' cluster (absolute paths below).

Sources
  Human corpora no-rate     narrative_project/results_endings/ending_judge_sh{1..6}.json
                            (consolidated in paper_revision/data/corpora/<Group>/)
  Model no-rate (Table 2)   creative-writing-bench/results/judge/ending_judge{,_opus46,_gpt52}.json
  Stability repeats 2-4     creative-writing-bench/results/judge_rep{2,3,4}/cwb_*_sonnet46_judge.json
  No-Step-0 run (App. J)    narrative_ablation/results/judge_ablation/cwb_narrative_sonnet46_nostep0_judge.json
  Kimi-K2 pipeline (Tab. 8) creative-writing-bench/results/judge/ending_judge_kimi.json
  EQ-Bench rubric           creative-writing-bench/creative_bench_runs{,_kimi}.json (models, iteration 1)
                            creative-writing-bench/corpus_eval/results/{eval_results,old_results}.json (humans)
"""
import ast, csv, glob, gzip, json, os, shutil, statistics, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "metric"))
import curve_metrics as M  # noqa: E402

NP = "/project/jevans/maxzhuyt/narrative_project"
CWB = "/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
PR = NP + "/paper_revision/data/corpora"
RUNS = CWB + "/creative_bench_runs.json"
RUNS_KIMI = CWB + "/creative_bench_runs_kimi.json"
NEG = set(l.strip() for l in open(CWB + "/data/negative_criteria.txt") if l.strip())
TENSION8 = ["Emotionally Engaging", "Emotionally Complex", "Overall Reader Engagement",
            "Well-earned Lightness or Darkness", "Unsurprising or Uncreative", "Meandering",
            "Incongruent Ending Positivity", "Unearned Transformations"]

# paper_revision group -> released group name
PR_GROUPS = {
    "NewYorker": "new_yorker", "TellMeAStory": "tell_me_a_story", "StoryStar": "storystar",
    "Top10": "top10_zeroshot",
    "Sonnet4.6-vanilla": "sonnet46_vanilla", "Sonnet4.6-pipeline": "sonnet46_pipeline",
    "Opus4.6-vanilla": "opus46_vanilla", "Opus4.6-pipeline": "opus46_pipeline",
    "GPT5.2-vanilla": "gpt52_vanilla", "GPT5.2-pipeline": "gpt52_pipeline",
}
# model groups: released name -> (model_slug used in judge/stories dirs, EQ-Bench run key)
MODEL_RUNS = {
    "sonnet46_vanilla": ("claude-sonnet-4-6", "run1__claude-sonnet-4-6"),
    "sonnet46_pipeline": ("narrative_sonnet46__anthropic_claude-sonnet-4-6",
                          "narrative_sonnet46__anthropic_claude-sonnet-4-6"),
    "opus46_vanilla": ("claude-opus-4-6", "run1__claude-opus-4-6"),
    "opus46_pipeline": ("narrative_opus46__anthropic_claude-opus-4-6",
                        "narrative_opus46__anthropic_claude-opus-4-6"),
    "gpt52_vanilla": ("openai__gpt-5.2", "run1__openai_gpt-5_2"),
    "gpt52_pipeline": ("narrative_gpt52__openai_gpt-5_2", "narrative_gpt52__openai_gpt-5_2"),
    "sonnet46_pipeline_nostep0": ("narrative_sonnet46_nostep0__anthropic_claude-sonnet-4-6",
                                  "narrative_sonnet46_nostep0__anthropic_claude-sonnet-4-6"),
    "kimik2_pipeline": ("narrative_kimi__moonshotai_kimi-k2", "narrative_kimi__moonshotai_kimi-k2"),
}
EXTRA_JUDGE = {
    "sonnet46_pipeline_nostep0": NP + "/narrative_ablation/results/judge_ablation/cwb_narrative_sonnet46_nostep0_judge.json",
    "kimik2_pipeline": CWB + "/results/judge/ending_judge_kimi.json",
}
EXTRA_ENDINGS = {
    "sonnet46_pipeline_nostep0": NP + "/narrative_ablation/results/endings_ablation/cwb_narrative_sonnet46_nostep0",
    "kimik2_pipeline": CWB + "/results/endings/narrative_kimi__moonshotai_kimi-k2",
}

OUT_SCORES = os.path.join(REPO, "results", "scores")
OUT_CURVES = os.path.join(REPO, "results", "curves")
OUT_END = os.path.join(REPO, "results", "endings")
OUT_STAB = os.path.join(REPO, "results", "stability")
OUT_STORIES = os.path.join(REPO, "data", "stories")
for d in (OUT_SCORES, OUT_CURVES, OUT_END, OUT_STAB, OUT_STORIES):
    os.makedirs(d, exist_ok=True)


def curve_from_judge(entry):
    pos = sorted(entry["positions"], key=lambda p: p["pct"])
    return [{"position": p["position"], "pct": round(p["pct"], 2),
             "no_rate": round(1 - p["yes_rate"], 4)} for p in pos if p["pct"] >= 10]


def curve_stats(curve):
    pct = np.array([c["pct"] for c in curve], float)
    nr = np.array([c["no_rate"] for c in curve], float)
    late = nr[pct >= 80]
    ret = M.retention(nr)
    return {
        "n_positions": len(nr),
        "mean_nr": round(float(nr.mean()), 6),
        "late_nr": round(float(late.mean()), 6) if len(late) else "",
        "retain_pct": "" if np.isnan(ret) else round(ret * 100, 6),
        "ir30": round(M.inflection_rate(nr, pct, 30), 6),
        "ir60": round(M.inflection_rate(nr, pct, 60), 6),
        "ir120": round(M.inflection_rate(nr, pct, 120), 6),
    }


def eq_scores(js):
    def sc(crits):
        v = [(js[c] if c not in NEG else 20 - js[c]) * 5 for c in crits if c in js]
        return v
    allv = sc(list(js))
    t8 = sc(TENSION8)
    return (sum(allv) / len(allv) if allv else None, sum(t8) / len(t8) if len(t8) == 8 else None)


def eq_for_run(runs, run_key, prompt_num):
    """Mean over modifiers of the 22-criterion composite and Tension8 (iteration 1)."""
    task = runs[run_key]["creative_tasks"]["1"].get(prompt_num)
    if task is None:
        return None, None
    o, t = [], []
    for m in task["results_by_modifier"].values():
        js = m.get("judge_scores")
        if not js:
            continue
        a, b = eq_scores(js)
        if a is not None:
            o.append(a)
        if b is not None:
            t.append(b)
    return (round(statistics.mean(o), 4) if o else "", round(statistics.mean(t), 4) if t else "")


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_jsonl_gz(path, recs):
    with gzip.open(path, "wt") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


print("loading EQ-Bench runs ...")
runs = json.load(open(RUNS))
runs.update(json.load(open(RUNS_KIMI)))

# ---------------- New Yorker titles/authors (metadata only; texts are not released) -------------
def nyer_meta(sid):
    for d in (NP + "/NEWCORPUS_CLEANED",):
        p = f"{d}/{sid}.txt"
        if os.path.exists(p):
            lines = [l.strip().lstrip("﻿") for l in open(p, encoding="utf-8") if l.strip()][:2]
            title = lines[0] if lines else ""
            author = lines[1][3:] if len(lines) > 1 and lines[1].startswith("By ") else ""
            return title, author
    return "", ""

ss_meta = {r["filename"][:-4]: r for r in csv.DictReader(open(NP + "/storystar_meta.csv"))}
# New Yorker genre labels (GLM-5 classifier, Williams 2017 Screenwriters Taxonomy)
GENRE_CSV = "/project/jevans/maxzhuyt/narrative_fiction/llm-fiction-genre/classification_results_20260316_191910.csv"
ny_genre = {os.path.splitext(r["filename"])[0]: (r.get("super_genre", "").strip(), r.get("macro_genre", "").strip())
            for r in csv.DictReader(open(GENRE_CSV))}

# ---------------- human EQ-Bench (Table 4): per-story overall + Tension8 -------------------------
ev = json.load(open(CWB + "/corpus_eval/results/eval_results.json"))["story_results"]
old = json.load(open(CWB + "/corpus_eval/results/old_results.json"))["story_results"]
eq_rows = []
for src, recs in (("eval_results.json", ev), ("old_results.json", old)):
    for s in recs:
        corpus = {"nyer": "new_yorker", "tell_me_a_story": "tell_me_a_story", "storystar": "storystar"}.get(s["corpus"])
        if corpus is None or s.get("status") != "completed":
            continue
        if src == "eval_results.json" and corpus == "storystar":
            continue
        if src == "old_results.json" and corpus != "storystar":
            continue
        raw = s["judge_scores_raw"]
        js = raw if isinstance(raw, dict) else ast.literal_eval(raw)
        overall, t8 = eq_scores(js)
        eq_rows.append({"corpus": corpus, "story_id": s["story_id"], "word_count": s.get("word_count"),
                        "eqbench": round(float(s["final_score_0_100"]), 4),
                        "eqbench_recomputed": round(overall, 4), "tension8": round(t8, 4) if t8 else "",
                        "source_file": src})
os.makedirs(os.path.join(REPO, "results", "eqbench"), exist_ok=True)
write_csv(os.path.join(REPO, "results", "eqbench", "human_corpora_eqbench.csv"), eq_rows,
          ["corpus", "story_id", "word_count", "eqbench", "eqbench_recomputed", "tension8", "source_file"])
print("human EQ rows:", len(eq_rows))

# ---------------- per-story scores + curves for every group ------------------------------------
SCORE_COLS = ["group", "model_slug", "story_id", "title", "author", "super_genre", "macro_genre", "words", "n_sentences",
              "n_positions", "mean_nr", "late_nr", "retain_pct", "ir30", "ir60", "ir120",
              "eqbench", "tension8"]
summary = {}
for prg, grp in PR_GROUPS.items():
    curves = {}
    for l in open(f"{PR}/{prg}/curves.jsonl"):
        c = json.loads(l)
        curves[(c.get("model_slug", ""), c["story_id"])] = c
    rows, outc = [], []
    for r in csv.DictReader(open(f"{PR}/{prg}/scores.csv")):
        slug = r.get("model_slug", "") or ""
        c = curves[(slug, r["story_id"])]
        cur = [{"position": None, "pct": p["pct"], "no_rate": p["no_rate"]} for p in sorted(c["positions"], key=lambda p: p["pct"]) if p["pct"] >= 10]
        st = curve_stats(cur)
        assert abs(st["mean_nr"] - float(r["no_rate_mean"])) < 1e-3, (grp, r["story_id"])
        row = {"group": grp, "model_slug": slug, "story_id": r["story_id"],
               "words": r["word_count"], "n_sentences": r["n_sentences"], **st,
               "eqbench": r["eqbench_0_100"], "tension8": ""}
        if grp == "new_yorker":
            row["title"], row["author"] = nyer_meta(r["story_id"])
            row["super_genre"], row["macro_genre"] = ny_genre.get(r["story_id"], ("", ""))
        if grp == "storystar":
            m = ss_meta.get(r["story_id"], {})
            row["title"], row["author"] = m.get("title", ""), m.get("author", "")
        if grp == "top10_zeroshot" or grp in MODEL_RUNS:
            run_key = MODEL_RUNS[grp][1] if grp in MODEL_RUNS else "run1__" + {
                "anthropic__claude-sonnet-4.5": "anthropic_claude-sonnet-4_5", "openai__gpt-5.2": "openai_gpt-5_2",
                "moonshotai__kimi-k2": "moonshotai_kimi-k2", "openrouter__horizon-alpha": "openrouter_horizon-alpha",
                "openrouter__pony-alpha": "openrouter_pony-alpha"}.get(slug, slug)
            o, t8 = eq_for_run(runs, run_key, str(int(r["story_id"].split("_p")[1])))
            assert r["eqbench_0_100"] == "" or abs(float(o) - float(r["eqbench_0_100"])) < 1e-3, (grp, slug, r["story_id"], o, r["eqbench_0_100"])
            row["tension8"] = t8
        rows.append(row)
        outc.append({"group": grp, "model_slug": slug, "story_id": r["story_id"],
                     "positions": [{"pct": p["pct"], "no_rate": p["no_rate"]} for p in cur]})
    write_csv(f"{OUT_SCORES}/{grp}.csv", rows, SCORE_COLS)
    write_jsonl_gz(f"{OUT_CURVES}/{grp}.jsonl.gz", outc)
    summary[grp] = (len(rows), round(np.mean([x["mean_nr"] for x in rows]), 4))

# extra model groups (no-Step-0, Kimi-K2 pipeline) straight from their judge files
for grp in ("sonnet46_pipeline_nostep0", "kimik2_pipeline"):
    slug, run_key = MODEL_RUNS[grp]
    d = json.load(open(EXTRA_JUDGE[grp]))
    rows, outc = [], []
    for k, e in sorted(d.items()):
        sid = k.split("/")[-1]
        if not sid.startswith("i1_"):
            continue
        cur = curve_from_judge(e)
        st = curve_stats(cur)
        txt = open(f"{CWB}/stories/{slug}/{sid}.txt", encoding="utf-8").read()
        endf = f"{EXTRA_ENDINGS[grp]}/{sid}_endings.json"
        nsent = json.load(open(endf)).get("n_sentences", "") if os.path.exists(endf) else ""
        o, t8 = eq_for_run(runs, run_key, str(int(sid.split("_p")[1])))
        rows.append({"group": grp, "model_slug": slug, "story_id": sid, "words": len(txt.split()),
                     "n_sentences": nsent, **st, "eqbench": o, "tension8": t8})
        outc.append({"group": grp, "model_slug": slug, "story_id": sid,
                     "positions": [{"pct": p["pct"], "no_rate": p["no_rate"]} for p in cur]})
    write_csv(f"{OUT_SCORES}/{grp}.csv", rows, SCORE_COLS)
    write_jsonl_gz(f"{OUT_CURVES}/{grp}.jsonl.gz", outc)
    summary[grp] = (len(rows), round(np.mean([x["mean_nr"] for x in rows]), 4))

# ---------------- stability repeats (Table 7): rep1 = main run, reps 2-4 re-generated + re-judged --
for rep in (2, 3, 4):
    for cond, grp in (("vanilla", "sonnet46_vanilla"), ("narrative", "sonnet46_pipeline")):
        d = json.load(open(f"{CWB}/results/judge_rep{rep}/cwb_{cond}_sonnet46_judge.json"))
        outc = [{"group": grp, "rep": rep, "story_id": "i1_" + k.split("_")[0],
                 "positions": [{"pct": p["pct"], "no_rate": p["no_rate"]} for p in curve_from_judge(e)]}
                for k, e in sorted(d.items())]
        write_jsonl_gz(f"{OUT_STAB}/rep{rep}_{grp}.jsonl.gz", outc)

# ---------------- endings + verdicts (model-generated groups only) ------------------------------
for prg, grp in PR_GROUPS.items():
    src = f"{PR}/{prg}/endings_with_verdicts.jsonl.gz"
    if grp in ("new_yorker", "storystar"):
        continue  # texts not redistributable -> their generated endings are withheld too (see README)
    if grp == "top10_zeroshot":
        os.makedirs(f"{OUT_END}/top10_zeroshot", exist_ok=True)
        by = {}
        with gzip.open(src, "rt") as f:
            for l in f:
                by.setdefault(json.loads(l)["model_slug"], []).append(l)
        for slug, ls in by.items():
            with gzip.open(f"{OUT_END}/top10_zeroshot/{slug}.jsonl.gz", "wt") as f:
                f.writelines(ls)
    else:
        shutil.copyfile(src, f"{OUT_END}/{grp}.jsonl.gz")

for grp in ("sonnet46_pipeline_nostep0", "kimik2_pipeline"):
    d = json.load(open(EXTRA_JUDGE[grp]))
    recs = []
    for k, e in sorted(d.items()):
        sid = k.split("/")[-1]
        if not sid.startswith("i1_"):
            continue
        endd = json.load(open(f"{EXTRA_ENDINGS[grp]}/{sid}_endings.json"))
        bypos = {p["position"]: p for p in endd["positions"]}
        poss = []
        for p in sorted(e["positions"], key=lambda p: p["pct"]):
            ends = bypos[p["position"]]["endings"]
            poss.append({"pct": p["pct"], "judged": p["pct"] >= 10, "no_rate": round(1 - p["yes_rate"], 4),
                         "endings": [{"text": t, "match": j} for t, j in zip(ends, p["judgments"])]})
        recs.append({"story_id": sid, "n_sentences": endd.get("n_sentences"), "positions": poss})
    write_jsonl_gz(f"{OUT_END}/{grp}.jsonl.gz", recs)

# ---------------- story texts + pipeline intermediates (model groups we generated) ---------------
for grp, (slug, run_key) in MODEL_RUNS.items():
    od = f"{OUT_STORIES}/{grp}"
    os.makedirs(od, exist_ok=True)
    steps = []
    for f in sorted(glob.glob(f"{CWB}/stories/{slug}/i1_p*.txt")):
        shutil.copyfile(f, f"{od}/{os.path.basename(f)}")
    tasks = runs[run_key]["creative_tasks"]["1"]
    for pnum in sorted(tasks, key=int):
        for mod, m in tasks[pnum]["results_by_modifier"].items():
            rec = {"story_id": f"i1_p{int(pnum):02d}", "eqbench_prompt_id": pnum}
            for k in ("narrative_source_id", "narrative_warmup_source_id", "narrative_step0", "narrative_warmup",
                      "narrative_step1", "narrative_beats"):
                if k in m:
                    rec[k] = m[k]
            if len(rec) > 2:
                steps.append(rec)
    if steps:
        write_jsonl_gz(f"{od}/pipeline_steps.jsonl.gz", steps)

# ---------------- Tell Me a Story sample texts (CC-BY 4.0, Huot et al. 2025) -------------------
tm = {f"tmas_{json.loads(l)['story_id']:04d}": json.loads(l)["text"]
      for l in open(NP + "/BASELINE_CORPUS/tell_me_a_story/all_stories.jsonl")}
od = f"{OUT_STORIES}/tell_me_a_story"
os.makedirs(od, exist_ok=True)
for sid in [r["story_id"] for r in csv.DictReader(open(f"{OUT_SCORES}/tell_me_a_story.csv"))]:
    open(f"{od}/{sid}.txt", "w", encoding="utf-8").write(tm[sid])
open(f"{od}/LICENSE.md", "w").write(
    "Texts from the Tell Me a Story dataset (Huot et al., 2025, \"Agents' Room\", ICLR 2025;\n"
    "https://github.com/google-deepmind/tell_me_a_story), licensed CC-BY 4.0\n"
    "(https://creativecommons.org/licenses/by/4.0/). These are the 100 stories of our evaluation sample,\n"
    "unmodified; NNNN in `tmas_NNNN.txt` is the story's index in our 230-story copy of the corpus.\n")

print(json.dumps(summary, indent=1))
