#!/usr/bin/env python3
"""
audit.py -- truncation / judge-coverage audit of story-generation runs
(Spoiler Alert / 100-Endings).

Read-only on all project files.  Writes only into OUT_DIR (this script's dir).

Inputs
  A. every dir under narrative_ablation/results/stories/
  B. paper runs in creative-writing-bench/creative_bench_runs.json
  C. judge files (ablation results/judge, paper results/judge, judge_ablation)
  D. OUT_DIR/overrides.csv  (hand-written after reading review_endings.txt;
     optional -- if absent, rule labels are final)

Outputs (OUT_DIR)
  per_story.csv, per_run.csv, model_ceilings.csv, review_endings.txt, SUMMARY.md

Re-run:  python3 audit.py
"""
import csv
import glob
import json
import os
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
ABL = "/project/jevans/maxzhuyt/narrative_project/narrative_ablation"
CWB = "/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
ABL_STORIES = f"{ABL}/results/stories"

# ----------------------------------------------------------------------------
# Ending rule
# ----------------------------------------------------------------------------
END_CHARS = set('.!?…"”’\')]—:;')      # . ! ? … " ” ’ ' ) ] — : ;
DASH_COLON = set('—:;')                             # rule-complete but reviewed by hand
DECOR_LINE = re.compile(r'^[\s\-\*_#~=`•·]*$')    # ---, ***, * * *, ###, ~~~, ```


def strip_decoration(t):
    t = t.rstrip()
    while True:
        old = t
        lines = t.split("\n")
        while lines and DECOR_LINE.match(lines[-1]):
            lines.pop()
        t = "\n".join(lines).rstrip()
        t = t.rstrip("*_#~` \t")
        if t == old:
            return t


def classify_ending(text):
    """Return (ends_mid_sentence: bool, reason: str, last_line: str)."""
    s = strip_decoration(text)
    if not s:
        return None, "empty", ""          # empty generation: neither complete nor mid-sentence
    last_line = [ln for ln in s.split("\n") if ln.strip()][-1].strip()
    if s[-1] in END_CHARS:
        return False, ("punct_dash_colon" if s[-1] in DASH_COLON else "punct"), last_line
    if len(last_line.split()) <= 5:
        return False, "short_last_line", last_line
    return True, "no_terminal_punct", last_line


# ----------------------------------------------------------------------------
# Judge helpers
# ----------------------------------------------------------------------------
def nr_from_positions(positions):
    p10 = [1 - p["yes_rate"] for p in positions if p["pct"] >= 10]
    p80 = [1 - p["yes_rate"] for p in positions if p["pct"] >= 80]
    return (statistics.mean(p10) if p10 else None,
            statistics.mean(p80) if p80 else None, len(p10), len(p80))


def mtime(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")


def is_late_march(path):
    return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m") == "2026-03"


_judge_cache = {}


def load_judge(path):
    if path not in _judge_cache:
        _judge_cache[path] = json.load(open(path)) if os.path.exists(path) else {}
    return _judge_cache[path]


ACT_RE = re.compile(r'"actual_ending":\s*"((?:[^"\\]|\\.)*)"')


def actual_ending(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(65536)
    m = ACT_RE.search(head)
    if not m:
        return None
    return json.loads('"' + m.group(1) + '"')


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


WORD_TARGET_RE = re.compile(r'\b\d[\d,]{2,}\s*(?:[-–]\s*\d[\d,]{2,}\s*)?(?:-\s*)?words?\b', re.I)

# ----------------------------------------------------------------------------
# A. Ablation dirs
# ----------------------------------------------------------------------------
IDEAS = {e["id"]: e for e in json.load(open(f"{ABL}/data/story_ideas.json"))}
IDEA_TEXTS = {(f"{e['title']}\n\n{e['idea']}" if e.get("title") else e["idea"]).strip(): k
              for k, e in IDEAS.items()}
EQB = json.load(open(f"{ABL}/data/creative_writing_prompts_v3.json"))
EQB_BASES = {k: v["writing_prompt"].split("<SEED>")[0].strip() for k, v in EQB.items()}

ABL_JUDGE_DIRS = {
    "results/judge": f"{ABL}/results/judge",
    "results/judge_rep2": f"{ABL}/results/judge_rep2",
    "results/judge_rep3": f"{ABL}/results/judge_rep3",
    "results/judge_rep4": f"{ABL}/results/judge_rep4",
    "results/judge_ablation": f"{ABL}/results/judge_ablation",
    "results/judge_rep1_archived_outlier": f"{ABL}/results/judge_rep1_archived_outlier",
    "results_archived_old_pipeline/judge": f"{ABL}/results_archived_old_pipeline/judge",
    "cwb:results/judge_rep2": f"{CWB}/results/judge_rep2",
    "cwb:results/judge_rep3": f"{CWB}/results/judge_rep3",
    "cwb:results/judge_rep4": f"{CWB}/results/judge_rep4",
    "cwb:results/judge_rep1_archived_outlier_ignore": f"{CWB}/results/judge_rep1_archived_outlier_ignore",
}


def seed_source(meta):
    op = (meta.get("narrative_idea_gen_original_prompt") or "").strip()
    if not op:
        return ""
    if op == "Write a short story.":
        return "dummy('Write a short story.')"
    if op in IDEA_TEXTS:
        return "story_ideas.json"
    for k, b in EQB_BASES.items():
        if op.startswith(b[:200]):
            return "EQ-Bench prompt"
    return "other"


stories = []   # list of dicts

for cfg in sorted(os.listdir(ABL_STORIES)):
    d = f"{ABL_STORIES}/{cfg}"
    for fp in sorted(glob.glob(f"{d}/*_iter*.txt")):
        sid = os.path.basename(fp)[:-4]
        m = re.match(r"(.+)_iter(\d+)$", sid)
        it = int(m.group(2))
        text = read(fp)
        meta_p = fp[:-4] + "_meta.json"
        meta = json.load(open(meta_p)) if os.path.exists(meta_p) else {}
        model = meta.get("model", "")
        jtxt = f"{ABL}/results/stories_txt/{cfg}/{sid}.txt"
        jmatch = None
        if os.path.exists(jtxt):
            jmatch = read(jtxt).strip() == text.strip()
        beats = meta.get("narrative_beats")
        stories.append(dict(
            run=cfg, source="ablation", story_id=sid, iteration=it, model=model,
            text=text, warmup_src=meta.get("narrative_warmup_source_id") or meta.get("narrative_source_id"),
            retry=("step2_retry_attempt=%s%s" % (meta["step2_retry_attempt"],
                   ",fallback" if meta.get("step2_retry_fallback") else "")
                   if "step2_retry_attempt" in meta else ("phase3_new" if meta.get("is_retry") else "")),
            seed_source=seed_source(meta),
            beats_has_word_target=(bool(WORD_TARGET_RE.search(beats)) if beats else None),
            judged_input_txt_exists=os.path.exists(jtxt),
            judged_input_matches=jmatch,
            endings_path=f"{ABL}/results/endings/{cfg}/{sid}_endings.json",
            judge_path=f"{ABL}/results/judge/{cfg}_judge.json",
            judge_key=sid,
        ))

# ----------------------------------------------------------------------------
# B. Paper runs
# ----------------------------------------------------------------------------
RUNS = json.load(open(f"{CWB}/creative_bench_runs.json"))
PAPER = {
    # run_key: (slug used by extraction/endings/judge, judge file, endings dir)
    "narrative_sonnet46__anthropic_claude-sonnet-4-6": (
        "narrative_sonnet46__anthropic_claude-sonnet-4-6",
        f"{CWB}/results/judge/ending_judge.json", f"{CWB}/results/endings"),
    "narrative_opus46__anthropic_claude-opus-4-6": (
        "narrative_opus46__anthropic_claude-opus-4-6",
        f"{CWB}/results/judge/ending_judge_opus46.json", f"{CWB}/results/endings"),
    "narrative_gpt52__openai_gpt-5_2": (
        "narrative_gpt52__openai_gpt-5_2",
        f"{CWB}/results/judge/ending_judge_gpt52.json", f"{CWB}/results/endings"),
    "narrative_gpt5__openai_gpt-5-2025-08-07": (
        "narrative_gpt5__openai_gpt-5-2025-08-07",
        f"{CWB}/results/judge/ending_judge_gpt5.json", f"{CWB}/results/endings"),
    "narrative_sonnet46_nostep0__anthropic_claude-sonnet-4-6": (
        "narrative_sonnet46_nostep0__anthropic_claude-sonnet-4-6",
        f"{ABL}/results/judge_ablation/cwb_narrative_sonnet46_nostep0_judge.json",
        f"{ABL}/results/endings_ablation"),
    "run1__claude-sonnet-4-6": ("claude-sonnet-4-6", f"{CWB}/results/judge/ending_judge.json",
                                f"{CWB}/results/endings"),
    "run1__claude-opus-4-6": ("claude-opus-4-6", f"{CWB}/results/judge/ending_judge.json",
                              f"{CWB}/results/endings"),
    "run1__openai_gpt-5_2": ("openai__gpt-5.2", f"{CWB}/results/judge/ending_judge.json",
                             f"{CWB}/results/endings"),
    # not in the requested list; vanilla counterpart of narrative_gpt5
    "run1__gpt-5-2025-08-07": ("gpt-5-2025-08-07", f"{CWB}/results/judge/ending_judge.json",
                               f"{CWB}/results/endings"),
}
assert all(k in RUNS for k in PAPER), [k for k in PAPER if k not in RUNS]
extra_narr = [k for k in RUNS if k.startswith("narrative_") and k not in PAPER]

for rk, (slug, jfile, edir) in PAPER.items():
    r = RUNS[rk]
    nostep0 = "nostep0" in rk
    for it, tasks in sorted(r["creative_tasks"].items()):
        for pid, t in sorted(tasks.items(), key=lambda x: int(x[0])):
            for mod, res in t["results_by_modifier"].items():
                text = res.get("model_response", "") or ""
                sid = f"i{it}_p{int(pid):02d}"
                if nostep0:
                    jtxt = f"{ABL}/results/stories_txt/cwb_narrative_sonnet46_nostep0/{sid}.txt"
                    ejson = f"{edir}/cwb_narrative_sonnet46_nostep0/{sid}_endings.json"
                    jkey = sid
                else:
                    jtxt = f"{CWB}/stories/{slug}/{sid}.txt"
                    if slug == "gpt-5-2025-08-07":
                        jtxt = f"{CWB}/stories/{slug}/p{int(pid):02d}.txt"
                    ejson = f"{edir}/{slug}/{sid}_endings.json"
                    jkey = f"{slug}/{sid}"
                jmatch = read(jtxt).strip() == text.strip() if (it == "1" and os.path.exists(jtxt)) else None
                beats = res.get("narrative_beats") or res.get("narrative_step1")
                stories.append(dict(
                    run=rk, source="paper", story_id=sid, iteration=int(it),
                    model=r.get("test_model", ""), text=text, retry="", pid=int(pid),
                    warmup_src=res.get("narrative_warmup_source_id") or res.get("narrative_source_id"),
                    seed_source="EQ-Bench prompt",
                    beats_has_word_target=(bool(WORD_TARGET_RE.search(beats)) if beats else None),
                    judged_input_txt_exists=(os.path.exists(jtxt) if it == "1" else None),
                    judged_input_matches=jmatch,
                    endings_path=ejson, judge_path=jfile, judge_key=jkey,
                    generation_failed=res.get("generation_failed"),
                ))


def norm_model(m):
    m = m.lower()
    if "opus-4-6" in m or "opus-4.6" in m:
        return "opus-4.6"
    if "sonnet-4-6" in m or "sonnet-4.6" in m:
        return "sonnet-4.6"
    if "gpt-5.2" in m or "gpt-5_2" in m:
        return "gpt-5.2"
    if "gpt-5-2025" in m:
        return "gpt-5"
    return m


for s in stories:
    s["model_n"] = norm_model(s["model"] or s["run"])
    s["words"] = len(s["text"].split())
    mid, reason, last_line = classify_ending(s["text"])
    s["rule_mid"] = mid
    s["rule_reason"] = reason
    s["last_line"] = last_line
    s["tail120"] = s["text"].rstrip()[-120:]

# ----------------------------------------------------------------------------
# Run metadata
# ----------------------------------------------------------------------------
def abl_cfg(cfg):
    p = f"{ABL_STORIES}/{cfg}/_config.json"
    return json.load(open(p)) if os.path.exists(p) else None


def run_type(s):
    return f"{s['source']}|{STEP2[s['run']]}"


STEP0, STEP1, STEP2, IDEASRC, NOTES = {}, {}, {}, {}, defaultdict(list)
for cfg in sorted(os.listdir(ABL_STORIES)):
    c = abl_cfg(cfg)
    if cfg == "cwb_narrative_sonnet46":
        STEP0[cfg], STEP1[cfg], STEP2[cfg] = "precomputed (6 templates)", "beats", "beats-only"
        IDEASRC[cfg] = "EQ-Bench prompts (32, p27 absent)"
        NOTES[cfg].append("no _config.json; copy of CWB Sonnet-4.6 narrative run (3 iters)")
        continue
    if cfg == "cwb_vanilla_sonnet46":
        STEP0[cfg], STEP1[cfg], STEP2[cfg] = "none", "none", "EQ-Bench prompt"
        IDEASRC[cfg] = "EQ-Bench prompts (32, p27 absent)"
        NOTES[cfg].append("no _config.json; copy of CWB Sonnet-4.6 vanilla run (3 iters); p29_iter2 missing")
        continue
    if c is None:
        STEP0[cfg] = STEP1[cfg] = STEP2[cfg] = IDEASRC[cfg] = ""
        NOTES[cfg].append("EMPTY dir (no _config.json, no stories; stories_txt/ and endings/ dirs of the same name also empty)")
        continue
    if c["step1"] == "none":
        STEP0[cfg] = "none"
        STEP1[cfg] = "none"
        instr = "with_instruction" in c.get("pipeline_repr", "")
        STEP2[cfg] = "idea+instruction" if instr else "raw idea"
    else:
        STEP0[cfg] = c["step0"]
        STEP1[cfg] = "beats" + (f"(max={c['step1_max_beats']})" if c.get("step1_max_beats") else "")
        STEP2[cfg] = "beats-only"
    if c.get("idea_gen"):
        flags = [f for f in ("no_genre_priming", "no_taxonomy", "no_tropes") if c.get(f)]
        IDEASRC[cfg] = "idea_gen(" + (",".join(flags) if flags else "full") + ")"
    else:
        IDEASRC[cfg] = "story_ideas.json (15)"
    if c.get("run_id") != cfg:
        NOTES[cfg].append(f"_config.json run_id='{c.get('run_id')}' != dir name")
    if c.get("idea_gen") and c["step1"] != "none":
        NOTES[cfg].append("README Phase-3 header says 'Step 0: fulltext warmup'; _config.json says step0=title")
    if c.get("iterations") == 3 and c.get("idea_gen"):
        NOTES[cfg].append("_config.json (last launch) says iterations=3; dir mixes p01-p33 x3 and t-series x1")
NOTES["idea_no_taxonomy_bugged"].append(
    "flags identical to idea_tropes_only (no_genre_priming, no_taxonomy, tropes on); "
    "results/logs/gen_idea_no_taxonomy.log and gen_idea_tropes_only.log both start 2026-03-20 13:28:00")

for rk in PAPER:
    r = RUNS[rk]
    if rk.startswith("run1__"):
        STEP0[rk], STEP1[rk], STEP2[rk] = "none", "none", "EQ-Bench prompt"
    elif "nostep0" in rk:
        STEP0[rk], STEP1[rk], STEP2[rk] = "none", "beats", "beats-only"
    else:
        STEP0[rk], STEP1[rk], STEP2[rk] = "precomputed (6 templates)", "beats", "beats-only"
    IDEASRC[rk] = "EQ-Bench prompts (32; p27 absent; each says '1000 words')"
    NOTES[rk].append(f"generated {r.get('start_time','')[:10]}")
NOTES["run1__gpt-5-2025-08-07"].append("NOT in requested list; vanilla counterpart of narrative_gpt5")

# ----------------------------------------------------------------------------
# Overrides
# ----------------------------------------------------------------------------
OVR_PATH = f"{OUT_DIR}/overrides.csv"
overrides = {}
if os.path.exists(OVR_PATH):
    with open(OVR_PATH) as f:
        for row in csv.DictReader(f):
            overrides[(row["run"], row["story_id"])] = row
used = set()
for s in stories:
    k = (s["run"], s["story_id"])
    if k in overrides:
        o = overrides[k]
        assert o["rule_ends_mid_sentence"] == str(s["rule_mid"]), (k, o, s["rule_mid"])
        s["ends_mid_sentence"] = o["final_ends_mid_sentence"] == "True"
        s["override"] = o["reason"]
        used.add(k)
    else:
        s["ends_mid_sentence"] = s["rule_mid"]
        s["override"] = ""
unused = set(overrides) - used
assert not unused, unused

# manual annotation: outputs that are planning text, not a story
NS_PATH = f"{OUT_DIR}/non_story.csv"
non_story = {}
if os.path.exists(NS_PATH):
    with open(NS_PATH) as f:
        non_story = {(r["run"], r["story_id"]): r["kind"] for r in csv.DictReader(f)}
for s in stories:
    s["non_story"] = non_story.get((s["run"], s["story_id"]), "")
assert set(non_story) <= {(s["run"], s["story_id"]) for s in stories}

# identical-text links between the ablation cwb_* copies and the paper runs
by_text = defaultdict(list)
for s in stories:
    if s["words"]:
        by_text[s["text"].strip()].append((s["run"], s["story_id"]))
for s in stories:
    s["dup_of"] = ";".join(f"{r}/{i}" for r, i in by_text.get(s["text"].strip(), [])
                           if (r, i) != (s["run"], s["story_id"]))

# ----------------------------------------------------------------------------
# Judge join
# ----------------------------------------------------------------------------
for s in stories:
    jd = load_judge(s["judge_path"]) if s["judge_path"] else {}
    e = jd.get(s["judge_key"])
    s["judged"] = e is not None
    s["judge_file"] = (os.path.relpath(s["judge_path"], "/project/jevans/maxzhuyt") +
                       f" [{mtime(s['judge_path'])}]") if (e is not None) else ""
    s["judged_late_march"] = bool(e is not None and is_late_march(s["judge_path"]))
    if e is not None:
        mn, ln, n10, n80 = nr_from_positions(e["positions"])
        s.update(mean_nr=mn, late_nr=ln, n_pos_pct10=n10, n_pos_pct80=n80)
        s["endings_actual_ending"] = actual_ending(s["endings_path"])
    else:
        s.update(mean_nr=None, late_nr=None, n_pos_pct10=None, n_pos_pct80=None,
                 endings_actual_ending=None)

# archived (2026-03-19/20) ablation judge + endings, for comparison only
for s in stories:
    s["mean_nr_oldjudge"] = None
    s["old_endings_same_ending"] = None
    if s["source"] != "ablation":
        continue
    name = "idea_no_taxonomy" if s["run"] == "idea_no_taxonomy_bugged" else s["run"]
    oj = load_judge(f"{ABL}/results_archived_old_pipeline/judge/{name}_judge.json")
    e = oj.get(s["story_id"])
    if e is not None:
        s["mean_nr_oldjudge"] = nr_from_positions(e["positions"])[0]
        if s["judged"]:
            a_old = actual_ending(f"{ABL}/results_archived_old_pipeline/endings/{name}/{s['story_id']}_endings.json")
            s["old_endings_same_ending"] = (a_old == s["endings_actual_ending"]) if a_old is not None else None

# ----------------------------------------------------------------------------
# Model ceilings
# ----------------------------------------------------------------------------
# Per model, over UNIQUE non-empty story texts (the cwb_* ablation dirs duplicate paper texts):
#   raw_min      = min word count among mid-sentence stories (requested number)
#   pile-up test = median(mid-sentence words) >= 75th percentile(complete words)
#   threshold    = 10th percentile (nearest rank) of mid-sentence word counts, used only if
#                  the pile-up test holds; at_length_ceiling = words >= threshold
#   (raw_min is not used as the threshold because mid-sentence stories also occur far below
#    the cap, e.g. Opus 49/341/381 words, GPT-5 1017 words; see model_ceilings.csv)
def nearest_rank(xs, q):
    xs = sorted(xs)
    return xs[max(0, -(-int(q * 100) * len(xs) // 100) - 1)] if xs else None


ceil = {}
model_rows = []
for mdl in sorted({s["model_n"] for s in stories}):
    seen, ss = set(), []
    for s in stories:
        if s["model_n"] == mdl and s["words"] and s["text"].strip() not in seen:
            seen.add(s["text"].strip())
            ss.append(s)
    mids = sorted(s["words"] for s in ss if s["ends_mid_sentence"])
    comps = sorted(s["words"] for s in ss if s["ends_mid_sentence"] is False)
    raw_min = mids[0] if mids else None
    p10 = nearest_rank(mids, 0.10)
    pile = bool(mids and comps and statistics.median(mids) >= nearest_rank(comps, 0.75))
    thr = p10 if pile else None
    ceil[mdl] = thr
    model_rows.append(dict(
        model=mdl, n_unique_nonempty=len(ss), n_mid_sentence=len(mids), n_complete=len(comps),
        raw_min_words_mid=raw_min,
        n_complete_at_or_above_raw_min=sum(1 for w in comps if raw_min is not None and w >= raw_min),
        p10_words_mid=p10, median_words_mid=(statistics.median(mids) if mids else None),
        max_words_mid=(mids[-1] if mids else None),
        p75_words_complete=nearest_rank(comps, 0.75), max_words_complete=(comps[-1] if comps else None),
        pile_up=pile, ceiling_threshold=thr,
        n_complete_at_or_above_threshold=(sum(1 for w in comps if w >= thr) if thr else None),
        lowest_10_mid=" ".join(map(str, mids[:10])),
    ))
for s in stories:
    t = ceil[s["model_n"]]
    s["at_length_ceiling"] = bool(t is not None and s["words"] >= t)

with open(f"{OUT_DIR}/model_ceilings.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(model_rows[0].keys()))
    w.writeheader()
    w.writerows(model_rows)

# ----------------------------------------------------------------------------
# Review file (what was looked at by hand)
# ----------------------------------------------------------------------------
def show(s):
    tail = s["tail120"].replace("\n", " ⏎ ")
    return f"{s['run']} | {s['story_id']} | w={s['words']} | {s['rule_reason']} | ...{tail}"


rng = random.Random(0)
with open(f"{OUT_DIR}/review_endings.txt", "w") as f:
    f.write("# A. rule-flagged ends_mid_sentence (all)\n")
    for s in stories:
        if s["rule_mid"]:
            f.write(show(s) + "\n")
    f.write("\n# B. rule-complete via short_last_line (all)\n")
    for s in stories:
        if s["rule_reason"] == "short_last_line":
            f.write(show(s) + "\n")
    f.write("\n# C. rule-complete ending in em dash / colon / semicolon (all)\n")
    for s in stories:
        if s["rule_reason"] == "punct_dash_colon":
            f.write(show(s) + "\n")
    f.write("\n# E. empty generations (0 words)\n")
    for s in stories:
        if s["rule_reason"] == "empty":
            f.write(show(s) + "\n")
    f.write("\n# F. scan of all rule-complete (punct) endings: dangling opening quote / markup / speaker label\n")
    pat_open = re.compile(r'(^|[\s:,(\[\u2014-])["\u201c\'\u2018\u2019]$')
    pat_label = re.compile(r"^(\*\*)?[A-Z][A-Z .'-]{0,30}[.:](\*\*)?$")
    for s in stories:
        if s["rule_reason"] != "punct":
            continue
        ll = s["last_line"]
        raw_last = [x for x in s["text"].rstrip().split("\n") if x.strip()][-1].strip()
        why = [w for w, c in (("opening-quote", pat_open.search(ll)), ("tiny-last-line", len(ll) <= 3),
                              ("speaker-label", pat_label.match(ll) and len(ll.split()) <= 3),
                              ("dangling-markup", re.fullmatch(r"[\*_>\s\"\u201c\[]+", raw_last))) if c]
        if why:
            f.write(show(s) + " | " + ",".join(why) + "\n")
    f.write("\n# G. scan of all outputs for planning-text openers (first 400 chars)\n")
    pat_plan = re.compile(r"^\s*(Let me|The user|I need to|I'll |I will |I'm going|This is a (complex|rich|fascinating|layered|challenging)|Here's|Here is|Okay|OK[,.]|Alright|Looking at|First,|Now I|I want to)", re.I)
    for s in stories:
        if pat_plan.search(s["text"][:400]):
            f.write(f"{s['run']} | {s['story_id']} | w={s['words']} | head: {s['text'][:120].replace(chr(10), ' ')}\n")
    f.write("\n# H. ablation stories < 2000 words that end complete\n")
    for s in stories:
        if s["source"] == "ablation" and 0 < s["words"] < 2000 and s["ends_mid_sentence"] is False \
                and not s["run"].startswith("cwb_"):
            f.write(show(s) + "\n")
    f.write("\n# D. random 20 rule-complete (reason=punct) per run type, seed 0\n")
    by_type = defaultdict(list)
    for s in stories:
        if s["rule_reason"] == "punct":
            by_type[run_type(s)].append(s)
    for rt in sorted(by_type):
        f.write(f"## {rt} (pool {len(by_type[rt])})\n")
        for s in rng.sample(by_type[rt], min(20, len(by_type[rt]))):
            f.write(show(s) + "\n")

# ----------------------------------------------------------------------------
# per_story.csv
# ----------------------------------------------------------------------------
PS_COLS = ["run", "source", "story_id", "iteration", "words", "ends_mid_sentence",
           "at_length_ceiling", "judged_late_march", "mean_nr", "late_nr",
           # extras
           "model", "rule_ends_mid_sentence", "rule_reason", "override", "judged",
           "judge_file", "n_pos_pct10", "n_pos_pct80", "judged_input_matches",
           "retry", "seed_source", "warmup_src", "beats_has_word_target", "not_a_story_kind",
           "mean_nr_old_pipeline_judge", "old_endings_same_actual_ending",
           "identical_text_in", "endings_actual_ending", "last_100_chars"]


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, str):
        return v.replace("\r", "\\r").replace("\n", "\\n")
    return "" if v is None else v


with open(f"{OUT_DIR}/per_story.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(PS_COLS)
    for s in stories:
        w.writerow([fmt(x) for x in [
            s["run"], s["source"], s["story_id"], s["iteration"], s["words"],
            s["ends_mid_sentence"], s["at_length_ceiling"], s["judged_late_march"],
            s["mean_nr"], s["late_nr"], s["model_n"], s["rule_mid"], s["rule_reason"],
            s["override"], s["judged"], s["judge_file"], s["n_pos_pct10"], s["n_pos_pct80"],
            s["judged_input_matches"], s["retry"], s["seed_source"], s["warmup_src"],
            s["beats_has_word_target"], s["non_story"], s["mean_nr_oldjudge"], s["old_endings_same_ending"], s["dup_of"],
            (s["endings_actual_ending"] or "")[:100] if s["endings_actual_ending"] is not None else None,
            s["text"].rstrip()[-100:]]])

# ----------------------------------------------------------------------------
# Other judge dirs per run
# ----------------------------------------------------------------------------
def other_judges(run):
    out = []
    if run in PAPER:
        slug = PAPER[run][0]
        alias = {"narrative_sonnet46__anthropic_claude-sonnet-4-6": "cwb_narrative_sonnet46",
                 "run1__claude-sonnet-4-6": "cwb_vanilla_sonnet46"}.get(run)
        cands = {
            "cwb:results/judge": f"{CWB}/results/judge",
            "cwb:results_archived_one_sentence/judge": f"{CWB}/results_archived_one_sentence/judge",
        }
        for name, d in cands.items():
            for fp in sorted(glob.glob(f"{d}/*.json")):
                if fp == PAPER[run][1]:
                    continue
                jd = load_judge(fp)
                if any(k.startswith(slug + "/") for k in jd):
                    out.append(f"{name}/{os.path.basename(fp)}")
        if alias:
            for name, d in ABL_JUDGE_DIRS.items():
                if name == "results/judge":
                    continue
                if os.path.exists(f"{d}/{alias}_judge.json"):
                    out.append(f"{name}/{alias}_judge.json")
        return out
    for name, d in ABL_JUDGE_DIRS.items():
        if name == "results/judge":
            continue
        if os.path.exists(f"{d}/{run}_judge.json"):
            out.append(f"{name}/{run}_judge.json")
    if run == "idea_no_taxonomy_bugged" and os.path.exists(
            f"{ABL}/results_archived_old_pipeline/judge/idea_no_taxonomy_judge.json"):
        out.append("results_archived_old_pipeline/judge/idea_no_taxonomy_judge.json")
    return out


# ----------------------------------------------------------------------------
# per_run.csv
# ----------------------------------------------------------------------------
def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


run_order = sorted({s["run"] for s in stories if s["source"] == "ablation"}) + list(PAPER)
run_order = [r for r in run_order]
empty_dirs = [c for c in sorted(os.listdir(ABL_STORIES))
              if not glob.glob(f"{ABL_STORIES}/{c}/*_iter*.txt")]

PR_COLS = ["run", "source", "model", "step0", "step1", "step2_input", "idea_source",
           "n_generated", "n_iter1", "stats_set", "words_min", "words_median", "words_max",
           "n_mid_sentence", "share_mid_sentence", "min_words_mid_sentence_run",
           "model_ceiling_words", "n_at_ceiling_complete",
           "n_judged", "n_judged_late_march", "n_judged_mid_sentence", "n_judged_complete",
           "mean_nr_judged", "mean_nr_judged_complete_only",
           "late_nr_judged", "late_nr_judged_complete_only",
           "n_judged_input_mismatch", "n_overrides", "n_mid_sentence_other_iters",
           "n_empty", "n_not_a_story", "n_judged_not_a_story", "n_words_2000_3000",
           "mean_nr_old_pipeline_judge", "n_old_judge_actual_ending_differs",
           "judge_file", "other_judge_dirs", "notes"]
per_run = []
for run in run_order + empty_dirs:
    allr = [s for s in stories if s["run"] == run]
    src = "paper" if run in PAPER else "ablation"
    if not allr:
        per_run.append(dict(run=run, source=src, notes="; ".join(NOTES[run]),
                            n_generated=0, n_iter1=0, step0="", step1="", step2_input="",
                            idea_source="", other_judge_dirs=" ".join(other_judges(run))))
        continue
    stats = [s for s in allr if s["iteration"] == 1] if src == "paper" else allr
    others = [s for s in allr if s["iteration"] != 1] if src == "paper" else []
    words = [s["words"] for s in stats]
    mids = [s for s in stats if s["ends_mid_sentence"]]
    judged = [s for s in allr if s["judged"]]
    jcomp = [s for s in judged if s["ends_mid_sentence"] is False]
    jfiles = sorted({s["judge_file"] for s in judged})
    notes = list(NOTES[run])
    rc = Counter(s["retry"].split("=")[0] for s in allr if s["retry"])
    if rc:
        notes.append("retries: " + ", ".join(f"{k}={v}" for k, v in rc.items()))
    sc = Counter(s["seed_source"] for s in allr if s["seed_source"])
    if sc and src == "ablation":
        notes.append("idea seeds: " + ", ".join(f"{k}={v}" for k, v in sorted(sc.items())))
    nj_missing_txt = sum(1 for s in judged if s["judged_input_txt_exists"] is False)
    if nj_missing_txt:
        notes.append(f"{nj_missing_txt} judged stories lack judged-input txt")
    bw = [s["beats_has_word_target"] for s in stats if s["beats_has_word_target"] is not None]
    if bw:
        notes.append(f"beats mention a word count: {sum(bw)}/{len(bw)}")
    if src == "ablation" and judged and len(judged) < len(allr):
        notes.append(f"judge covers {len(judged)}/{len(allr)} files")
    wsrc = Counter(("NYer-id" if s["warmup_src"][0].isdigit() else s["warmup_src"][0] + "*")
                   for s in allr if s["warmup_src"])
    if wsrc:
        notes.append("warmup source ids: " + ", ".join(f"{k}={v}" for k, v in sorted(wsrc.items())))
    dl = Counter(d.split("/")[0] for s in allr for d in s["dup_of"].split(";") if d)
    if dl:
        notes.append("identical text in: " + ", ".join(f"{k} ({v})" for k, v in sorted(dl.items())))
    per_run.append(dict(
        run=run, source=src, model=allr[0]["model_n"],
        step0=STEP0.get(run, ""), step1=STEP1.get(run, ""), step2_input=STEP2.get(run, ""),
        idea_source=IDEASRC.get(run, ""),
        n_generated=len(allr), n_iter1=sum(1 for s in allr if s["iteration"] == 1),
        stats_set=("iteration 1" if src == "paper" else "all files in dir"),
        words_min=min(words), words_median=statistics.median(words), words_max=max(words),
        n_mid_sentence=len(mids),
        share_mid_sentence=len(mids) / max(1, sum(1 for s in stats if s["words"])),
        min_words_mid_sentence_run=(min(s["words"] for s in mids) if mids else None),
        model_ceiling_words=ceil[allr[0]["model_n"]],
        n_at_ceiling_complete=sum(1 for s in stats if s["at_length_ceiling"] and s["ends_mid_sentence"] is False),
        n_judged=len(judged),
        n_judged_late_march=sum(1 for s in allr if s["judged_late_march"]),
        n_judged_mid_sentence=sum(1 for s in judged if s["ends_mid_sentence"]),
        n_judged_complete=len(jcomp),
        mean_nr_judged=mean_or_none([s["mean_nr"] for s in judged]),
        mean_nr_judged_complete_only=mean_or_none([s["mean_nr"] for s in jcomp]),
        late_nr_judged=mean_or_none([s["late_nr"] for s in judged]),
        late_nr_judged_complete_only=mean_or_none([s["late_nr"] for s in jcomp]),
        n_judged_input_mismatch=sum(1 for s in judged if s["judged_input_matches"] is False),
        n_overrides=sum(1 for s in allr if s["override"]),
        n_mid_sentence_other_iters=(f"{sum(1 for s in others if s['ends_mid_sentence'])}/{len(others)}"
                                    if others else ""),
        n_empty=sum(1 for s in allr if not s["words"]),
        n_not_a_story=sum(1 for s in allr if s["non_story"]),
        n_judged_not_a_story=sum(1 for s in judged if s["non_story"]),
        mean_nr_old_pipeline_judge=mean_or_none([s["mean_nr_oldjudge"] for s in allr]),
        n_old_judge_actual_ending_differs=(sum(1 for s in judged if s["old_endings_same_ending"] is False)
                                           if src == "ablation" else None),
        n_words_2000_3000=sum(1 for s in stats if 2000 <= s["words"] <= 3000),
        judge_file=" ".join(jfiles),
        other_judge_dirs=" ".join(other_judges(run)),
        notes="; ".join(notes),
    ))

with open(f"{OUT_DIR}/per_run.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=PR_COLS)
    w.writeheader()
    for r in per_run:
        w.writerow({k: fmt(r.get(k)) for k in PR_COLS})

# ----------------------------------------------------------------------------
# SUMMARY.md
# ----------------------------------------------------------------------------
def f3(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.3f}"
    return str(v)


SHORT = {k: (k.split("__")[0] if k.startswith("narrative_") else k) for k in run_order + empty_dirs}
L = []
L.append("# Story-generation validity audit (Spoiler Alert / 100-Endings)\n")
L.append(f"Generated by `audit.py` on {datetime.now().strftime('%Y-%m-%d %H:%M')}. "
         f"Stories audited: {len(stories)} ({sum(1 for s in stories if s['source']=='ablation')} ablation files, "
         f"{sum(1 for s in stories if s['source']=='paper')} paper-run responses over all iterations). "
         f"Manual overrides applied: {len(used)} (see `overrides.csv`).\n")
L.append("## Definitions\n")
L.append("- **words**: `len(text.split())` (same as the pipeline's `words`).")
L.append("- **ends_mid_sentence (rule)**: strip trailing whitespace, trailing decoration-only lines (`---`, `***`, `###`, `~~~`, backticks) and trailing `* _ # ~ \\``; "
         "complete if the last character is one of `. ! ? … \" ” ’ ' ) ] — : ;`, or if the final non-empty line has <= 5 words; otherwise mid-sentence. "
         "Empty responses (0 words) are left blank (neither).")
L.append("- **Manual review**: every rule-flagged story (A), every story made complete only by the <=5-word rule (B), every ending on — : ; (C), "
         "20 random rule-complete stories per run type with seed 0 (D), plus a regex scan of all rule-complete endings for a dangling opening quote, "
         "dangling `*`/`**`, or a bare speaker label (scan). All looked-at endings are in `review_endings.txt`; every change is in `overrides.csv`.")
L.append("- **Run stats set**: ablation = every story file in the dir; paper = iteration 1 (other iterations reported in `n_mid_sentence_other_iters`). "
         "`share_mid_sentence` = n_mid_sentence / non-empty stories in that set.")
L.append("- **at_length_ceiling**: per model over unique non-empty texts; if median(words of mid-sentence stories) >= 75th percentile(words of complete stories), "
         "threshold = 10th percentile (nearest rank) of mid-sentence word counts and at_length_ceiling = words >= threshold; otherwise no threshold. "
         "The requested raw minimum is reported beside it (`model_ceilings.csv`).")
L.append("- **mean_nr** = mean over positions with pct >= 10 of (1 - yes_rate); **late_nr** = same over pct >= 80. Run means are unweighted means over judged stories "
         "(all stories present in the judge file, not the 15-story selection of ANALYSIS.md).")
L.append("- **judged_late_march**: the story key is present in a judge file written in 2026-03 (ablation: `results/judge/{config}_judge.json`, 2026-03-29; "
         "paper: `ending_judge*.json`, 2026-03-29..31). The no-Step-0 run was judged 2026-06-01 (`judge_ablation/`) with `sbatch/run_judge_rep.py`, "
         "whose system prompt and Qwen3-8B model are the same text as the March scripts.\n")

L.append("## Per-run table (generation)\n")
L.append("| run | model | step0 | step1 | step2_input | idea_source | n_gen | n_iter1 | words min/med/max | n_mid (share) | n_at_ceiling_complete | n_empty | n_not_a_story |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in per_run:
    if not r.get("n_generated"):
        L.append(f"| {SHORT[r['run']]} | | | | | | 0 | 0 | | | | | |")
        continue
    L.append(f"| {SHORT[r['run']]} | {r['model']} | {r['step0']} | {r['step1']} | {r['step2_input']} | {r['idea_source'][:40]} | "
             f"{r['n_generated']} | {r['n_iter1']} | {r['words_min']}/{f3(float(r['words_median']))}/{r['words_max']} | "
             f"{r['n_mid_sentence']} ({r['share_mid_sentence']:.3f}) | {r['n_at_ceiling_complete']} | {r['n_empty']} | {r['n_not_a_story']} |")
L.append("")
L.append("## Per-run table (judge)\n")
L.append("| run | n_judged | n_judged_late_march | n_judged_mid | n_judged_complete | n_judged_not_a_story | mean_nr | mean_nr complete-only | late_nr | late_nr complete-only | judged-input text mismatch | mean_nr, archived 03-19/20 judge (all keys) | other judge dirs |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in per_run:
    if not r.get("n_generated"):
        L.append(f"| {SHORT[r['run']]} | 0 | 0 | | | | | | | | | | {r.get('other_judge_dirs','')} |")
        continue
    L.append(f"| {SHORT[r['run']]} | {r['n_judged']} | {r['n_judged_late_march']} | {r['n_judged_mid_sentence']} | {r['n_judged_complete']} | {r['n_judged_not_a_story']} | "
             f"{f3(r['mean_nr_judged'])} | {f3(r['mean_nr_judged_complete_only'])} | {f3(r['late_nr_judged'])} | "
             f"{f3(r['late_nr_judged_complete_only'])} | {r['n_judged_input_mismatch']} | {f3(r['mean_nr_old_pipeline_judge'])} | {r['other_judge_dirs'].replace(' ', '<br>')} |")
L.append("")
L.append("## Notes per run\n")
for r in per_run:
    if r.get("notes"):
        L.append(f"- **{SHORT[r['run']]}**: {r['notes']}")
L.append("")
L.append("## Length ceiling per model (`model_ceilings.csv`)\n")
L.append("| model | unique non-empty | n_mid | n_complete | raw min words (mid) | complete >= raw min | p10 words (mid) | median (mid) | p75 words (complete) | pile-up | threshold | complete >= threshold | lowest 10 mid-sentence word counts |")
L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for m in model_rows:
    L.append(f"| {m['model']} | {m['n_unique_nonempty']} | {m['n_mid_sentence']} | {m['n_complete']} | {m['raw_min_words_mid']} | "
             f"{m['n_complete_at_or_above_raw_min']} | {m['p10_words_mid']} | {f3(m['median_words_mid'])} | {m['p75_words_complete']} | "
             f"{m['pile_up']} | {f3(m['ceiling_threshold'])} | {f3(m['n_complete_at_or_above_threshold'])} | {m['lowest_10_mid']} |")
L.append("")
L.append("## 10 example flagged endings (judged stories, last 100 characters; seed 1)\n")
rng2 = random.Random(1)
pool = [s for s in stories if s["judged"] and s["ends_mid_sentence"]]
seen_runs, ex = set(), []
for s in rng2.sample(pool, len(pool)):
    if s["run"] not in seen_runs:
        seen_runs.add(s["run"]); ex.append(s)
    if len(ex) == 10:
        break
for s in ex:
    tail = s["text"].rstrip()[-100:].replace("\n", " / ").replace("|", "\\|")
    L.append(f"- `{SHORT[s['run']]}` {s['story_id']} ({s['words']} words): ...{tail}")
L.append("")
L.append(f"## Overrides ({len(used)})\n")
L.append("| run | story_id | rule -> final | section | reason |")
L.append("|---|---|---|---|---|")
for (run, sid), o in overrides.items():
    L.append(f"| {SHORT.get(run, run)} | {sid} | {o['rule_ends_mid_sentence']} -> {o['final_ends_mid_sentence']} | {o['review_section']} | {o['reason'].replace('|', '/')} |")
L.append("")
BORDERLINE = [  # looked at by hand, kept as the rule labels them (complete)
    ("narrative_sonnet46_nostep0__anthropic_claude-sonnet-4-6", "i1_p02", "letter breaks off 'But Mum. Could you come' far below the length cap; read as a deliberate unfinished sentence"),
    ("narrative_sonnet46__anthropic_claude-sonnet-4-6", "i1_p14", "'I go back to the beginning. / One—' (dance count restarting); read as deliberate"),
    ("idea_no_tropes", "p02_iter3", "'I will try again. This story could be called —' echoes the opening line; read as deliberate"),
    ("full_fulltext", "t2_005_iter3", "ends on a closed interrupted quote '\"Your *office*—\"'"),
    ("full_title", "t2_004_iter2", "ends on a closed quote followed by two empty blockquote lines '>'"),
    ("idea_vanilla", "t2_003_iter1", "ends on a full sentence followed by '---' (section break, nothing after)"),
]
smap = {(s["run"], s["story_id"]): s for s in stories}
L.append(f"## Borderline endings kept as the rule labels them ({len(BORDERLINE)})\n")
for run, sid, why in BORDERLINE:
    st_ = smap[(run, sid)]
    L.append(f"- `{SHORT[run]}` {sid} ({st_['words']} words; ends_mid_sentence={st_['ends_mid_sentence']}; "
             f"at_length_ceiling={st_['at_length_ceiling']}; judged={st_['judged']}): {why}")
L.append("")
L.append(f"## Outputs that are not a full story ({len(non_story)}; `non_story.csv`)\n")
for s in stories:
    if s["non_story"]:
        L.append(f"- `{SHORT[s['run']]}` {s['story_id']} [{s['non_story']}] ({s['words']} words; judged={s['judged']}; ends_mid_sentence={s['ends_mid_sentence']}): "
                 f"{s['text'][:90].strip().replace(chr(10), ' ')}...")
L.append("")
if extra_narr:
    L.append(f"narrative_* run keys present but not audited: {extra_narr}")
open(f"{OUT_DIR}/SUMMARY.md", "w").write("\n".join(L) + "\n")

print(f"stories={len(stories)} overrides_applied={len(used)} "
      f"rule_mid={sum(1 for s in stories if s['rule_mid'])} final_mid={sum(1 for s in stories if s['ends_mid_sentence'])}")
print("extra narrative_* keys not audited:", extra_narr)
for r in model_rows:
    print(r)
