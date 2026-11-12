#!/usr/bin/env python3
"""Apply audit.py's end-of-text rule to the two model groups audit.py does not cover:
the Top-10 zero-shot stories (Tables 1, 6) and the Kimi-K2 pipeline run (human study, Table 8).
Input: the last sentence of each story (`actual_ending` in the ending files of
creative-writing-bench/results/endings/<model_slug>/). Output: other_groups.csv.
Rule labels only; rule-positive and short-last-line cases were then read by hand (see OVERRIDES below)."""
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
END_CHARS = set('.!?…"”’\')]—:;')
DECOR_LINE = re.compile(r'^[\s\-\*_#~=`•·]*$')

def strip_decoration(t):  # copied from audit.py
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

def classify_ending(text):  # copied from audit.py
    s = strip_decoration(text)
    if not s:
        return None, "empty", ""
    last_line = [ln for ln in s.split("\n") if ln.strip()][-1].strip()
    if s[-1] in END_CHARS:
        return False, "punct", last_line
    if len(last_line.split()) <= 5:
        return False, "short_last_line", last_line
    return True, "no_terminal_punct", last_line

CWB = "/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench/results/endings"
# (model_slug, story_id) -> (final_label, reason). Set after reading the last 250 characters of every
# rule-positive and short_last_line case in creative_bench_runs.json (iteration 1).
_H = "openrouter__horizon-alpha"
OVERRIDES = {
    (_H, "i1_p02"): (True, "cut inside a letter header ('Letter: 16'); 3,138 words"),
    (_H, "i1_p07"): (True, "clause without final period ('The plates stack higher'); 2,666 words; every complete story of this model ends with punctuation"),
    (_H, "i1_p13"): (True, "clause without final period ('Liu's eyes narrow'); 3,048 words; same reasoning"),
    (_H, "i1_p19"): (True, "ends on the article 'A'; 3,037 words"),
    (_H, "i1_p24"): (True, "cut inside an addendum header ('Addendum 6579'); 2,437 words"),
    (_H, "i1_p28"): (True, "ends 'After school, the cat was'; 3,112 words"),
    (_H, "i1_p31"): (True, "ends 'It was closer, like'; 3,125 words"),
}
# Kept as complete after reading: signatures and sign-offs (all i1_p02 letters), o3 word-count lines,
# Kimi-K2 i1_p02 (verse coda) and i1_p23 (memo timestamp).
if __name__ == "__main__":
    groups = {"kimik2_pipeline": ["narrative_kimi__moonshotai_kimi-k2"]}
    top = [os.path.basename(p)[:-9] for p in glob.glob(os.path.join(os.path.dirname(__file__), "../../results/endings/top10_zeroshot/*.jsonl.gz"))]
    groups["top10_zeroshot"] = sorted(top)
    rows = []
    for g, slugs in groups.items():
        for slug in slugs:
            for p in sorted(glob.glob(f"{CWB}/{slug}/i1_p*_endings.json")):
                d = json.load(open(p))
                mid, reason, last = classify_ending(d["actual_ending"])
                key = (slug, d["story_id"])
                if key in OVERRIDES:
                    mid, reason = OVERRIDES[key]
                rows.append(dict(group=g, model_slug=slug, story_id=d["story_id"], ends_mid_sentence=mid,
                                 reason=reason, last_80_chars=d["actual_ending"][-80:]))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "other_groups.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(Counter((r["group"], r["ends_mid_sentence"], r["reason"]) for r in rows))
