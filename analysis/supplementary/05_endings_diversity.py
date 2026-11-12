#!/usr/bin/env python3
"""
ANALYSIS 5 - Lexical diversity of the 100 independently-generated endings.

For each evaluated position in a story, the generator (Qwen3-32B, T=1.2) produced
100 endings INDEPENDENTLY (not as one ranked list / not beam search): each is an
i.i.d. sample at high temperature. This script quantifies how lexically diverse /
distinct those 100 endings are, using LEXICAL (not embedding) measures:

  (1) mean Self-BLEU   (BLEU-4, lower => more diverse / less self-overlap)
       Self-BLEU (Zhu et al. 2018 / Texygen): each ending = hypothesis,
       the other 99 endings = reference set; corpus-style modified n-gram
       precision with brevity penalty; average over the 100 hypotheses.
  (2) distinct-1       unique unigrams / total unigrams  (higher => more diverse)
  (3) distinct-2       unique bigrams  / total bigrams   (higher => more diverse)
  (4) mean pairwise TF-IDF cosine  (lower => more diverse / less similar)

Aggregation: metrics computed per position over the 100 endings, averaged
per story (over sampled positions), then averaged per group (over sampled stories).

Groups (9): 6 model conditions + NewYorker + TellMeAStory + StoryStar.

Caps (documented):
  - <=30 stories/group, chosen deterministically as the first 30 by sorted story_id.
  - <=8 positions/story, chosen deterministically as an evenly-spaced subsample of
    the canonical no-rate position set (positions with pct_story_revealed >= 10,
    sorted by pct). Capping keeps Self-BLEU/TF-IDF tractable and balances stories
    that have very different numbers of evaluated positions.

Goals:
  (a) confirm endings are genuinely diverse / independent (reviewer R2);
  (b) test whether higher-no-rate corpora have more lexically diverse endings
      (correlate group diversity vs group no-rate).

Outputs:
  analyses/05_diversity.csv          (per-group summary + per-story detail)
  analyses/05_endings_diversity.md   (written separately by the .md generation step)
"""
import os, glob, json, math, re, random
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = "/project/jevans/maxzhuyt/narrative_project/paper_revision"
MODEL_BASE = "/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench/results/endings"
HUMAN_BASE = "/project/jevans/maxzhuyt/narrative_project/results_endings"

STORY_CAP = 30
POS_CAP = 8
PCT_MIN = 10.0          # canonical no-rate "story mean" position filter
N_ENDINGS_EXPECT = 100
SEED = 12345            # only used if a deterministic tie-break is ever needed

# ---- group definitions ----------------------------------------------------
# Model conditions: (group_label, slug)
MODEL_GROUPS = [
    ("Sonnet4.6-vanilla",  "claude-sonnet-4-6"),
    ("Sonnet4.6-pipeline", "narrative_sonnet46__anthropic_claude-sonnet-4-6"),
    ("Opus4.6-vanilla",    "claude-opus-4-6"),
    ("Opus4.6-pipeline",   "narrative_opus46__anthropic_claude-opus-4-6"),
    ("GPT5.2-vanilla",     "openai__gpt-5.2"),
    ("GPT5.2-pipeline",    "narrative_gpt52__openai_gpt-5_2"),
]
# Human corpora: (group_label, [corpus subdir(s)])
HUMAN_GROUPS = [
    ("NewYorker",    ["nyer_expanded", "nyer_83"]),
    ("TellMeAStory", ["tell_me_a_story"]),
    ("StoryStar",    ["storystar"]),
]

# Group no-rate (story mean) from validated validation_summary.json (Table 1 / Table 2).
# Used for goal (b) correlation. These were reproduced against published numbers.
GROUP_NORATE = {
    "Sonnet4.6-vanilla":  0.6059,
    "Sonnet4.6-pipeline": 0.7494,
    "Opus4.6-vanilla":    0.6107,
    "Opus4.6-pipeline":   0.6966,
    "GPT5.2-vanilla":     0.6942,
    "GPT5.2-pipeline":    0.7397,
    "NewYorker":          0.7647,
    "TellMeAStory":       0.7208,
    "StoryStar":          0.6562,
}

_token_re = re.compile(r"[a-z0-9']+")
def tokenize(s):
    return _token_re.findall(s.lower())

# ---- Self-BLEU (BLEU-4) ----------------------------------------------------
def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else Counter()

def modified_precision_counts(hyp_ng, ref_ng_list):
    """Return (clipped_match, total) for one n-gram order.
    Clip each hyp n-gram count by the MAX count across the reference set
    (standard multi-reference modified precision, as in nltk corpus_bleu)."""
    if not hyp_ng:
        return 0, 0
    total = sum(hyp_ng.values())
    # max ref count per ngram
    max_ref = {}
    for r in ref_ng_list:
        for g, c in r.items():
            if c > max_ref.get(g, 0):
                max_ref[g] = c
    clipped = 0
    for g, c in hyp_ng.items():
        m = max_ref.get(g, 0)
        clipped += min(c, m)
    return clipped, total

def self_bleu_position(token_lists, max_n=4):
    """Average Self-BLEU over the 100 endings (each hyp vs the other 99 refs).
    BLEU-4, uniform weights 1/4, with epsilon smoothing on zero-precision orders
    (NLTK SmoothingFunction method1 style) and standard brevity penalty.
    Precompute per-ending n-gram counts and lengths once (O(N) memory)."""
    N = len(token_lists)
    if N < 2:
        return float("nan")
    # Precompute n-gram counters per ending for each order
    ng = {n: [ngram_counts(t, n) for t in token_lists] for n in range(1, max_n+1)}
    lengths = [len(t) for t in token_lists]
    weights = 1.0 / max_n
    bleus = []
    for i in range(N):
        # references = all but i
        log_p_sum = 0.0
        ok = True
        for n in range(1, max_n+1):
            hyp = ng[n][i]
            refs = [ng[n][j] for j in range(N) if j != i]
            clipped, total = modified_precision_counts(hyp, refs)
            if total == 0:
                ok = False
                break
            if clipped == 0:
                # method1 smoothing: avoid log(0)
                p = 1e-9
            else:
                p = clipped / total
            log_p_sum += weights * math.log(p)
        if not ok:
            continue
        # brevity penalty: closest ref length to hyp length
        hyp_len = lengths[i]
        ref_lens = [lengths[j] for j in range(N) if j != i]
        # closest ref length (ties -> shorter), as in BLEU
        closest = min(ref_lens, key=lambda rl: (abs(rl - hyp_len), rl))
        if hyp_len == 0:
            bp = 0.0
        elif hyp_len > closest:
            bp = 1.0
        else:
            bp = math.exp(1 - closest / hyp_len)
        bleus.append(bp * math.exp(log_p_sum))
    return float(np.mean(bleus)) if bleus else float("nan")

# ---- distinct-n ------------------------------------------------------------
def distinct_n(token_lists, n):
    total = 0
    uniq = set()
    for t in token_lists:
        grams = [tuple(t[i:i+n]) for i in range(len(t)-n+1)]
        total += len(grams)
        uniq.update(grams)
    return (len(uniq) / total) if total > 0 else float("nan")

# ---- exact-duplicate fraction (R2: are these a ranked list w/ near-dupes?) --
def exact_unique_frac(texts):
    """Fraction of the N endings that are EXACTLY unique (normalized whitespace).
    1.0 => zero exact duplicates. Direct evidence that the 100 endings are
    independent i.i.d. draws, not a beam/ranked list that repeats."""
    if not texts:
        return float("nan")
    norm = [re.sub(r"\s+", " ", t.strip().lower()) for t in texts]
    return len(set(norm)) / len(norm)

# ---- mean pairwise token Jaccard (alt to TF-IDF cosine) --------------------
def mean_pairwise_jaccard(token_lists, max_pairs=4950):
    """Mean Jaccard over the set-of-tokens of each ending, upper triangle.
    Lower => more diverse. (max_pairs guard = C(100,2)=4950 by default.)"""
    sets = [set(t) for t in token_lists]
    N = len(sets)
    if N < 2:
        return float("nan")
    vals = []
    for i in range(N):
        si = sets[i]
        for j in range(i + 1, N):
            sj = sets[j]
            u = len(si | sj)
            vals.append((len(si & sj) / u) if u else 0.0)
    return float(np.mean(vals)) if vals else float("nan")

# ---- mean pairwise TF-IDF cosine ------------------------------------------
def mean_pairwise_tfidf_cosine(texts):
    """Mean of the upper-triangle pairwise cosine over the 100 endings.
    Uses word 1-2 grams, sublinear tf. Lower => more diverse."""
    try:
        vec = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z0-9']+",
                              ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        X = vec.fit_transform(texts)
        if X.shape[1] == 0:
            return float("nan")
        S = cosine_similarity(X)
        n = S.shape[0]
        iu = np.triu_indices(n, k=1)
        vals = S[iu]
        return float(np.mean(vals)) if vals.size else float("nan")
    except Exception:
        return float("nan")

# ---- position sampling -----------------------------------------------------
def select_positions(positions):
    """positions: list of dicts with 'pct_story_revealed' and 'endings'.
    Keep pct>=PCT_MIN, sort by pct, then evenly subsample up to POS_CAP."""
    kept = [p for p in positions if p.get("pct_story_revealed", 0) >= PCT_MIN
            and isinstance(p.get("endings"), list) and len(p["endings"]) >= 2]
    kept.sort(key=lambda p: p["pct_story_revealed"])
    if len(kept) <= POS_CAP:
        return kept
    # evenly spaced indices including endpoints
    idx = np.linspace(0, len(kept) - 1, POS_CAP)
    idx = sorted(set(int(round(x)) for x in idx))
    return [kept[i] for i in idx]

# ---- per-story metric ------------------------------------------------------
def story_metrics(endings_path):
    d = json.load(open(endings_path))
    positions = d.get("positions", [])
    sel = select_positions(positions)
    if not sel:
        return None
    rows = []
    for p in sel:
        endings = p["endings"]
        if len(endings) < 2:
            continue
        toks = [tokenize(e) for e in endings]
        # guard: drop empty tokenizations for distinct/self-bleu robustness
        sb = self_bleu_position(toks, max_n=4)
        d1 = distinct_n(toks, 1)
        d2 = distinct_n(toks, 2)
        cos = mean_pairwise_tfidf_cosine(endings)
        jac = mean_pairwise_jaccard(toks)
        euf = exact_unique_frac(endings)
        rows.append((sb, d1, d2, cos, jac, euf, len(endings),
                    float(np.mean([len(t) for t in toks]))))
    if not rows:
        return None
    arr = np.array([r[:6] for r in rows], dtype=float)
    n_end = int(np.mean([r[6] for r in rows]))
    mean_wlen = float(np.mean([r[7] for r in rows]))
    return {
        "n_positions": len(rows),
        "self_bleu": float(np.nanmean(arr[:, 0])),
        "distinct_1": float(np.nanmean(arr[:, 1])),
        "distinct_2": float(np.nanmean(arr[:, 2])),
        "tfidf_cos": float(np.nanmean(arr[:, 3])),
        "jaccard": float(np.nanmean(arr[:, 4])),
        "exact_unique_frac": float(np.nanmean(arr[:, 5])),
        "n_endings": n_end,
        "mean_ending_wordlen": mean_wlen,
    }

# ---- gather story files per group -----------------------------------------
def model_story_files(slug):
    fs = sorted(glob.glob(os.path.join(MODEL_BASE, slug, "i1_p*_endings.json")))
    return [(os.path.basename(f).replace("_endings.json", ""), f) for f in fs]

def human_story_files(subdirs):
    out = []
    for sub in subdirs:
        fs = glob.glob(os.path.join(HUMAN_BASE, sub, "*_endings.json"))
        for f in fs:
            sid = os.path.basename(f).replace("_endings.json", "")
            out.append((sid, f))
    out.sort(key=lambda x: x[0])
    return out

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    all_groups = []
    for label, slug in MODEL_GROUPS:
        all_groups.append((label, "model", model_story_files(slug)))
    for label, subs in HUMAN_GROUPS:
        all_groups.append((label, "human", human_story_files(subs)))

    per_story_rows = []
    group_rows = []

    for label, kind, files in all_groups:
        files_sorted = sorted(files, key=lambda x: x[0])
        n_avail = len(files_sorted)
        chosen = files_sorted[:STORY_CAP]
        smetrics = []
        for sid, fpath in chosen:
            m = story_metrics(fpath)
            if m is None:
                continue
            m["group"] = label
            m["kind"] = kind
            m["story_id"] = sid
            smetrics.append(m)
            per_story_rows.append(m)
        if not smetrics:
            continue
        def gm(key):
            return float(np.mean([s[key] for s in smetrics]))
        def gsd(key):
            return float(np.std([s[key] for s in smetrics], ddof=1)) if len(smetrics) > 1 else 0.0
        group_rows.append({
            "group": label,
            "kind": kind,
            "n_stories_used": len(smetrics),
            "n_stories_available": n_avail,
            "mean_positions_per_story": float(np.mean([s["n_positions"] for s in smetrics])),
            "self_bleu": gm("self_bleu"),
            "self_bleu_sd": gsd("self_bleu"),
            "distinct_1": gm("distinct_1"),
            "distinct_1_sd": gsd("distinct_1"),
            "distinct_2": gm("distinct_2"),
            "distinct_2_sd": gsd("distinct_2"),
            "tfidf_cos": gm("tfidf_cos"),
            "tfidf_cos_sd": gsd("tfidf_cos"),
            "jaccard": gm("jaccard"),
            "jaccard_sd": gsd("jaccard"),
            "exact_unique_frac": gm("exact_unique_frac"),
            "exact_unique_frac_min": float(min(s["exact_unique_frac"] for s in smetrics)),
            "no_rate_mean": GROUP_NORATE.get(label, float("nan")),
            "mean_ending_wordlen": gm("mean_ending_wordlen"),
        })
        print(f"[{label}] used {len(smetrics)}/{n_avail} stories | "
              f"selfBLEU={gm('self_bleu'):.4f} d1={gm('distinct_1'):.4f} "
              f"d2={gm('distinct_2'):.4f} tfidf_cos={gm('tfidf_cos'):.4f} "
              f"jac={gm('jaccard'):.4f} exact_uniq={gm('exact_unique_frac'):.4f} "
              f"no_rate={GROUP_NORATE.get(label, float('nan')):.4f}", flush=True)

    # ---- correlations: group diversity vs group no-rate ----
    def corr(xs, ys):
        from scipy.stats import pearsonr, spearmanr
        xs = np.array(xs); ys = np.array(ys)
        mask = ~(np.isnan(xs) | np.isnan(ys))
        if mask.sum() < 3:
            return (float("nan"), float("nan"), float("nan"), float("nan"))
        pr, pp = pearsonr(xs[mask], ys[mask])
        sr, sp = spearmanr(xs[mask], ys[mask])
        return (float(pr), float(pp), float(sr), float(sp))

    nr = [g["no_rate_mean"] for g in group_rows]
    corrs = {}
    for metric in ["self_bleu", "distinct_1", "distinct_2", "tfidf_cos", "jaccard"]:
        vals = [g[metric] for g in group_rows]
        pr, pp, sr, sp = corr(vals, nr)
        corrs[metric] = {"pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp}

    # also: model-only (n=6) correlation — within-model comparison
    model_idx = [i for i, g in enumerate(group_rows) if g["kind"] == "model"]
    corrs_model = {}
    if len(model_idx) >= 3:
        nrm = [group_rows[i]["no_rate_mean"] for i in model_idx]
        for metric in ["self_bleu", "distinct_1", "distinct_2", "tfidf_cos", "jaccard"]:
            vals = [group_rows[i][metric] for i in model_idx]
            pr, pp, sr, sp = corr(vals, nrm)
            corrs_model[metric] = {"pearson_r": pr, "pearson_p": pp,
                                   "spearman_r": sr, "spearman_p": sp}

    # correlation across the 9 groups, and also human-only(3) for robustness
    human_idx = [i for i, g in enumerate(group_rows) if g["kind"] == "human"]
    corrs_human = {}
    if len(human_idx) >= 3:
        nrh = [group_rows[i]["no_rate_mean"] for i in human_idx]
        for metric in ["self_bleu", "distinct_1", "distinct_2", "tfidf_cos", "jaccard"]:
            vals = [group_rows[i][metric] for i in human_idx]
            pr, pp, sr, sp = corr(vals, nrh)
            corrs_human[metric] = {"pearson_r": pr, "pearson_p": pp,
                                  "spearman_r": sr, "spearman_p": sp}

    # ---- write CSV (group summary then per-story) ----
    out_csv = os.path.join(ROOT, "analyses", "05_diversity.csv")
    import csv
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["# ANALYSIS 5: lexical diversity of the 100 independently-generated endings"])
        w.writerow(["# endings generated INDEPENDENTLY at T=1.2 (i.i.d. samples), NOT a ranked list / NOT beam search"])
        w.writerow([f"# caps: <={STORY_CAP} stories/group (first N by sorted story_id), "
                    f"<={POS_CAP} positions/story (even subsample of pct_story_revealed>={PCT_MIN} set), "
                    f"{N_ENDINGS_EXPECT} endings/position"])
        w.writerow(["# self_bleu lower=more diverse; distinct_1/2 higher=more diverse; tfidf_cos & jaccard lower=more diverse"])
        w.writerow(["# exact_unique_frac = fraction of the 100 endings that are exactly unique (1.0 => no duplicates; R2 independence check)"])
        w.writerow([])
        w.writerow(["SECTION", "GROUP_SUMMARY"])
        cols = ["group", "kind", "n_stories_used", "n_stories_available",
                "mean_positions_per_story", "self_bleu", "self_bleu_sd",
                "distinct_1", "distinct_1_sd", "distinct_2", "distinct_2_sd",
                "tfidf_cos", "tfidf_cos_sd", "jaccard", "jaccard_sd",
                "exact_unique_frac", "exact_unique_frac_min",
                "no_rate_mean", "mean_ending_wordlen"]
        w.writerow(cols)
        for g in group_rows:
            w.writerow([g[c] for c in cols])
        w.writerow([])
        w.writerow(["SECTION", "CORRELATION_diversity_vs_no_rate (across 9 groups)"])
        w.writerow(["metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
        for m, c in corrs.items():
            w.writerow([m, c["pearson_r"], c["pearson_p"], c["spearman_r"], c["spearman_p"]])
        w.writerow([])
        w.writerow(["SECTION", "CORRELATION_diversity_vs_no_rate (human corpora only, n=3)"])
        w.writerow(["metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
        for m, c in corrs_human.items():
            w.writerow([m, c["pearson_r"], c["pearson_p"], c["spearman_r"], c["spearman_p"]])
        w.writerow([])
        w.writerow(["SECTION", "CORRELATION_diversity_vs_no_rate (model conditions only, n=6)"])
        w.writerow(["metric", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
        for m, c in corrs_model.items():
            w.writerow([m, c["pearson_r"], c["pearson_p"], c["spearman_r"], c["spearman_p"]])
        w.writerow([])
        w.writerow(["SECTION", "PER_STORY"])
        scols = ["group", "kind", "story_id", "n_positions", "n_endings",
                 "self_bleu", "distinct_1", "distinct_2", "tfidf_cos", "jaccard",
                 "exact_unique_frac", "mean_ending_wordlen"]
        w.writerow(scols)
        for s in per_story_rows:
            w.writerow([s[c] for c in scols])

    # also dump a json of the structured results for the .md step
    out_json = os.path.join(ROOT, "analyses", "05_diversity_results.json")
    json.dump({"groups": group_rows, "corr_all": corrs, "corr_human": corrs_human,
               "corr_model": corrs_model,
               "caps": {"story_cap": STORY_CAP, "pos_cap": POS_CAP, "pct_min": PCT_MIN,
                        "n_endings_expect": N_ENDINGS_EXPECT}},
              open(out_json, "w"), indent=2)
    print("WROTE", out_csv)
    print("WROTE", out_json)
    print("\n=== CORRELATIONS (diversity vs no_rate, 9 groups) ===")
    for m, c in corrs.items():
        print(f"  {m}: pearson r={c['pearson_r']:.3f} p={c['pearson_p']:.3f} | "
              f"spearman r={c['spearman_r']:.3f} p={c['spearman_p']:.3f}")
    # global exact-unique summary (R2)
    all_euf = [s["exact_unique_frac"] for s in per_story_rows]
    print(f"\n=== R2 INDEPENDENCE: exact_unique_frac over {len(all_euf)} stories: "
          f"mean={np.mean(all_euf):.4f} min={np.min(all_euf):.4f} "
          f"(1.0 => zero exact-duplicate endings) ===")

if __name__ == "__main__":
    main()
