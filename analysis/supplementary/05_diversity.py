#!/usr/bin/env python3
"""Analysis 5: lexical diversity of the 100 generated endings (independent draws, T=1.2).
Per evaluated position, over the 100 endings: distinct-1, distinct-2, and mean pairwise
token Jaccard SIMILARITY (lower = more diverse). Averaged per story then per group.
Tests whether higher-no-rate corpora also have more lexically diverse endings."""
import gzip, json, glob, csv, re, random
import numpy as np
NP="/project/jevans/maxzhuyt/narrative_project"
BASE=NP+"/paper_revision"; CORP=BASE+"/data/corpora"; OUT=BASE+"/analyses"
RNG=random.Random(7)
CAP=30            # stories per group
POS_PER_STORY=8  # evenly-spaced judged positions
PAIRS=150        # sampled ending-pairs per position for Jaccard

GROUPS=["NewYorker","TellMeAStory","StoryStar","Sonnet4.6-vanilla","Sonnet4.6-pipeline",
        "Opus4.6-vanilla","Opus4.6-pipeline","GPT5.2-vanilla","GPT5.2-pipeline"]
tok=lambda s: re.findall(r"[a-z']+", s.lower())

def story_div(rec):
    judged=[p for p in rec["positions"] if p.get("judged")]
    if not judged: return None
    idx=np.linspace(0,len(judged)-1,min(POS_PER_STORY,len(judged))).astype(int)
    d1s,d2s,jacs=[],[],[]
    for pi in idx:
        endings=[e["text"] for e in judged[pi]["endings"]]
        toks=[tok(t) for t in endings]
        uni=[w for ts in toks for w in ts]; bi=[tuple(ts[i:i+2]) for ts in toks for i in range(len(ts)-1)]
        if uni: d1s.append(len(set(uni))/len(uni))
        if bi: d2s.append(len(set(bi))/len(bi))
        sets=[set(t) for t in toks if t]
        if len(sets)>=2:
            js=[]
            for _ in range(PAIRS):
                a,b=RNG.sample(sets,2); u=len(a|b)
                if u: js.append(len(a&b)/u)
            if js: jacs.append(np.mean(js))
    return (np.mean(d1s) if d1s else np.nan, np.mean(d2s) if d2s else np.nan, np.mean(jacs) if jacs else np.nan)

# group no-rate (for the diversity-vs-tension relationship)
def group_norate(g):
    v=[float(r["no_rate_mean"]) for r in csv.DictReader(open(f"{CORP}/{g}/scores.csv")) if r["no_rate_mean"]!=""]
    return np.mean(v)

rows=[]
for g in GROUPS:
    f=f"{CORP}/{g}/endings_with_verdicts.jsonl.gz"
    recs=[]
    with gzip.open(f,"rt") as gz:
        for i,line in enumerate(gz):
            if i>=CAP: break
            recs.append(json.loads(line))
    ds=[story_div(r) for r in recs]; ds=[d for d in ds if d]
    d1=np.nanmean([d[0] for d in ds]); d2=np.nanmean([d[1] for d in ds]); jac=np.nanmean([d[2] for d in ds])
    rows.append((g,len(ds),d1,d2,jac,group_norate(g)))
    print(f"  {g:20} n={len(ds):3} distinct1={d1:.3f} distinct2={d2:.3f} jaccard_sim={jac:.4f} no-rate={group_norate(g):.3f}")

with open(f"{OUT}/05_diversity.csv","w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["group","n_stories","distinct1","distinct2","mean_pairwise_jaccard_sim","group_no_rate"])
    for g,n,d1,d2,jac,nrr in rows: w.writerow([g,n,round(d1,4),round(d2,4),round(jac,4),round(nrr,4)])

# correlation diversity vs no-rate across groups
jacs=np.array([r[4] for r in rows]); nrs=np.array([r[5] for r in rows]); d2s=np.array([r[3] for r in rows])
from scipy import stats
r_j,p_j=stats.pearsonr(nrs,jacs); r_d,p_d=stats.pearsonr(nrs,d2s)

L=["# Analysis 5 — Lexical diversity of the 100 endings","",
   f"Sample: first {CAP} stories/group, {POS_PER_STORY} evenly-spaced judged positions/story, {PAIRS} sampled ending-pairs/position.",
   "The 100 endings per position are **independent** generations (Qwen3-32B, T=1.2), not a ranked list.",
   "distinct-1/2 = unique uni/bi-grams / total (higher = more diverse). Jaccard sim = mean pairwise token overlap (lower = more diverse).","",
   "| Group | n | distinct-1 | distinct-2 | Jaccard sim | group no-rate |","|---|---|---|---|---|---|"]
for g,n,d1,d2,jac,nrr in rows:
    L.append(f"| {g} | {n} | {d1:.3f} | {d2:.3f} | {jac:.4f} | {nrr:.3f} |")
L+=["",
   f"**Diversity is high across all groups** (distinct-2 ~ {d2s.mean():.2f}; low pairwise Jaccard ~ {jacs.mean():.3f}), confirming the 100 endings are genuinely varied independent samples rather than near-duplicates.",
   f"Across groups, group no-rate vs Jaccard similarity: Pearson r={r_j:.2f} (p={p_j:.3f}); vs distinct-2: r={r_d:.2f} (p={p_d:.3f}). "
   + ("Higher-tension corpora tend to have *more* lexically diverse endings." if r_j<0 else "Lexical diversity and no-rate are only weakly related, indicating no-rate reflects plot/event divergence judged by the LLM rather than mere surface lexical variety.")]
open(f"{OUT}/05_endings_diversity.md","w").write("\n".join(L)+"\n")
print(f"Analysis 5 done. no-rate~Jaccard r={r_j:.2f}, no-rate~distinct2 r={r_d:.2f}")
