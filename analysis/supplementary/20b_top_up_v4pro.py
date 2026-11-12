#!/usr/bin/env python3
"""Top-up the 21 V4-Pro failures in 20_two_judges_v4pro_gemini.csv with higher max_tokens
to accommodate longer reasoning chains. Rewrites the CSV in place."""
import json, glob, os, csv, re, sys, time, random
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from collections import Counter

NP="/project/jevans/maxzhuyt/narrative_project"
CWB="/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
CSV_PATH=f"{NP}/paper_revision/analyses/20_two_judges_v4pro_gemini.csv"

def load_openrouter_key():
    key=None
    for line in open("/project/jevans/maxzhuyt/.env"):
        line=line.strip()
        if not line.startswith("OPENROUTER_API_KEY="): continue
        v=line.split("=",1)[1].strip().strip('"').strip("'")
        if v and "your_openrouter_key" not in v: key=v
    if not key: raise RuntimeError("No OPENROUTER_API_KEY")
    return key

client=OpenAI(api_key=load_openrouter_key(), base_url="https://openrouter.ai/api/v1")
MODEL="deepseek/deepseek-v4-pro"
TEMPERATURE=0.3
MAX_TOKENS=8000   # doubled from 4000

PROMPT="""Below are 100 alternative one-sentence continuations that were independently generated as candidate endings for a story. Rate how diverse these 100 endings are on a 1-5 scale based on the events, characters, and outcomes they portray:

1 = nearly identical (most endings describe essentially the same event)
2 = mostly repetitive with minor variations
3 = moderately diverse (multiple distinct threads but lots of overlap)
4 = highly diverse (mostly distinct events/outcomes)
5 = completely diverse (each ending a meaningfully different event)

Output ONLY a single digit 1-5 on the first line.

Endings:
{endings}"""

def endings_at_midpoint(path):
    d=json.load(open(path))
    pos=min(d["positions"], key=lambda x: abs(x.get("pct_story_revealed",50)-50))
    return pos["endings"]

def judge_v4(endings, attempt=0):
    text="\n".join(f"{i+1}. {e}" for i,e in enumerate(endings))
    try:
        r=client.chat.completions.create(model=MODEL,
            messages=[{"role":"user","content":PROMPT.format(endings=text)}],
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, timeout=300)
        out=r.choices[0].message.content or ""
        finish=r.choices[0].finish_reason
        matches = re.findall(r'\b([1-5])\b', out)
        return (int(matches[-1]) if matches else None), finish, r.usage.completion_tokens
    except Exception as e:
        if attempt<3: time.sleep(5+attempt*5); return judge_v4(endings, attempt+1)
        return None, f"err:{str(e)[:60]}", 0

# Build lookup: (group, story_id) → file path
GROUPS=[
 ("NewYorker", [f"{NP}/results_endings/nyer_expanded", f"{NP}/results_endings/nyer_83"]),
 ("Top10",     [f"{CWB}/results/endings/{slug}" for slug in
                ["anthropic__claude-sonnet-4.5","claude-opus-4-5-20251101","claude-opus-4-6","claude-sonnet-4-6",
                 "kimi-k2-thinking","moonshotai__kimi-k2","o3","openai__gpt-5.2",
                 "openrouter__horizon-alpha","openrouter__pony-alpha"]]),
]
sid2path={}
for group,dirs in GROUPS:
    for d in dirs:
        for fp in glob.glob(f"{d}/*_endings.json"):
            sid=os.path.basename(fp)[:-len('_endings.json')]
            sid2path[(group,sid)] = fp

# Find rows needing top-up
rows=list(csv.DictReader(open(CSV_PATH)))
todo=[(i, r) for i, r in enumerate(rows) if not r["v4pro"].strip()]
print(f"top-up: {len(todo)} V4-Pro scores missing")

def work(item):
    i, r = item
    fp = sid2path.get((r["group"], r["story_id"]))
    if not fp: return (i, None, "no_file_path", 0)
    try:
        ends = endings_at_midpoint(fp)
    except Exception as e:
        return (i, None, f"err:{str(e)[:60]}", 0)
    score, finish, tokens = judge_v4(ends)
    return (i, score, finish, tokens)

t0=time.time()
results={}
with ThreadPoolExecutor(max_workers=6) as ex:
    futs={ex.submit(work, item): item for item in todo}
    for fut in as_completed(futs):
        i, score, finish, tokens = fut.result()
        sid=rows[i]["story_id"]; grp=rows[i]["group"]
        status = f"score={score}" if score is not None else f"FAIL ({finish})"
        print(f"  {grp}/{sid}: {status}  tokens={tokens}")
        results[i] = score

# Update rows in place
for i, score in results.items():
    if score is not None:
        rows[i]["v4pro"] = str(score)

# Rewrite CSV
with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["group","story_id","pct","v4pro","gemini","jaccard"])
    w.writeheader()
    for r in rows: w.writerow(r)
print(f"\nrewrote {CSV_PATH} ({len(rows)} rows total) in {int(time.time()-t0)}s")

# Re-print summary
print("\n=== updated summary ===")
def weighted_kappa(y1, y2, weights="linear", levels=None):
    y1=np.asarray(y1); y2=np.asarray(y2)
    if levels is None: levels=sorted(set(y1)|set(y2))
    k=len(levels); idx={v:i for i,v in enumerate(levels)}
    O=np.zeros((k,k))
    for a,b in zip(y1,y2): O[idx[a],idx[b]]+=1
    n=O.sum(); P=O/n
    row=P.sum(axis=1); col=P.sum(axis=0)
    E=np.outer(row,col)
    if weights=="linear": W=np.array([[abs(i-j)/(k-1) for j in range(k)] for i in range(k)])
    else: W=np.array([[((i-j)/(k-1))**2 for j in range(k)] for i in range(k)])
    num=(W*P).sum(); den=(W*E).sum()
    return 1 - num/den if den else float("nan")

rows=[r for r in csv.DictReader(open(CSV_PATH)) if r["v4pro"] and r["gemini"]]
for g in ["NewYorker","Top10"]:
    rs=[r for r in rows if r["group"]==g]
    v4=np.array([int(r["v4pro"]) for r in rs])
    gm=np.array([int(r["gemini"]) for r in rs])
    jac=np.array([float(r["jaccard"]) for r in rs if r["jaccard"]])
    ctr_v=Counter(v4.tolist()); ctr_g=Counter(gm.tolist())
    print(f"\n{g}: n={len(rs)}")
    print(f"  V4-Pro          : mean={v4.mean():.2f} ± {v4.std(ddof=1):.2f}  dist={dict(sorted(ctr_v.items()))}")
    print(f"  Gemini 2.5 Flash: mean={gm.mean():.2f} ± {gm.std(ddof=1):.2f}  dist={dict(sorted(ctr_g.items()))}")
    rho,p=stats.spearmanr(v4,gm)
    kappa=weighted_kappa(v4,gm,weights="linear", levels=[1,2,3,4,5])
    print(f"  inter-judge     : Spearman ρ={rho:+.3f} (p={p:.3g}); weighted κ (linear)={kappa:.3f}")
    rho_v4,p_v4=stats.spearmanr(v4, jac); rho_gm,p_gm=stats.spearmanr(gm, jac)
    print(f"  V4-Pro↔Jaccard  : ρ={rho_v4:+.3f} (p={p_v4:.3g})")
    print(f"  Gemini ↔Jaccard : ρ={rho_gm:+.3f} (p={p_gm:.3g})")

ny_v4=np.array([int(r["v4pro"]) for r in rows if r["group"]=="NewYorker"])
t10_v4=np.array([int(r["v4pro"]) for r in rows if r["group"]=="Top10"])
ny_gm=np.array([int(r["gemini"]) for r in rows if r["group"]=="NewYorker"])
t10_gm=np.array([int(r["gemini"]) for r in rows if r["group"]=="Top10"])
print("\n=== cross-group ===")
for label, a, b in [("V4-Pro", ny_v4, t10_v4), ("Gemini", ny_gm, t10_gm)]:
    t,p_t=stats.ttest_ind(a,b,equal_var=False)
    u,p_u=stats.mannwhitneyu(a,b,alternative="two-sided")
    print(f"  {label}: Welch t={t:+.2f} p={p_t:.3g}  MW U={u:.0f} p={p_u:.3g}")
