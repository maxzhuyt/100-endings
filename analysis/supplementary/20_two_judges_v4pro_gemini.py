#!/usr/bin/env python3
"""Two-judge LLM diversity rating at n=200 per group with V4-Pro + Gemini 2.5 Flash,
both at T=0.3. V4-Pro replaces V3.1: better dynamic range, uses full 1-5 scale,
visible reasoning. Reports per-group means, Spearman between judges, and weighted
Cohen's kappa."""
import json, glob, os, csv, re, sys, time, random
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from collections import Counter

NP="/project/jevans/maxzhuyt/narrative_project"
CWB="/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
OUT=f"{NP}/paper_revision/analyses/20_two_judges_v4pro_gemini.csv"

def load_openrouter_key():
    key=None
    for line in open("/project/jevans/maxzhuyt/.env"):
        line=line.strip()
        if not line.startswith("OPENROUTER_API_KEY="): continue
        v=line.split("=",1)[1].strip().strip('"').strip("'")
        if v and "your_openrouter_key" not in v: key=v
    if not key: raise RuntimeError("No real OPENROUTER_API_KEY in /project/jevans/maxzhuyt/.env")
    return key

client=OpenAI(api_key=load_openrouter_key(), base_url="https://openrouter.ai/api/v1")
JUDGES={"v4pro":"deepseek/deepseek-v4-pro","gemini":"google/gemini-2.5-flash"}
TEMPERATURE=0.3
SAMPLE=200
N_WORKERS=8

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
    return pos["endings"], pos["pct_story_revealed"]

def judge_with(model, endings, max_tokens, attempt=0):
    text="\n".join(f"{i+1}. {e}" for i,e in enumerate(endings))
    try:
        r=client.chat.completions.create(model=model,
            messages=[{"role":"user","content":PROMPT.format(endings=text)}],
            max_tokens=max_tokens, temperature=TEMPERATURE, timeout=180)
        out=r.choices[0].message.content or ""
        # Pick the LAST 1-5 digit (V4-Pro may emit reasoning before the final answer)
        matches = re.findall(r'\b([1-5])\b', out)
        return int(matches[-1]) if matches else None
    except Exception as e:
        if attempt<2: time.sleep(5+attempt*3); return judge_with(model, endings, max_tokens, attempt+1)
        return None

def lexical_jaccard(endings, n_pairs=300, rng=random.Random(7)):
    toks=[set(re.findall(r"[a-z']+", e.lower())) for e in endings]
    sets=[t for t in toks if t]
    if len(sets)<2: return None
    js=[]
    for _ in range(n_pairs):
        a,b=rng.sample(sets,2); u=len(a|b)
        if u: js.append(len(a&b)/u)
    return float(np.mean(js))

done=set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        if r.get("v4pro","").strip() and r.get("gemini","").strip():
            done.add((r["group"], r["story_id"]))
    print(f"resuming: {len(done)} already complete")

GROUPS=[
 ("NewYorker", [f"{NP}/results_endings/nyer_expanded", f"{NP}/results_endings/nyer_83"]),
 ("Top10",     [f"{CWB}/results/endings/{slug}" for slug in
                ["anthropic__claude-sonnet-4.5","claude-opus-4-5-20251101","claude-opus-4-6","claude-sonnet-4-6",
                 "kimi-k2-thinking","moonshotai__kimi-k2","o3","openai__gpt-5.2",
                 "openrouter__horizon-alpha","openrouter__pony-alpha"]]),
]

tasks=[]
for group,dirs in GROUPS:
    files=[]
    for d in dirs: files.extend(sorted(glob.glob(f"{d}/*_endings.json")))
    random.Random(42).shuffle(files)
    files=files[:SAMPLE]
    for fp in files:
        sid=os.path.basename(fp)[:-len('_endings.json')]
        if (group,sid) in done: continue
        tasks.append((group,sid,fp))
print(f"total: {len(tasks)} stories × 2 judges = {len(tasks)*2} calls")

fh=open(OUT,"a",newline="",buffering=1); w=csv.writer(fh)
if not done: w.writerow(["group","story_id","pct","v4pro","gemini","jaccard"])

def work(item):
    group,sid,fp=item
    try:
        ends,pct=endings_at_midpoint(fp)
        s_v4 = judge_with(JUDGES["v4pro"],  ends, max_tokens=4000)
        s_gm = judge_with(JUDGES["gemini"], ends, max_tokens=8)
        jac  = lexical_jaccard(ends)
    except Exception as e:
        return (group,sid,None,None,None,None,str(e))
    return (group,sid,pct,s_v4,s_gm,jac,None)

t0=time.time(); n=0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs={ex.submit(work,t): t for t in tasks}
    for fut in as_completed(futs):
        group,sid,pct,s_v4,s_gm,jac,err=fut.result()
        if err: print(f"  ERR {group}/{sid}: {err}", file=sys.stderr)
        w.writerow([group,sid,pct if pct is not None else "",
                    s_v4 if s_v4 is not None else "",
                    s_gm if s_gm is not None else "",
                    f"{jac:.6f}" if jac is not None else ""])
        n+=1
        if n%20==0:
            elapsed=time.time()-t0; rate=n/elapsed
            print(f"  [{n}/{len(tasks)}] rate={rate:.1f}/s eta={int((len(tasks)-n)/rate)}s", flush=True)
fh.close()
print(f"\nfinished in {time.time()-t0:.0f}s")

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

print("\n=== summary ===")
rows=[r for r in csv.DictReader(open(OUT)) if r["v4pro"] and r["gemini"]]
for g in ["NewYorker","Top10"]:
    rs=[r for r in rows if r["group"]==g]
    v4=np.array([int(r["v4pro"]) for r in rs])
    gm=np.array([int(r["gemini"]) for r in rs])
    jac=np.array([float(r["jaccard"]) for r in rs if r["jaccard"]])
    ctr_v=Counter(v4.tolist()); ctr_g=Counter(gm.tolist())
    print(f"\n{g}: n={len(rs)}")
    print(f"  V4-Pro          : mean={v4.mean():.2f} ± {v4.std(ddof=1):.2f}  "
          f"dist={dict(sorted(ctr_v.items()))}")
    print(f"  Gemini 2.5 Flash: mean={gm.mean():.2f} ± {gm.std(ddof=1):.2f}  "
          f"dist={dict(sorted(ctr_g.items()))}")
    rho_judges,p_judges=stats.spearmanr(v4,gm)
    kappa=weighted_kappa(v4,gm,weights="linear", levels=[1,2,3,4,5])
    print(f"  inter-judge     : Spearman ρ={rho_judges:+.3f} (p={p_judges:.3g}); "
          f"weighted κ (linear)={kappa:.3f}")
    rho_v4,p_v4=stats.spearmanr(v4, jac); rho_gm,p_gm=stats.spearmanr(gm, jac)
    print(f"  V4-Pro↔Jaccard  : ρ={rho_v4:+.3f} (p={p_v4:.3g})")
    print(f"  Gemini ↔Jaccard : ρ={rho_gm:+.3f} (p={p_gm:.3g})")

print("\n=== cross-group ===")
ny_v4=np.array([int(r["v4pro"]) for r in rows if r["group"]=="NewYorker"])
t10_v4=np.array([int(r["v4pro"]) for r in rows if r["group"]=="Top10"])
ny_gm=np.array([int(r["gemini"]) for r in rows if r["group"]=="NewYorker"])
t10_gm=np.array([int(r["gemini"]) for r in rows if r["group"]=="Top10"])
for label, a, b in [("V4-Pro", ny_v4, t10_v4), ("Gemini", ny_gm, t10_gm)]:
    t,p_t=stats.ttest_ind(a,b,equal_var=False)
    u,p_u=stats.mannwhitneyu(a,b,alternative="two-sided")
    print(f"  {label}: Welch t={t:+.2f} p={p_t:.3g}  MW U={u:.0f} p={p_u:.3g}")
