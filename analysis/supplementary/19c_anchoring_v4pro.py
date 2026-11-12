#!/usr/bin/env python3
"""Anchoring test with DeepSeek V4-Pro at T=0.3. Same 15 stories × 5 conditions as
19_judge_anchoring.py (V3.1) and 19b_anchoring_r1.py (R1), for direct comparison."""
import json, glob, os, csv, re, sys, time, random
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats
from collections import Counter

NP="/project/jevans/maxzhuyt/narrative_project"
CWB="/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
OUT=f"{NP}/paper_revision/analyses/19c_anchoring_v4pro.csv"

def load_openrouter_key():
    # Read the fiction-writing project's OpenRouter key from ~maxzhuyt/.env.
    # The file has two OPENROUTER_API_KEY lines (one placeholder, one real); take
    # the last non-placeholder definition.
    key=None
    for line in open("/project/jevans/maxzhuyt/.env"):
        line=line.strip()
        if not line.startswith("OPENROUTER_API_KEY="): continue
        v=line.split("=",1)[1].strip().strip('"').strip("'")
        if v and "your_openrouter_key" not in v: key=v
    if not key: raise RuntimeError("No real OPENROUTER_API_KEY found in /project/jevans/maxzhuyt/.env")
    return key
client=OpenAI(api_key=load_openrouter_key(), base_url="https://openrouter.ai/api/v1")

JUDGE_MODEL = "deepseek/deepseek-v4-pro"
TEMPERATURE = 0.3
MAX_TOKENS  = 4000
N_STORIES   = 15
N_WORKERS   = 8

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
        r=client.chat.completions.create(model=JUDGE_MODEL,
            messages=[{"role":"user","content":PROMPT.format(endings=text)}],
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE, timeout=180)
        out=r.choices[0].message.content or ""
        matches = re.findall(r'\b([1-5])\b', out)
        return int(matches[-1]) if matches else None, out, r.usage.completion_tokens
    except Exception as e:
        if attempt<2: time.sleep(5+attempt*3); return judge_v4(endings, attempt+1)
        return None, str(e), 0

nyer_dirs=[f"{NP}/results_endings/nyer_expanded", f"{NP}/results_endings/nyer_83"]
files=[]
for d in nyer_dirs: files.extend(sorted(glob.glob(f"{d}/*_endings.json")))
random.Random(42).shuffle(files)
cross_pool=[]
for fp in files:
    try:
        ends=endings_at_midpoint(fp)
        if ends: cross_pool.append((fp, ends[0]))
    except: pass
target_files=files[:N_STORIES]
print(f"V4-Pro anchoring | T={TEMPERATURE} | n_stories={N_STORIES}")
print(f"cross-story pool: {len(cross_pool)}")

def build_variants(target_endings, target_fp, rng):
    seed = target_endings[0]
    pool=[e for fp,e in cross_pool if fp != target_fp]; rng.shuffle(pool)
    cross100 = pool[:100]
    while len(cross100)<100: cross100.append(target_endings[len(cross100)%len(target_endings)])
    return {
        "dup100"  : [seed]*100,
        "dup75"   : [seed]*75 + list(target_endings[1:26]),
        "dup50"   : [seed]*50 + list(target_endings[1:51]),
        "original": list(target_endings[:100]),
        "cross"   : cross100,
    }

CONDITIONS=["dup100","dup75","dup50","original","cross"]
done=set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        if r.get("score","").strip(): done.add((r["story_id"], r["condition"]))

tasks=[]
for fp in target_files:
    sid=os.path.basename(fp)[:-len('_endings.json')]
    try: target_endings = endings_at_midpoint(fp)
    except Exception as e: print(f"skip {sid}: {e}"); continue
    rng=random.Random(hash(sid) & 0xFFFFFFFF)
    variants=build_variants(target_endings, fp, rng)
    for cond in CONDITIONS:
        if (sid,cond) in done: continue
        tasks.append((sid, cond, variants[cond]))
print(f"tasks: {len(tasks)}")

fh=open(OUT,"a",newline="",buffering=1); w=csv.writer(fh)
if not done: w.writerow(["story_id","condition","score","completion_tokens","raw_tail"])

def work(item):
    sid, cond, ends = item
    score, raw, tokens = judge_v4(ends)
    return (sid, cond, score, raw, tokens)

t0=time.time(); n=0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs={ex.submit(work,t): t for t in tasks}
    for fut in as_completed(futs):
        sid,cond,score,raw,tokens=fut.result()
        w.writerow([sid, cond,
                    score if score is not None else "",
                    tokens,
                    (raw or "")[-200:].replace("\n"," ")])
        n+=1
        if n%10==0:
            print(f"  [{n}/{len(tasks)}] elapsed={int(time.time()-t0)}s", flush=True)
fh.close()
print(f"\nfinished in {int(time.time()-t0)}s")

print("\n=== DeepSeek V4-Pro (T=0.3) anchoring ===")
rows=[r for r in csv.DictReader(open(OUT)) if r["score"]]
print(f"{'condition':12s} {'mean ± SD':14s} {'range':8s} distribution")
for cond in CONDITIONS:
    rs=[r for r in rows if r["condition"]==cond]
    vals=np.array([int(r["score"]) for r in rs])
    ctr=Counter(int(r["score"]) for r in rs)
    dist=" ".join(f"{s}:{ctr[s]}" for s in [1,2,3,4,5] if ctr[s])
    print(f"{cond:12s} {vals.mean():.2f} ± {vals.std(ddof=1):.2f}  "
          f"{vals.min()}-{vals.max()}     {dist}")

order_score={"dup100":1,"dup75":2,"dup50":3,"original":4,"cross":5}
pairs=[(order_score[r["condition"]], int(r["score"])) for r in rows]
x=np.array([p[0] for p in pairs]); y=np.array([p[1] for p in pairs])
rho,p=stats.spearmanr(x,y)
print(f"\nV4-Pro: monotonicity Spearman ρ={rho:+.3f} (p={p:.3g})")

tok=[int(r["completion_tokens"]) for r in rows if r.get("completion_tokens","").isdigit()]
if tok:
    print(f"completion tokens: min={min(tok)} median={sorted(tok)[len(tok)//2]} max={max(tok)} mean={int(np.mean(tok))}")
