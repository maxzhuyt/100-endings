#!/usr/bin/env python3
"""Judge anchoring / dynamic-range validation.
For 15 NYer stories, build 5 endings-list variants spanning min-to-max diversity, and ask
both judges to rate each. If both anchors land at 1 and 5 (and intermediate conditions are
monotonic), the 3.8 score on the actual data is real signal, not a default response."""
import json, glob, os, csv, re, sys, time, random
import numpy as np
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy import stats

NP="/project/jevans/maxzhuyt/narrative_project"
CWB="/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
OUT=f"{NP}/paper_revision/analyses/19_judge_anchoring.csv"

env={}
for line in open(f"{CWB}/.env"):
    line=line.strip()
    if line.startswith("#") or "=" not in line: continue
    k,_,v=line.partition("="); env[k.strip()]=v.strip()
client=OpenAI(api_key=env["JUDGE_API_KEY"], base_url="https://openrouter.ai/api/v1")
JUDGES={"deepseek":"deepseek/deepseek-chat-v3.1","gemini":"google/gemini-2.5-flash"}
N_STORIES=15
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
    return pos["endings"]

def judge_with(model, endings, attempt=0):
    text="\n".join(f"{i+1}. {e}" for i,e in enumerate(endings))
    try:
        r=client.chat.completions.create(model=model,
            messages=[{"role":"user","content":PROMPT.format(endings=text)}],
            max_tokens=8, temperature=0, timeout=90)
        out=r.choices[0].message.content.strip()
        m=re.search(r'[1-5]', out)
        return int(m.group(0)) if m else None
    except Exception as e:
        if attempt<2: time.sleep(3+attempt*2); return judge_with(model, endings, attempt+1)
        return None

# Collect NYer story files (same shuffle seed as elsewhere)
nyer_dirs=[f"{NP}/results_endings/nyer_expanded", f"{NP}/results_endings/nyer_83"]
files=[]
for d in nyer_dirs: files.extend(sorted(glob.glob(f"{d}/*_endings.json")))
random.Random(42).shuffle(files)

# Build a pool of "cross-story" endings: take the first ending from each of many DIFFERENT
# stories. Skip stories that will be used as the target story to avoid contamination.
cross_pool=[]
for fp in files:
    try:
        ends=endings_at_midpoint(fp)
        if ends: cross_pool.append((fp, ends[0]))
    except: pass
print(f"cross-story pool size: {len(cross_pool)}")

# Pick 15 target stories from the front of the shuffled list
target_files=files[:N_STORIES]
print(f"target stories: {len(target_files)}")

def build_variants(target_endings, target_fp, rng):
    """Return dict of condition_name → list-of-100-strings."""
    seed_ending = target_endings[0]
    # cross-story pool, excluding the target story's own first ending
    pool = [e for fp,e in cross_pool if fp != target_fp]
    rng.shuffle(pool)
    cross100 = pool[:100]
    # If pool too small, top up with target_endings (shouldn't happen with 261 stories)
    while len(cross100)<100: cross100.append(target_endings[len(cross100)%len(target_endings)])
    # Variants:
    variants = {
        "dup100"  : [seed_ending]*100,
        "dup75"   : [seed_ending]*75 + list(target_endings[1:26]),
        "dup50"   : [seed_ending]*50 + list(target_endings[1:51]),
        "original": list(target_endings[:100]),
        "cross"   : cross100,
    }
    # Sanity check each is exactly 100
    for k,v in variants.items(): assert len(v)==100, f"{k}: {len(v)}"
    return variants

CONDITIONS = ["dup100","dup75","dup50","original","cross"]

# resume support
done=set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        if r.get("deepseek","").strip() and r.get("gemini","").strip():
            done.add((r["story_id"], r["condition"]))
    print(f"resuming: {len(done)} (story, condition) already complete")

# Build task list
tasks=[]
for fp in target_files:
    sid=os.path.basename(fp)[:-len('_endings.json')]
    try:
        target_endings = endings_at_midpoint(fp)
    except Exception as e:
        print(f"skip {sid}: {e}"); continue
    rng = random.Random(hash(sid) & 0xFFFFFFFF)
    variants = build_variants(target_endings, fp, rng)
    for cond in CONDITIONS:
        if (sid, cond) in done: continue
        tasks.append((sid, cond, variants[cond]))
print(f"tasks: {len(tasks)} (story×condition) × 2 judges = {len(tasks)*2} calls")

fh=open(OUT,"a",newline="",buffering=1); w=csv.writer(fh)
if not done: w.writerow(["story_id","condition","deepseek","gemini"])

def work(item):
    sid, cond, ends = item
    s_ds = judge_with(JUDGES["deepseek"], ends)
    s_gm = judge_with(JUDGES["gemini"],   ends)
    return (sid, cond, s_ds, s_gm)

t0=time.time(); n=0
with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
    futs={ex.submit(work,t): t for t in tasks}
    for fut in as_completed(futs):
        sid, cond, s_ds, s_gm = fut.result()
        w.writerow([sid, cond,
                    s_ds if s_ds is not None else "",
                    s_gm if s_gm is not None else ""])
        n+=1
        if n%15==0:
            print(f"  [{n}/{len(tasks)}]  elapsed={int(time.time()-t0)}s", flush=True)
fh.close()
print(f"\nfinished in {int(time.time()-t0)}s")

# Summary
print("\n=== anchoring results (mean ± SD per condition × judge, n=15 stories) ===")
rows=[r for r in csv.DictReader(open(OUT)) if r["deepseek"] and r["gemini"]]
print(f"{'condition':12s} {'DeepSeek mean ± SD':22s} {'Gemini mean ± SD':22s}")
for cond in CONDITIONS:
    rs=[r for r in rows if r["condition"]==cond]
    ds=np.array([int(r["deepseek"]) for r in rs])
    gm=np.array([int(r["gemini"])   for r in rs])
    print(f"{cond:12s} {ds.mean():.2f} ± {ds.std(ddof=1):.2f}            "
          f"{gm.mean():.2f} ± {gm.std(ddof=1):.2f}            "
          f"(D range {ds.min()}-{ds.max()}, G range {gm.min()}-{gm.max()})")

# Monotonicity test: does each judge's mean score increase monotonically across conditions?
order_score = {"dup100":1,"dup75":2,"dup50":3,"original":4,"cross":5}
for judge in ["deepseek","gemini"]:
    pairs=[]
    for r in rows:
        pairs.append((order_score[r["condition"]], int(r[judge])))
    x=np.array([p[0] for p in pairs]); y=np.array([p[1] for p in pairs])
    rho,p=stats.spearmanr(x,y)
    print(f"\n{judge}: Spearman(condition-rank, judge-score) ρ={rho:+.3f} (p={p:.3g})")
