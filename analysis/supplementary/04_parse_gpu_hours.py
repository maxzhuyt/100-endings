#!/usr/bin/env python3
"""
ANALYSIS 4 - Computational cost / GPU-hours of the 100-Endings narrative-tension metric.

Two GPU stages (1 NVIDIA H200 per slurm job  =>  GPU-hours == wall-clock hours):
  GENERATION : Qwen3-32B (T=1.2)  -> 100 candidate endings at each evaluated story position
  JUDGE      : Qwen3-8B  (T=0)     -> binary satisfying/not over the 100 endings/position

WALL-CLOCK per job. We parse every timestamp source and reconcile:
  - shell header/footer "=== ... at <DOW Mon DD HH:MM:SS TZ YYYY> ===" / "=== Finished|Done at ... ==="
  - vLLM   "INFO MM-DD HH:MM:SS"          (covers init only on GENERATION logs)
  - logger "YYYY-MM-DD HH:MM:SS,mmm INFO" (covers full run on JUDGE logs)
GENERATION logs print progress with relative durations only ("Done in Ns", batch "(Ns)") and no
absolute clock after init, so their last *timestamp* understates wall time. For those we take the
END BOUND = file mtime (validated: for jobs WITH a shell footer, mtime == 'Finished at' to the second).
We also reconstruct GPU-BUSY seconds = sum of per-story "Done in Ns" (or single-batch "(Ns)") as an
independent cross-check of the wall-clock.

Buckets:
  canonical_rep1        the primary metric run feeding the published tables (models top-10 + human corpora)
  superseded_32b_judge  early human-judge attempts that used Qwen3-32B and were DISCARDED & rerun on 8B
  pipeline_condition    narrative_* generator-pipeline conditions (combined gen+judge eval jobs)
  replicate_rep2_3      robustness replicate generations (rep2/rep3)
  failed_or_aborted     crashed engine / OOM / mis-config (no usable output)
  empty_run             ran to completion but 0 new stories (idempotent re-submission)
  api_eqbench_nogpu     EQ-Bench API runs that allocated a node but did no local GPU inference
WritingPrompts (wp_human) is flagged is_wp and EXCLUDED from all headline totals (reported separately).

Outputs:
  data/04_gpu_hours.json          full per-(job,stage) records
  data/04_gpu_hours_agg.json      aggregate GPU-hours + per-unit derivations
  analyses/04_gpu_hours.csv       per-(job,stage) rows + TOTALS block
"""
import os, re, glob, json, csv, datetime, statistics

BASE = "/project/jevans/maxzhuyt/narrative_project/paper_revision"
CWB  = "/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
NP   = "/project/jevans/maxzhuyt/narrative_project"
os.makedirs(f"{BASE}/data", exist_ok=True); os.makedirs(f"{BASE}/analyses", exist_ok=True)

LOG_DIRS = {
    "models_gen":   sorted(glob.glob(f"{CWB}/results/endings/slurm_*.out")),
    "models_judge": sorted(glob.glob(f"{CWB}/results/judge/slurm_*.out")),
    "human":        sorted(glob.glob(f"{NP}/results_endings/logs/*.out")),
    "cwb_logs":     sorted(glob.glob(f"{CWB}/results/logs/*.out")),
}
YEAR = 2026
MONTHS = {m:i for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],1)}
RE_VLLM2  = re.compile(r"(?:INFO|WARNING|ERROR|DEBUG) (\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)")
RE_LOGGER = re.compile(r"^(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d),(\d{3})")
RE_SHELL  = re.compile(r"===.*?\bat\s+\w{3}\s+(\w{3})\s+(\d{1,2})\s+(\d\d):(\d\d):(\d\d)\s+\w+\s+(\d{4})")
RE_FINISH = re.compile(r"===\s*(?:Finished|Done|Both done).*?\bat\s+\w{3}\s+(\w{3})\s+(\d{1,2})\s+(\d\d):(\d\d):(\d\d)\s+\w+\s+(\d{4})")

def shell_dt(m):
    return datetime.datetime(int(m.group(6)), MONTHS[m.group(1)], int(m.group(2)),
                             int(m.group(3)), int(m.group(4)), int(m.group(5)))
def vllm_dt(m):
    mo,da,hh,mm,ss=(int(x) for x in m.groups()); return datetime.datetime(YEAR,mo,da,hh,mm,ss)
def logger_dt(m):
    yr,mo,da,hh,mm,ss,ms=(int(x) for x in m.groups()); return datetime.datetime(yr,mo,da,hh,mm,ss,ms*1000)

def timestamps(lines):
    ts=[]
    for ln in lines:
        m=RE_LOGGER.search(ln)
        if m: ts.append(logger_dt(m)); continue
        m=RE_SHELL.search(ln)
        if m: ts.append(shell_dt(m))
        m=RE_VLLM2.search(ln)
        if m: ts.append(vllm_dt(m))
    return sorted(ts)

def first_int(lines, pat):
    rx=re.compile(pat)
    for ln in lines:
        m=rx.search(ln)
        if m: return int(m.group(1).replace(",",""))
    return None
def last_int(lines, pat):
    rx=re.compile(pat); v=None
    for ln in lines:
        m=rx.search(ln)
        if m: v=int(m.group(1).replace(",",""))
    return v
def sizes(lines): return sorted({int(x) for x in re.findall(r"models/Qwen3-(\d+)B","\n".join(lines))})
def gpu_busy_secs(lines):
    txt="".join(lines)
    dones=[int(x) for x in re.findall(r"Done in (\d+)s", txt)]
    if dones: return sum(dones)
    # CWB-style per-story "(NN pos, Ns)"
    perstory=[int(x) for x in re.findall(r"\(\d+\s*pos,\s*(\d+)s\)", txt)]
    if perstory: return sum(perstory)
    return None
def hhmm(h):
    if h is None: return ""
    s=int(round(h*3600)); return f"{s//3600:d}:{(s%3600)//60:02d}:{s%60:02d}"

def segs(lines):
    idx=[]
    for i,ln in enumerate(lines):
        if "=== Generating endings" in ln: idx.append((i,"gen"))
        elif "=== Judging endings" in ln:  idx.append((i,"judge"))
    if not idx: return None
    out=[]
    for k,(i,lab) in enumerate(idx):
        j=idx[k+1][0] if k+1<len(idx) else len(lines)
        out.append((lab,i,j))
    return out

def parse(path, group):
    fn=os.path.basename(path); jid=re.search(r"(\d{6,})",fn); jid=jid.group(1) if jid else fn
    mtime=datetime.datetime.fromtimestamp(os.path.getmtime(path))
    with open(path,errors="replace") as f: lines=f.readlines()
    header=lines[0].strip() if lines else ""
    out=[]
    sg=segs(lines)
    if sg:  # combined eval_narrative_* (gen then judge, sequential)
        for lab,i,j in sg:
            seg=lines[i:j]; ts=timestamps(seg)
            if not ts: continue
            # phase end: for gen use start of judge marker time (next seg start) -> approximated by seg's last ts;
            # but gen seg lacks late timestamps -> use next seg first shell 'at' if available
            t0=ts[0]; t1=ts[-1]
            # if this is gen and a following judge seg exists, set gen end = judge start shell time
            if lab=="gen":
                # find '=== Judging endings ... at' shell time after this seg
                jm=None
                for ln in lines[j:j+3]:
                    m=RE_SHELL.search(ln)
                    if m and "Judging" in ln: jm=shell_dt(m); break
                if jm: t1=jm
            wall=(t1-t0).total_seconds()/3600.0
            out.append(dict(jobid=jid,file=fn,group=group,
                            stage="GENERATION" if lab=="gen" else "JUDGE",
                            qwen_b=32 if lab=="gen" else 8, phase=lab, header=header,
                            t_start=t0.isoformat(sep=" "), t_end=t1.isoformat(sep=" "),
                            end_source="phase_marker", wall_hours=round(wall,4),
                            gpu_busy_secs=gpu_busy_secs(seg),
                            n_stories_total=first_int(seg,r"Total stories:\s*(\d+)"),
                            n_stories_done=last_int(seg,r"DONE\.\s*(\d+)\s*stories"),
                            judge_calls=last_int(seg,r"judge calls:\s*([\d,]+)"),
                            finished=any("=== Done at" in x for x in seg), n_lines=len(seg)))
        return out

    ts=timestamps(lines)
    has_footer=any(RE_FINISH.search(ln) for ln in lines)
    if not ts:
        out.append(dict(jobid=jid,file=fn,group=group,stage="UNPARSEABLE",qwen_b=None,phase=None,
                        header=header,t_start=None,t_end=None,end_source=None,wall_hours=None,
                        gpu_busy_secs=None,n_stories_total=None,n_stories_done=None,judge_calls=None,
                        finished=False,n_lines=len(lines)))
        return out
    t0=ts[0]
    # END BOUND: shell footer if present else max(last ts, mtime) for GEN logs (timestamps truncate)
    foot=None
    for ln in lines:
        m=RE_FINISH.search(ln)
        if m: foot=shell_dt(m)
    sz=sizes(lines)
    is_gen = (sz==[32]) and not any(("judge calls" in ln) or ("ending_judge" in ln) for ln in lines) \
             or any("Generation complete" in ln for ln in lines) \
             or any(re.search(r"Evaluating .*positions", ln) for ln in lines)
    if foot is not None:
        t1=foot; esrc="shell_footer"
    elif is_gen:
        t1=max(ts[-1], mtime); esrc="file_mtime"
    else:
        t1=ts[-1]; esrc="last_timestamp"
    wall=(t1-t0).total_seconds()/3600.0
    # stage
    if is_gen: stage,qwen="GENERATION",32
    elif sz==[8]: stage,qwen="JUDGE",8
    elif sz==[32]: stage,qwen="JUDGE",32     # early 32B judge attempts
    elif sz==[]: stage,qwen="ABORTED_NOMODEL",None
    else: stage,qwen="MIXED",None
    out.append(dict(jobid=jid,file=fn,group=group,stage=stage,qwen_b=qwen,phase=None,header=header,
                    t_start=t0.isoformat(sep=" "), t_end=t1.isoformat(sep=" "), end_source=esrc,
                    wall_hours=round(wall,4), gpu_busy_secs=gpu_busy_secs(lines),
                    n_stories_total=(first_int(lines,r"Total stories in shard:\s*(\d+)")
                                     or first_int(lines,r"Total stories:\s*(\d+)")
                                     or first_int(lines,r"Discovered\s+(\d+)\s+stories")),
                    n_stories_done=last_int(lines,r"DONE\.\s*(\d+)\s*stories"),
                    judge_calls=last_int(lines,r"judge calls:\s*([\d,]+)"),
                    finished=has_footer, n_lines=len(lines)))
    return out

recs=[]
for g,fs in LOG_DIRS.items():
    for p in fs: recs.extend(parse(p,g))

for r in recs:
    h=(r.get("header") or "")+" "+r["file"]
    r["is_wp"]=bool(re.search(r"wp_human|WritingPrompts|/wp/",h,re.I))

# explicit job-id lists for superseded early 32B human-judge attempts (verified by inspection:
#  - 47658211: 32B judge, 383 stories, output ending_judge.json (NOT the canonical sh*.json) -> rerun on 8B
#  - 47680420: 32B judge, partial (13 stories) -> aborted/rerun
#  - 47681574/47681575/47681576: "Shard 1/2/3" jobs that loaded Qwen3-32B and crashed (engine died)
#  - 47682009: 32B judge, 1 story then superseded by 8B rerun 47682295
SUPERSEDED_32B_JUDGE = {"47658211","47680420","47681574","47681575","47681576","47682009"}
# models_gen jobs that crashed at startup (11-line logs, immediate 'Finished', 0 stories generated)
DEAD_GEN = {"47712483","47712548"}

def bucket(r):
    fn=r["file"]; g=r["group"]; stg=r["stage"]
    if g=="cwb_logs":
        if fn.startswith(("cwb_rep2","cwb_rep3")): return "replicate_rep2_3"
        if fn.startswith("eval_narrative"): return "pipeline_condition"
        if fn.startswith("narrative_"): return "api_eqbench_nogpu"
    if r["jobid"] in SUPERSEDED_32B_JUDGE: return "superseded_32b_judge"
    if r["jobid"] in DEAD_GEN: return "failed_or_aborted"
    if stg in ("ABORTED_NOMODEL","UNPARSEABLE","MIXED"): return "failed_or_aborted"
    if (not r["finished"]) and (r["wall_hours"] is not None and r["wall_hours"]<0.05) and not r.get("n_stories_done"):
        return "failed_or_aborted"
    if r.get("n_stories_done")==0 and r["finished"]: return "empty_run"
    return "canonical_rep1"
for r in recs: r["bucket"]=bucket(r)

# ---------------- denominators from endings JSONs ----------------
def scan(globpat):
    nf=0; spos=0; send=0; nsent=[]
    for f in glob.glob(globpat):
        try: d=json.load(open(f))
        except Exception: continue
        pos=d.get("positions",[]); spos+=d.get("n_positions_evaluated",len(pos))
        send+=sum(len(p.get("endings",[])) for p in pos)
        if d.get("n_sentences"): nsent.append(d["n_sentences"])
        nf+=1
    return dict(files=nf,positions=spos,endings=send,mean_nsent=round(statistics.mean(nsent),1) if nsent else None)

TOP10=['anthropic__claude-sonnet-4.5','claude-opus-4-5-20251101','claude-opus-4-6','claude-sonnet-4-6',
       'kimi-k2-thinking','moonshotai__kimi-k2','o3','openai__gpt-5.2','openrouter__horizon-alpha','openrouter__pony-alpha']
den={}
den["models_top10"]={"files":0,"positions":0,"endings":0}
for s in TOP10:
    d=scan(f"{CWB}/results/endings/{s}/i1_p*_endings.json")
    for k in ("files","positions","endings"): den["models_top10"][k]+=d[k]
for name in ["nyer_expanded","nyer_83","storystar","tell_me_a_story","gutenberg"]:
    den[f"human_{name}"]=scan(f"{NP}/results_endings/{name}/*_endings.json")
# headline human corpora (NewYorker=nyer_expanded+nyer_83, StoryStar, TellMeAStory) -- EXCLUDE gutenberg
den["human_headline"]={
    "files":den["human_nyer_expanded"]["files"]+den["human_nyer_83"]["files"]+den["human_storystar"]["files"]+den["human_tell_me_a_story"]["files"],
    "positions":den["human_nyer_expanded"]["positions"]+den["human_nyer_83"]["positions"]+den["human_storystar"]["positions"]+den["human_tell_me_a_story"]["positions"],
    "endings":den["human_nyer_expanded"]["endings"]+den["human_nyer_83"]["endings"]+den["human_storystar"]["endings"]+den["human_tell_me_a_story"]["endings"],
}
# canonical_rep1 union (models top10 + ALL human incl gutenberg, EXCL wp)
den["canonical_union_excl_wp"]={
    "files":den["models_top10"]["files"]+sum(den[f"human_{n}"]["files"] for n in ["nyer_expanded","nyer_83","storystar","tell_me_a_story","gutenberg"]),
    "positions":den["models_top10"]["positions"]+sum(den[f"human_{n}"]["positions"] for n in ["nyer_expanded","nyer_83","storystar","tell_me_a_story","gutenberg"]),
    "endings":den["models_top10"]["endings"]+sum(den[f"human_{n}"]["endings"] for n in ["nyer_expanded","nyer_83","storystar","tell_me_a_story","gutenberg"]),
}

# ---------------- aggregate GPU-hours ----------------
def S(pred): return round(sum(r["wall_hours"] for r in recs if r["wall_hours"] is not None and pred(r)),3)
def Sbusy(pred): return round(sum((r["gpu_busy_secs"] or 0) for r in recs if pred(r))/3600.0,3)
agg={}
agg["canonical_gen_GPUh"]   = S(lambda r:r["bucket"]=="canonical_rep1" and r["stage"]=="GENERATION" and not r["is_wp"])
agg["canonical_judge_GPUh"] = S(lambda r:r["bucket"]=="canonical_rep1" and r["stage"]=="JUDGE" and not r["is_wp"])
agg["canonical_total_GPUh"] = round(agg["canonical_gen_GPUh"]+agg["canonical_judge_GPUh"],3)
agg["canonical_gen_GPUbusy_h"] = Sbusy(lambda r:r["bucket"]=="canonical_rep1" and r["stage"]=="GENERATION" and not r["is_wp"])
agg["pipeline_gen_GPUh"]    = S(lambda r:r["bucket"]=="pipeline_condition" and r["stage"]=="GENERATION")
agg["pipeline_judge_GPUh"]  = S(lambda r:r["bucket"]=="pipeline_condition" and r["stage"]=="JUDGE")
agg["pipeline_total_GPUh"]  = round(agg["pipeline_gen_GPUh"]+agg["pipeline_judge_GPUh"],3)
agg["replicate_gen_GPUh"]   = S(lambda r:r["bucket"]=="replicate_rep2_3")
agg["superseded_32b_judge_GPUh"] = S(lambda r:r["bucket"]=="superseded_32b_judge")
agg["failed_overhead_GPUh"] = S(lambda r:r["bucket"] in ("failed_or_aborted","empty_run"))
agg["wp_gen_GPUh"]          = S(lambda r:r["is_wp"] and r["stage"]=="GENERATION")
agg["wp_judge_GPUh"]        = S(lambda r:r["is_wp"] and r["stage"]=="JUDGE")
agg["wp_total_GPUh"]        = round(agg["wp_gen_GPUh"]+agg["wp_judge_GPUh"],3)
agg["headline_metric_excl_wp_GPUh"] = round(agg["canonical_total_GPUh"]+agg["pipeline_total_GPUh"],3)
agg["all_productive_incl_reps_excl_wp_GPUh"] = round(agg["headline_metric_excl_wp_GPUh"]+agg["replicate_gen_GPUh"],3)
agg["all_gpu_jobs_any_GPUh"] = S(lambda r:r["stage"] in ("GENERATION","JUDGE"))

# GPU-busy reconciliation (sum of per-story "Done in Ns"); only human-gen + cwb-gen logs carry it.
# Models-gen jobs (slurm_476/477*) use cumulative batch logging without per-story markers, so their
# busy-secs are unrecoverable -> for those vLLM ran saturated and wall ~= busy. Report the gap.
agg["human_gen_GPUbusy_h_with_redundant"] = round(
    sum((r["gpu_busy_secs"] or 0) for r in recs
        if r["group"]=="human" and r["stage"]=="GENERATION" and r["bucket"]=="canonical_rep1" and not r["is_wp"])/3600.0,3)
# de-duplicated human-gen busy (unique story-id across resume runs), computed inline:
import collections as _c
_runs={'nyer_83':['nyer83_47649565.out','nyer83_47650174.out','nyer83_47658204.out'],
 'storystar':['storystar_47650175.out','storystar_47658205.out','storystar_47685401.out','storystar_47686066.out'],
 'tell_me_a_story':['tmas_47650180.out','tmas_47658208.out'],
 'nyer_expanded':['nyer_expanded_47685456.out'],'gutenberg':['gutenberg_47650176.out']}
_dd=0
for _corp,_fs in _runs.items():
    _best={}
    for _f in _fs:
        try: _s=open(f"{NP}/results_endings/logs/{_f}",errors="replace").read()
        except Exception: continue
        for _i,_sec in re.findall(r"\[([0-9A-Za-z_]+)\] Done in (\d+)s",_s):
            _best[_i]=max(_best.get(_i,0),int(_sec))
    _dd+=sum(_best.values())
agg["human_gen_GPUbusy_h_dedup"]=round(_dd/3600.0,3)
agg["human_gen_redundant_recompute_h"]=round(agg["human_gen_GPUbusy_h_with_redundant"]-agg["human_gen_GPUbusy_h_dedup"],3)

# per-unit derivations (use canonical union, excl WP) for GENERATION and JUDGE separately + combined
U=den["canonical_union_excl_wp"]
def per(h, n): return round(h*3600.0/n,4) if n else None
agg["denominators"]=den
agg["derived_per_unit"]={
  "n_stories_canonical_excl_wp": U["files"],
  "n_positions_canonical_excl_wp": U["positions"],
  "n_endings_canonical_excl_wp": U["endings"],
  "GEN_gpu_sec_per_story":    per(agg["canonical_gen_GPUh"],U["files"]),
  "GEN_gpu_sec_per_position": per(agg["canonical_gen_GPUh"],U["positions"]),
  "GEN_gpu_ms_per_ending":    round(agg["canonical_gen_GPUh"]*3600*1000/U["endings"],3) if U["endings"] else None,
  "JUDGE_gpu_sec_per_story":    per(agg["canonical_judge_GPUh"],U["files"]),
  "JUDGE_gpu_sec_per_position": per(agg["canonical_judge_GPUh"],U["positions"]),
  "JUDGE_gpu_ms_per_ending":    round(agg["canonical_judge_GPUh"]*3600*1000/U["endings"],3) if U["endings"] else None,
  "TOTAL_gpu_sec_per_story":    per(agg["canonical_total_GPUh"],U["files"]),
  "TOTAL_gpu_sec_per_position": per(agg["canonical_total_GPUh"],U["positions"]),
  "TOTAL_gpu_ms_per_ending":    round(agg["canonical_total_GPUh"]*3600*1000/U["endings"],3) if U["endings"] else None,
}

# ---------------- print ----------------
print(f"{'jobid':>9} {'group':<11} {'stage':<11} {'bucket':<20} {'wall_h':>7} {'hh:mm:ss':>9} {'busy_h':>6} {'tot':>5} {'done':>5} {'end':>4} fin wp file")
for r in sorted(recs,key=lambda x:(x['group'],x['file'])):
    wh=r['wall_hours']; bh=(r['gpu_busy_secs']/3600.0) if r['gpu_busy_secs'] else None
    print(f"{r['jobid']:>9} {r['group']:<11} {r['stage']:<11} {r['bucket']:<20} "
          f"{(f'{wh:.3f}' if wh is not None else ''):>7} {hhmm(wh):>9} {(f'{bh:.2f}' if bh else ''):>6} "
          f"{str(r.get('n_stories_total') or ''):>5} {str(r.get('n_stories_done') or ''):>5} "
          f"{r.get('end_source','')[:4]:>4} {'Y' if r['finished'] else 'n':>3} {'WP' if r['is_wp'] else '':>2} {r['file']}")

print("\n=== AGGREGATE GPU-HOURS (1 H200/job => GPU-h == wall-h) ===")
for k,v in agg.items():
    if isinstance(v,(int,float)): print(f"  {k:<38} {v:>8.3f} h  ({hhmm(v)})")

print("\n=== DENOMINATORS (stories / positions / endings) ===")
for k,v in den.items(): print(f"  {k:<28} {v}")
print("\n=== DERIVED PER-UNIT GPU COST (canonical, excl WP) ===")
for k,v in agg["derived_per_unit"].items(): print(f"  {k:<32} {v}")

# ---------------- write outputs ----------------
with open(f"{BASE}/data/04_gpu_hours.json","w") as f: json.dump(recs,f,indent=2)
with open(f"{BASE}/data/04_gpu_hours_agg.json","w") as f: json.dump(agg,f,indent=2)

cols=["jobid","file","group","bucket","stage","qwen_b","is_wp","t_start","t_end","end_source",
      "wall_hours","hh_mm_ss","gpu_busy_secs","n_stories_total","n_stories_done","judge_calls","finished","n_lines"]
with open(f"{BASE}/analyses/04_gpu_hours.csv","w",newline="") as f:
    w=csv.writer(f); w.writerow(cols)
    for r in sorted(recs,key=lambda x:(x['group'],x['file'])):
        w.writerow([r['jobid'],r['file'],r['group'],r['bucket'],r['stage'],r.get('qwen_b'),int(r['is_wp']),
                    r.get('t_start'),r.get('t_end'),r.get('end_source'),r.get('wall_hours'),
                    hhmm(r['wall_hours']),r.get('gpu_busy_secs'),r.get('n_stories_total'),
                    r.get('n_stories_done'),r.get('judge_calls'),int(bool(r['finished'])),r['n_lines']])
    w.writerow([]); w.writerow(["AGGREGATE_GPU_HOURS"])
    for k,v in agg.items():
        if isinstance(v,(int,float)): w.writerow([k,round(v,3),hhmm(v)])
    w.writerow([]); w.writerow(["DERIVED_PER_UNIT"])
    for k,v in agg["derived_per_unit"].items(): w.writerow([k,v])
print(f"\nWrote analyses/04_gpu_hours.csv, data/04_gpu_hours.json, data/04_gpu_hours_agg.json")
