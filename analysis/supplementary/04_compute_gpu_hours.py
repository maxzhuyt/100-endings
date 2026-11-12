#!/usr/bin/env python3
"""
ANALYSIS 4 - Computational cost / GPU-hours of the 100-Endings narrative-tension metric.

Parses slurm logs for the two stages of the 100-Endings pipeline:
  GENERATION  (continuation model, Qwen3-32B, T=1.2) -- writes 100 endings per judged position
  JUDGE       (verdict model,    Qwen3-8B,  T=0)     -- scores each ending vs the actual ending

Each slurm job ran on a single NVIDIA H200, so wall-clock seconds == GPU-seconds for that job.

LOG GEOMETRY (validated by inspection 2026-05-28):
  - vLLM emits 'INFO MM-DD HH:MM:SS [...]' lines ONLY during model load / startup (~first 40-60 s).
  - The verdict (judge) pipeline ALSO logs every story with a custom logger:
        'YYYY-MM-DD HH:MM:SS,mmm  INFO  ...'  -> these span the WHOLE run.
  - The generation pipeline logs per-story progress WITHOUT timestamps
        ('  [id] Done in Ns', '    Batch k: positions a-b (Ns)') -> embedded durations.
  - Wrapper lines bracket the sbatch wrapper:
        '=== <name> at <DOW> <Mon> <DD> <HH:MM:SS> <TZ> <YYYY> ==='  (start)
        '=== Finished at ... ==='  /  '=== Done at ... ==='          (end)
    The results_endings GENERATION logs have NO wrapper and NO progress timestamps.

WALL-CLOCK STRATEGY (per job):
  wall_clock = max(all parsed timestamps) - min(all parsed timestamps),
  where timestamps are parsed from BOTH the 'MM-DD HH:MM:SS' forms (vLLM + custom logger)
  AND the wrapper '... <Mon> <DD> <HH:MM:SS> ...' forms (month name -> number, year known).
  This is exact for JUDGE logs (custom logger covers the whole run) and for the CWB
  GENERATION logs (the wrapper 'Finished' line gives the true end).

  For the results_endings GENERATION logs (no wrapper, no progress timestamps) the
  timestamp span only covers startup, so we instead reconstruct compute time as
        sum('Done in Ns')  [+ startup span]
  and cross-check against (file mtime - start). We report this as a BOUNDED estimate.

Outputs:
  analyses/04_gpu_hours.csv   (per-job rows + stage/overall totals)
  (markdown written separately by the calling agent)
"""
import os, re, glob, json, csv, statistics
from datetime import datetime

ROOT = "/project/jevans/maxzhuyt"
GEN_CWB_DIR   = f"{ROOT}/narrative_fiction/creative-writing-bench/results/endings"
JUDGE_CWB_DIR = f"{ROOT}/narrative_fiction/creative-writing-bench/results/judge"
RE_LOG_DIR    = f"{ROOT}/narrative_project/results_endings/logs"
OUT_CSV       = f"{ROOT}/narrative_project/paper_revision/analyses/04_gpu_hours.csv"
YEAR = 2026

MONTHS = {m:i for i,m in enumerate(
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], start=1)}

# ---- timestamp extraction --------------------------------------------------
re_mmdd   = re.compile(r'(?<!\d)(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')          # 03-27 15:49:00  (also matches YYYY-MM-DD's MM-DD tail, fine)
re_ymd    = re.compile(r'(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')         # 2026-03-27 13:02:22
re_wrap   = re.compile(r'=== .*?(?:at|Finished at|Done at) (?:Sun|Mon|Tue|Wed|Thu|Fri|Sat) '
                       r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +(\d{1,2}) '
                       r'(\d{2}):(\d{2}):(\d{2})')
re_doneins = re.compile(r'Done in (\d+)s')
re_total_shard = re.compile(r'Total stories in shard:\s*(\d+)')
re_discovered  = re.compile(r'Discovered\s+(\d+)\s+stories')
re_done_n      = re.compile(r'DONE\.\s*(\d+)\s+stories')
re_time_h      = re.compile(r'Time:\s*([\d.]+)h')

def parse_all_timestamps(text):
    """Return sorted list of datetime objects from every recognizable timestamp form."""
    dts = []
    # Full YYYY-MM-DD first (custom logger) -- most reliable, spans whole judge run.
    ymd_spans = []
    for m in re_ymd.finditer(text):
        y,mo,d,H,M,S = map(int, m.groups())
        dts.append(datetime(y,mo,d,H,M,S))
        ymd_spans.append(m.span())
    # MM-DD HH:MM:SS (vLLM) -- but skip the tail of a YYYY-MM-DD match to avoid double count.
    def inside_ymd(pos):
        for a,b in ymd_spans:
            if a <= pos < b: return True
        return False
    for m in re_mmdd.finditer(text):
        if inside_ymd(m.start()):
            continue
        mo,d,H,M,S = map(int, m.groups())
        dts.append(datetime(YEAR,mo,d,H,M,S))
    # wrapper lines with month name
    for m in re_wrap.finditer(text):
        mon,d,H,M,S = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        dts.append(datetime(YEAR, MONTHS[mon], d, H, M, S))
    return sorted(dts)

def classify(path):
    """Return ('gen'|'judge', source_tag, model)."""
    base = os.path.basename(path)
    if GEN_CWB_DIR in path:
        return ("gen", "cwb_top10", "Qwen3-32B")
    if JUDGE_CWB_DIR in path:
        return ("judge", "cwb_top10", "Qwen3-8B")
    # results_endings/logs
    if base.startswith("judge"):
        return ("judge", "human_corpora", "Qwen3-8B")
    return ("gen", "human_corpora", "Qwen3-32B")

def story_count(text, stage):
    m = re_total_shard.search(text)
    if m: return int(m.group(1)), "Total stories in shard"
    m = re_discovered.search(text)
    if m: return int(m.group(1)), "Discovered N stories"
    # judge: take the LAST 'DONE. N stories' (final save)
    dn = re_done_n.findall(text)
    if dn: return int(dn[-1]), "DONE. N stories (final)"
    return None, "none"

# ---------------------------------------------------------------------------
rows = []
all_logs = (sorted(glob.glob(f"{GEN_CWB_DIR}/slurm_*.out")) +
            sorted(glob.glob(f"{JUDGE_CWB_DIR}/slurm_*.out")) +
            sorted(glob.glob(f"{RE_LOG_DIR}/*.out")))

for path in all_logs:
    text = open(path, errors="replace").read()
    stage, src, model = classify(path)
    dts = parse_all_timestamps(text)
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    nstories, ncnt_src = story_count(text, stage)

    # span from timestamps
    if dts:
        span_s = (dts[-1] - dts[0]).total_seconds()
        first, last = dts[0], dts[-1]
    else:
        span_s = 0.0; first = last = None

    # sum of embedded per-story 'Done in Ns' (generation compute)
    done_secs = [int(x) for x in re_doneins.findall(text)]
    sum_done = sum(done_secs)
    # reported 'Time: X.Xh' if present (CWB gen)
    mt = re_time_h.search(text)
    reported_h = float(mt.group(1)) if mt else None

    has_wrapper = ("=== Finished at" in text) or ("=== Done at" in text)
    method = ""
    flags = []

    if stage == "judge":
        # custom logger spans whole run -> timestamp span is authoritative
        wall_s = span_s
        method = "ts_span (custom logger, whole run)"
        if nstories == 0 or nstories is None:
            flags.append("zero/no stories -> startup-only or preempted")
    else:
        # GENERATION
        if src == "cwb_top10" and has_wrapper:
            # wrapper Finished line is the true end; ts_span includes it -> authoritative
            wall_s = span_s
            method = "ts_span (incl. wrapper Finished)"
            if reported_h is not None:
                # cross-check
                diff = abs(wall_s/3600.0 - reported_h)
                if diff > 0.25:
                    flags.append(f"wall {wall_s/3600:.2f}h vs reported {reported_h}h (diff {diff:.2f}h)")
        else:
            # results_endings gen: no wrapper, no progress timestamps.
            # vLLM timestamps cover startup only (unless a late ERROR line exists).
            # Wall-clock proxy = file-close mtime minus first parsed (model-load) timestamp.
            # This is faithful because each job had a dedicated H200 and the log is the
            # final artifact written at job end. Cross-checked against sum('Done in Ns')
            # (the sequential per-story GPU compute), which matches to within startup (~1 min).
            if first is not None:
                mtime_span = (mtime - first).total_seconds()
            else:
                mtime_span = sum_done  # degenerate (no ts at all)
            wall_s = mtime_span
            method = "mtime - first_ts [reconstructed]"
            xcheck = sum_done / 3600.0
            flags.append("no wrapper/progress-ts: reconstructed from mtime")
            flags.append(f"xcheck sum(Done in Ns)={xcheck:.2f}h")
            if sum_done > 0:
                rel = abs(wall_s - sum_done) / sum_done
                if rel > 0.15:
                    flags.append(f"WALL vs sum_done differ {rel*100:.0f}%")

    rows.append(dict(
        job=os.path.basename(path).replace(".out",""),
        stage=stage, source=src, model=model,
        n_stories=nstories, n_stories_src=ncnt_src,
        first_ts=str(first) if first else "",
        last_ts=str(last) if last else "",
        ts_span_h=round(span_s/3600.0, 4),
        sum_done_in_s=sum_done,
        reported_time_h=reported_h if reported_h is not None else "",
        wall_clock_h=round(wall_s/3600.0, 4),
        gpu_hours=round(wall_s/3600.0, 4),   # 1 H200 per job
        method=method,
        flags="; ".join(flags),
        path=path,
    ))

# ---- per-job table sorted by start time -----------------------------------
def sortkey(r):
    return (r["first_ts"] or "9999")
rows.sort(key=sortkey)

# ---- totals ----------------------------------------------------------------
def sum_gh(pred):
    return round(sum(r["gpu_hours"] for r in rows if pred(r)), 3)

tot_gen   = sum_gh(lambda r: r["stage"]=="gen")
tot_judge = sum_gh(lambda r: r["stage"]=="judge")
tot_all   = round(tot_gen + tot_judge, 3)

# "productive" jobs = jobs that actually produced stories (n_stories not in {None,0})
def productive(r): return (r["n_stories"] not in (None, 0))
tot_gen_prod   = sum_gh(lambda r: r["stage"]=="gen" and productive(r))
tot_judge_prod = sum_gh(lambda r: r["stage"]=="judge" and productive(r))
tot_prod = round(tot_gen_prod + tot_judge_prod, 3)

# wasted = startup-only/preempted/aborted jobs (no stories produced)
tot_wasted = round(tot_all - tot_prod, 3)

# ---- write CSV -------------------------------------------------------------
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
fields = ["job","stage","source","model","n_stories","n_stories_src",
          "first_ts","last_ts","ts_span_h","sum_done_in_s","reported_time_h",
          "wall_clock_h","gpu_hours","method","flags","path"]
with open(OUT_CSV,"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows: w.writerow(r)
    # blank + totals rows
    w.writerow({k:"" for k in fields})
    def trow(job, gh, note):
        w.writerow({"job":job,"stage":"TOTAL","gpu_hours":gh,"flags":note})
    trow("TOTAL_GENERATION_all_jobs", tot_gen, "Qwen3-32B, all jobs incl reruns/aborts")
    trow("TOTAL_JUDGE_all_jobs", tot_judge, "Qwen3-8B, all jobs incl reruns/aborts")
    trow("TOTAL_ALL_jobs", tot_all, "generation+judge, all jobs")
    trow("TOTAL_GENERATION_productive", tot_gen_prod, "jobs that produced stories")
    trow("TOTAL_JUDGE_productive", tot_judge_prod, "jobs that produced stories")
    trow("TOTAL_productive", tot_prod, "generation+judge, productive only")
    trow("TOTAL_wasted_startup_preempt", tot_wasted, "aborted/startup-only/preempted jobs")

# ---- positions/endings denominators from a sample -------------------------
def sample_positions(files, cap):
    npos=[]; nend=set()
    for fp in files[:cap]:
        try: d=json.load(open(fp))
        except Exception: continue
        judged=[p for p in d.get('positions',[]) if isinstance(p,dict) and p.get('endings')]
        npos.append(len(judged))
        for p in judged: nend.add(len(p['endings']))
    return npos, nend

human_dirs = ['gutenberg','storystar','nyer_83','nyer_expanded','tell_me_a_story','wp_human']
pos_by_corpus = {}
all_pos=[]; end_sizes=set()
for c in human_dirs:
    fs = sorted(glob.glob(f"{ROOT}/narrative_project/results_endings/{c}/*_endings.json"))
    npos, nend = sample_positions(fs, 400)
    pos_by_corpus[c] = dict(n_files=len(npos),
                            mean_pos=round(statistics.mean(npos),2) if npos else 0,
                            total_pos=sum(npos))
    all_pos += npos; end_sizes |= nend
mfiles = sorted(glob.glob(f"{GEN_CWB_DIR}/claude-sonnet-4-6/i1_p*_endings.json"))
mpos, mend = sample_positions(mfiles, 400)
pos_by_corpus['MODEL_claude-sonnet-4-6'] = dict(n_files=len(mpos),
        mean_pos=round(statistics.mean(mpos),2) if mpos else 0, total_pos=sum(mpos))
all_pos += mpos; end_sizes |= mend

overall_mean_pos = round(statistics.mean(all_pos),2) if all_pos else 0

# ---- print summary ---------------------------------------------------------
print("="*78)
print("PER-JOB (sorted by start). gpu_hours == wall_clock (1 H200/job).")
print("="*78)
hdr = f"{'job':<28}{'stage':<6}{'src':<14}{'n_st':>6}{'gpu_h':>9}  method"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['job']:<28}{r['stage']:<6}{r['source']:<14}"
          f"{(r['n_stories'] if r['n_stories'] is not None else '-'):>6}"
          f"{r['gpu_hours']:>9}  {r['method']}")
    if r["flags"]:
        print(f"    flags: {r['flags']}")

print("\n"+"="*78); print("TOTALS (GPU-hours)"); print("="*78)
print(f"  GENERATION (Qwen3-32B), all jobs : {tot_gen:8.3f}")
print(f"  JUDGE      (Qwen3-8B),  all jobs : {tot_judge:8.3f}")
print(f"  ALL jobs                         : {tot_all:8.3f}")
print(f"  -- productive generation         : {tot_gen_prod:8.3f}")
print(f"  -- productive judge              : {tot_judge_prod:8.3f}")
print(f"  -- productive total              : {tot_prod:8.3f}")
print(f"  -- wasted (aborts/preempt/startup): {tot_wasted:8.3f}")

print("\n"+"="*78); print("POSITION / ENDING DENOMINATORS (sampled from endings JSON)"); print("="*78)
for k,v in pos_by_corpus.items():
    print(f"  {k:<26} files={v['n_files']:>4}  mean_judged_pos/story={v['mean_pos']:>7}  total_pos={v['total_pos']}")
print(f"  endings-per-position observed set: {sorted(end_sizes)}")
print(f"  OVERALL mean judged positions/story (pooled sample): {overall_mean_pos}")

# Save a small JSON of denominators + totals for the markdown step
summ = dict(
  totals=dict(gen_all=tot_gen, judge_all=tot_judge, all=tot_all,
              gen_productive=tot_gen_prod, judge_productive=tot_judge_prod,
              productive=tot_prod, wasted=tot_wasted),
  denominators=dict(pos_by_corpus=pos_by_corpus,
                    endings_per_position=sorted(end_sizes),
                    overall_mean_pos=overall_mean_pos),
  n_jobs=len(rows),
)
json.dump(summ, open(f"{ROOT}/narrative_project/paper_revision/analyses/04_gpu_hours_summary.json","w"), indent=2)
print("\nWrote", OUT_CSV)
print("Wrote", f"{ROOT}/narrative_project/paper_revision/analyses/04_gpu_hours_summary.json")
