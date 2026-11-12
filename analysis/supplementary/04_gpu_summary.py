#!/usr/bin/env python3
"""Analysis 4: GPU-hours summary. Aggregates the per-job parse (04_gpu_hours.csv) into
generation vs judge totals and amortizes over stories / positions / ending-generations.
1 NVIDIA H200 per job; generator Qwen3-32B (T=1.2), judge Qwen3-8B (T=0)."""
import csv, json, glob
NP="/project/jevans/maxzhuyt/narrative_project"
CWB="/project/jevans/maxzhuyt/narrative_fiction/creative-writing-bench"
BASE=NP+"/paper_revision"; OUT=BASE+"/analyses"

# 1) GPU-hours by stage from the per-job parse
rows=list(csv.DictReader(open(f"{OUT}/04_gpu_hours.csv")))
def hrs(stage): return sum(float(r["wall_hours"]) for r in rows if r["stage"]==stage and r["wall_hours"] not in ("","nan"))
gen_h, judge_h = hrs("GENERATION"), hrs("JUDGE")
total_h = gen_h + judge_h

# 2) denominators: stories + positions actually evaluated by the metric
SH={}
for f in glob.glob(f"{NP}/results_endings/ending_judge_sh*.json"): SH.update(json.load(open(f)))
MJ={}
for f in ["ending_judge.json","ending_judge_gpt52.json","ending_judge_kimi.json","ending_judge_opus46.json"]:
    MJ.update(json.load(open(f"{CWB}/results/judge/{f}")))
def count(d):
    ns=len(d); npos=sum(len(v["positions"]) for v in d.values()); return ns,npos
h_ns,h_pos=count(SH); m_ns,m_pos=count(MJ)
N=h_ns+m_ns; P=h_pos+m_pos; E=P*100   # 100 endings per position

per_story_min = total_h*60/N
per_pos_sec   = total_h*3600/P
per_ending_sec= total_h*3600/E
gen_pos_sec   = gen_h*3600/P
judge_pos_sec = judge_h*3600/P

L=["# Analysis 4 — Computational complexity / GPU-hours of 100-Endings","",
   "Hardware: 1 NVIDIA H200 (141 GB) per job. Generator Qwen3-32B (vLLM, bf16, T=1.2); judge Qwen3-8B (T=0).",
   "GPU-hours = sum of per-job wall-clock (one GPU per job), parsed from slurm logs (see 04_gpu_hours.csv).","",
   "## Totals",
   "| Stage | GPU-hours |","|---|---|",
   f"| Generation (Qwen3-32B) | {gen_h:.1f} |",
   f"| Judge (Qwen3-8B) | {judge_h:.1f} |",
   f"| **Total** | **{total_h:.1f}** |","",
   "## Denominators (stories the metric was run on)",
   f"- Stories evaluated: **{N}** ({h_ns} human/baseline + {m_ns} model)",
   f"- Sentence-positions evaluated: **{P:,}** (~{P/N:.0f} per story)",
   f"- Ending generations: **{E:,}** (100 per position)","",
   "## Amortized cost",
   "| Unit | Cost |","|---|---|",
   f"| per story | {per_story_min:.2f} GPU-min |",
   f"| per position | {per_pos_sec:.2f} GPU-sec (gen {gen_pos_sec:.2f} + judge {judge_pos_sec:.2f}) |",
   f"| per ending generation | {per_ending_sec:.3f} GPU-sec |","",
   "## Scaling & answer to reviewer R1 (does it scale to long texts?)",
   f"Cost grows **linearly** as #stories x #positions x 100 generations. A story of S sentences contributes ~0.9·S evaluated positions (10-99% range), so per-story cost scales linearly in length. At ~{per_story_min:.1f} GPU-min/story for ~{P/N:.0f}-position short fiction, a 10x longer text costs ~10x (a few GPU-hours each).",
   "The judge stage (Qwen3-8B) batches all 100 verdicts per position in one call, so it is cheaper than generation despite equal position counts. Generation dominates (" + f"{gen_h/total_h*100:.0f}%" + " of GPU-hours).",
   "Practical note: prefix-caching (enable_prefix_caching=True) amortizes the shared story-prefix across the 100 endings and across positions, which is what keeps per-position cost near " + f"{per_pos_sec:.1f}s rather than 100x a single generation.",
   "",
   "*Caveat:* totals include the full evaluated corpora (incl. WritingPrompts/Gutenberg and stability repeats) since that GPU time was actually spent; per-story/per-position figures are the meaningful amortized cost of the metric."]
open(f"{OUT}/04_compute_gpu_hours.md","w").write("\n".join(L)+"\n")
json.dump({"gen_h":round(gen_h,2),"judge_h":round(judge_h,2),"total_h":round(total_h,2),
           "N_stories":N,"P_positions":P,"E_endings":E,
           "per_story_min":round(per_story_min,3),"per_pos_sec":round(per_pos_sec,3),
           "per_ending_sec":round(per_ending_sec,4)}, open(f"{OUT}/04_gpu_hours_final.json","w"), indent=1)
print(f"Analysis 4 done. gen={gen_h:.1f}h judge={judge_h:.1f}h total={total_h:.1f}h | N={N} P={P} | per-story={per_story_min:.2f}min per-pos={per_pos_sec:.2f}s")
