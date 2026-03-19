#!/usr/bin/env python3
"""
Stage 1: Generate stories using the narrative pipeline.

All pipeline settings are CLI flags — no code edits needed for ablations.

Examples:
    # Full pipeline (title warmup, default beats)
    python generate_stories.py --run-id full

    # Vanilla (no pipeline)
    python generate_stories.py --step1 none --run-id vanilla

    # Ablate warmup (beats from scratch)
    python generate_stories.py --step0 none --run-id no_warmup

    # Full-text warmup from NYer stories
    python generate_stories.py --step0 fulltext --run-id fulltext

    # Short beats (3 only)
    python generate_stories.py --step1-max-beats 3 --run-id short_beats

    # Fixed warmup source
    python generate_stories.py --step0-source A1 --run-id fixed_leguin

    # Custom step1 prompts from JSON
    python generate_stories.py --step1 my_prompts.json --run-id custom

    # Resume (skips existing files)
    python generate_stories.py --run-id full
"""

import os
import sys
import json
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Add creative-writing-bench to path for API client
sys.path.insert(0, "/project/jevans/maxzhuyt/creative-writing-bench")
from dotenv import load_dotenv
load_dotenv("/project/jevans/maxzhuyt/creative-writing-bench/.env")

from utils.api import APIClient
from configs import add_pipeline_args, build_pipeline_from_args, describe_pipeline, load_prompts


BASE_DIR = "/project/jevans/maxzhuyt/narrative_ablation"
RESULTS_DIR = os.path.join(BASE_DIR, "results", "stories")


def generate_one(api, model, pipeline, prompt_id, final_prompt, run_id,
                 iteration, out_dir):
    """Generate a single story and save to disk. Returns (success, path)."""
    fname = f"p{prompt_id:02d}_iter{iteration}.txt"
    out_path = os.path.join(out_dir, fname)

    if os.path.exists(out_path):
        return True, out_path  # already done

    t0 = time.time()
    try:
        response, ctx_dict = pipeline.run(final_prompt, api, model)
        response = response.strip()

        if len(response) < 200:
            log.warning(f"[{run_id}] p{prompt_id} iter{iteration}: "
                        f"too short ({len(response)} chars), skipping")
            return False, out_path

        # Save story text
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response)

        # Save metadata (pipeline intermediates)
        meta_path = out_path.replace(".txt", "_meta.json")
        meta = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "iteration": iteration,
            "model": model,
            "chars": len(response),
            "words": len(response.split()),
            **ctx_dict,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        dt = time.time() - t0
        log.info(f"  [{run_id}] p{prompt_id:02d} iter{iteration}: "
                 f"{len(response.split())} words, {dt:.1f}s")
        return True, out_path

    except Exception as e:
        log.error(f"  [{run_id}] p{prompt_id:02d} iter{iteration}: {e}")
        return False, out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate stories using narrative pipeline ablation configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline configuration:
  --step0       Warmup mode: title | fulltext | precomputed | none
  --step1       Beat generation: default | none | path/to/prompts.json
  --step1=none  → step0 auto-skipped, step2 = vanilla (single-shot)

Examples:
  Full pipeline:     --step0 title --step1 default --run-id full
  Vanilla:           --step1 none --run-id vanilla
  No warmup:         --step0 none --run-id no_warmup
  Full-text warmup:  --step0 fulltext --run-id fulltext
  Short beats:       --step1-max-beats 3 --run-id short_beats
        """)

    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4-6",
                        help="Model identifier (default: anthropic/claude-sonnet-4-6)")
    parser.add_argument("--run-id", type=str, required=True,
                        help="Unique identifier for this run (used as output folder name)")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Comma-separated prompt IDs (default: all 32)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations per prompt (default: 3)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Parallel API calls (default: 4)")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)

    add_pipeline_args(parser)
    args = parser.parse_args()

    # Build pipeline from CLI args
    pipeline = build_pipeline_from_args(args)
    desc = describe_pipeline(args)

    log.info(f"Run ID: {args.run_id}")
    log.info(f"Pipeline: {desc}")
    log.info(f"Pipeline repr: {pipeline}")
    log.info(f"Model: {args.model}")

    # Load prompts
    prompts = load_prompts()
    if args.prompts:
        prompt_ids = [p.strip() for p in args.prompts.split(",")]
        prompts = {k: v for k, v in prompts.items() if k in prompt_ids}

    log.info(f"Prompts: {len(prompts)}, Iterations: {args.iterations}")

    # Output directory
    out_dir = os.path.join(args.results_dir, args.run_id)
    os.makedirs(out_dir, exist_ok=True)

    # Save config metadata
    config_path = os.path.join(out_dir, "_config.json")
    config_meta = {
        "run_id": args.run_id,
        "model": args.model,
        "step0": args.step0,
        "step0_style": args.step0_style,
        "step0_source": args.step0_source,
        "step1": args.step1,
        "step1_max_beats": args.step1_max_beats,
        "iterations": args.iterations,
        "description": desc,
        "pipeline_repr": repr(pipeline),
    }
    with open(config_path, "w") as f:
        json.dump(config_meta, f, indent=2)

    # API client
    api = APIClient(model_type="test")

    # Count tasks
    total = 0
    existing = 0
    for prompt_key in prompts:
        for iteration in range(1, args.iterations + 1):
            fname = f"p{int(prompt_key):02d}_iter{iteration}.txt"
            if os.path.exists(os.path.join(out_dir, fname)):
                existing += 1
            total += 1

    log.info(f"Total tasks: {total} ({existing} done, {total - existing} remaining)")

    # Generate
    t_start = time.time()
    success_count = 0
    fail_count = 0

    tasks = []
    for prompt_key, prompt_obj in prompts.items():
        base_prompt = prompt_obj.get("writing_prompt", "")
        seed_mods = prompt_obj.get("seed_modifiers", [])

        for iteration in range(1, args.iterations + 1):
            seed_mod = seed_mods[(iteration - 1) % len(seed_mods)]
            final_prompt = base_prompt.replace("<SEED>", seed_mod)
            tasks.append((int(prompt_key), final_prompt, iteration))

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {}
        for prompt_id, final_prompt, iteration in tasks:
            fut = executor.submit(
                generate_one, api, args.model, pipeline,
                prompt_id, final_prompt, args.run_id, iteration, out_dir
            )
            futures[fut] = (prompt_id, iteration)

        for fut in as_completed(futures):
            pid, it = futures[fut]
            try:
                ok, path = fut.result()
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                log.error(f"  p{pid} iter{it}: unhandled error: {e}")
                fail_count += 1

    elapsed = time.time() - t_start
    log.info(f"\n{'='*60}")
    log.info(f"Done in {elapsed:.0f}s. Success: {success_count}, Failed: {fail_count}")
    log.info(f"Stories saved to: {out_dir}/")


if __name__ == "__main__":
    main()
