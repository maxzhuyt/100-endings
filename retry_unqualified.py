#!/usr/bin/env python3
"""
Retry unqualified stories (outside 2000-3000 word range).

Phase 2: Re-run step2 only (reuse warmup+beats from metadata). Up to 3 retries.
Phase 3: Generate entirely new stories (new idea + full pipeline) to fill to n=15.

Usage:
    # Phase 2 retries
    python retry_unqualified.py --phase 2

    # Phase 3 retries
    python retry_unqualified.py --phase 3

    # Both
    python retry_unqualified.py --phase 2 --phase 3

    # Dry run (show what would be retried)
    python retry_unqualified.py --phase 2 --phase 3 --dry-run
"""

import os
import sys
import json
import time
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

from utils.api import APIClient
from configs import build_pipeline_from_args, add_pipeline_args, load_prompts

BASE_DIR = "/project/jevans/maxzhuyt/narrative_ablation"
STORIES_DIR = os.path.join(BASE_DIR, "results", "stories")
MIN_WORDS, MAX_WORDS = 2000, 3000
MAX_STEP2_RETRIES = 3


def is_qualified(text):
    wc = len(text.split())
    return MIN_WORDS <= wc <= MAX_WORDS


def get_unqualified(config_id):
    """Return list of (filename, word_count) for unqualified stories."""
    sdir = os.path.join(STORIES_DIR, config_id)
    unq = []
    for f in sorted(os.listdir(sdir)):
        if f.endswith('.txt') and '_meta' not in f and '_config' not in f:
            text = open(os.path.join(sdir, f)).read()
            wc = len(text.split())
            if not is_qualified(text):
                unq.append((f, wc))
    return unq


def retry_step2(api, model, config_id, filename):
    """Re-run step2 using beats from existing metadata. Returns True if qualified."""
    sdir = os.path.join(STORIES_DIR, config_id)
    meta_path = os.path.join(sdir, filename.replace('.txt', '_meta.json'))
    story_path = os.path.join(sdir, filename)

    if not os.path.exists(meta_path):
        log.warning(f"  No metadata for {config_id}/{filename}, skipping")
        return False

    meta = json.load(open(meta_path))
    beats = meta.get('narrative_beats')

    # Determine prompt style: beats-based or vanilla-instruction
    is_vanilla = beats is None or beats == ""

    if is_vanilla:
        # Reconstruct the original idea from metadata
        original_prompt = meta.get('narrative_idea_gen') or meta.get('prompt_id', '')
        # Try to load original idea from story_ideas.json
        if not original_prompt or len(original_prompt) < 50:
            ideas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "data", "story_ideas.json")
            if os.path.exists(ideas_path):
                with open(ideas_path) as f:
                    all_ideas = json.load(f)
                pid = meta.get('prompt_id', '')
                for entry in all_ideas:
                    if entry.get('id') == pid:
                        idea = entry.get('idea', '')
                        title = entry.get('title')
                        original_prompt = f"{title}\n\n{idea}" if title else idea
                        break
        if not original_prompt:
            log.warning(f"  Cannot reconstruct prompt for {config_id}/{filename}, skipping")
            return False
        log.info(f"  Vanilla-instruction retry (no beats)")

    best_response, best_wc, best_attempt = None, 0, 0

    for attempt in range(1, MAX_STEP2_RETRIES + 1):
        log.info(f"  Step2 retry {attempt}/{MAX_STEP2_RETRIES} for {config_id}/{filename}")

        if is_vanilla:
            prompt = (
                "Here is a story concept:\n\n"
                "---\n"
                f"{original_prompt}\n"
                "---\n\n"
                "Write the story now. "
                "Write only the story — no commentary."
            )
        else:
            prompt = (
                "Here is a structural to-do list for a short story:\n\n"
                "---\n"
                f"{beats}\n"
                "---\n\n"
                "Write the story now. Follow the structural beats closely. "
                "Write only the story — no commentary."
            )

        response = api.generate(
            model, prompt,
            temperature=0.7, max_tokens=4000, min_p=0.1, include_seed=False
        )
        response = response.strip()
        wc = len(response.split())

        # Track the closest to target range
        if best_response is None or abs(wc - 2500) < abs(best_wc - 2500):
            best_response, best_wc, best_attempt = response, wc, attempt

        if is_qualified(response):
            log.info(f"  SUCCESS: {wc} words (attempt {attempt})")
            # Overwrite story
            with open(story_path, 'w') as f:
                f.write(response)
            # Update metadata
            meta['narrative_story'] = response
            meta['words'] = wc
            meta['chars'] = len(response)
            meta['step2_retry_attempt'] = attempt
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            return True
        else:
            log.info(f"  Attempt {attempt}: {wc} words (still unqualified)")

    # All retries failed — use the closest attempt anyway
    log.warning(f"  All {MAX_STEP2_RETRIES} retries unqualified. "
                f"Using best attempt {best_attempt} ({best_wc} words)")
    with open(story_path, 'w') as f:
        f.write(best_response)
    meta['narrative_story'] = best_response
    meta['words'] = best_wc
    meta['chars'] = len(best_response)
    meta['step2_retry_attempt'] = best_attempt
    meta['step2_retry_fallback'] = True
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    return True


def retry_phase3(api, model, config_id, config_args, n_needed):
    """Generate new stories for Phase 3 to fill gaps."""
    sdir = os.path.join(STORIES_DIR, config_id)
    config_path = os.path.join(sdir, '_config.json')

    if not os.path.exists(config_path):
        log.warning(f"  No _config.json for {config_id}, skipping")
        return 0

    # Build pipeline from saved config
    pipeline = build_pipeline_from_args(config_args)

    # Find existing qualified story IDs and max index
    existing = set()
    max_idx = 0
    for f in os.listdir(sdir):
        if f.endswith('.txt') and '_meta' not in f and '_config' not in f:
            text = open(os.path.join(sdir, f)).read()
            if is_qualified(text):
                existing.add(f.replace('.txt', ''))
            # Track max index for new filenames
            try:
                parts = f.replace('.txt', '').split('_iter')
                idx = int(parts[0].split('_')[-1])
                if idx > max_idx:
                    max_idx = idx
            except:
                pass

    n_qualified = len(existing)
    log.info(f"  {config_id}: {n_qualified}/15 qualified, need {n_needed} more")

    success = 0
    # Use a dummy prompt (idea-gen replaces it anyway)
    dummy_prompt = "Write a short story."
    next_id = max_idx + 100  # start new IDs high to avoid collisions

    for i in range(n_needed):
        for attempt in range(1, MAX_STEP2_RETRIES + 1):
            log.info(f"  Generating new story {i+1}/{n_needed} (attempt {attempt})")
            t0 = time.time()

            try:
                response, ctx_dict = pipeline.run(dummy_prompt, api, model)
                response = response.strip()
                wc = len(response.split())

                if is_qualified(response):
                    # Save with new ID
                    new_id = f"t4_{next_id:03d}_iter1"
                    next_id += 1
                    story_path = os.path.join(sdir, f"{new_id}.txt")
                    meta_path = os.path.join(sdir, f"{new_id}_meta.json")

                    with open(story_path, 'w') as f:
                        f.write(response)

                    meta = {
                        "run_id": config_id,
                        "prompt_id": new_id,
                        "iteration": 1,
                        "model": model,
                        "chars": len(response),
                        "words": wc,
                        "is_retry": True,
                        **ctx_dict,
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(meta, f, indent=2)

                    dt = time.time() - t0
                    log.info(f"  SUCCESS: {new_id} = {wc} words ({dt:.1f}s)")
                    success += 1
                    break
                else:
                    log.info(f"  Attempt {attempt}: {wc} words (unqualified)")
            except Exception as e:
                log.error(f"  Attempt {attempt} error: {e}")

    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, action='append', default=[],
                        help="Phase(s) to retry: 2, 3, or both")
    parser.add_argument("--model", default="anthropic/claude-opus-4-6")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just show what would be retried")
    parser.add_argument("--threads", type=int, default=1,
                        help="Sequential by default for retries")
    add_pipeline_args(parser)
    args = parser.parse_args()

    if not args.phase:
        args.phase = [2, 3]

    api = APIClient(model_type="test")

    # ── Phase 2: Step2-only retries ──────────────────────────────────
    if 2 in args.phase:
        log.info("=" * 60)
        log.info("PHASE 2: Step2 retries for unqualified stories")
        ph2_configs = ['fulltext_beats3', 'fulltext_beats5', 'fulltext_beats10',
                       'instruction_vanilla']

        for cfg in ph2_configs:
            unq = get_unqualified(cfg)
            if not unq:
                log.info(f"  {cfg}: all qualified")
                continue

            log.info(f"  {cfg}: {len(unq)} unqualified: {[(f,w) for f,w in unq]}")
            if args.dry_run:
                continue

            for filename, wc in unq:
                retry_step2(api, args.model, cfg, filename)

    # ── Phase 3: Full redo for missing stories ───────────────────────
    if 3 in args.phase:
        log.info("=" * 60)
        log.info("PHASE 3: Generate replacement stories")

        ph3_configs_args = {
            'idea_full': ['--idea-gen'],
            'idea_no_priming': ['--idea-gen', '--no-genre-priming'],
            'idea_no_tropes': ['--idea-gen', '--no-tropes'],
            'idea_tax_only': ['--idea-gen', '--no-genre-priming', '--no-tropes'],
            'idea_tropes_only': ['--idea-gen', '--no-taxonomy'],
            'idea_bare': ['--idea-gen', '--no-taxonomy', '--no-tropes'],
            'idea_vanilla': ['--idea-gen', '--step1', 'none'],
            'idea_instruction_vanilla': ['--idea-gen', '--step1', 'none', '--vanilla-instruction'],
        }

        for cfg, extra_args in ph3_configs_args.items():
            sdir = os.path.join(STORIES_DIR, cfg)
            if not os.path.isdir(sdir):
                log.warning(f"  {cfg}: no stories dir, skipping")
                continue

            unq = get_unqualified(cfg)
            n_qualified = 15 - len(unq)
            n_needed = len(unq)

            if n_needed == 0:
                log.info(f"  {cfg}: all 15 qualified")
                continue

            log.info(f"  {cfg}: {n_qualified}/15 qualified, need {n_needed} more")
            if args.dry_run:
                continue

            # Build args for this config
            retry_parser = argparse.ArgumentParser()
            add_pipeline_args(retry_parser)
            cfg_args = retry_parser.parse_args(extra_args)

            retry_phase3(api, args.model, cfg, cfg_args, n_needed)

    log.info("=" * 60)
    log.info("Done. Re-run ending generation + judge for updated stories.")


if __name__ == "__main__":
    main()
