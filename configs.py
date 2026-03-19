"""
Pipeline configuration builder — all settings controlled via CLI flags.

Step logic:
  --step0 controls warmup mode: "title", "fulltext", "precomputed", "none"
  --step1 controls beat generation: "default", a JSON file path, or "none"
  --step2 is always present: either takes beats (from step1) or is vanilla

  If step1 is "none", step0 is automatically skipped and step2 = vanilla.

Usage from generate_stories.py:
    pipeline = build_pipeline_from_args(args)
"""

import os
import sys
import json
from typing import Optional

# Allow importing from creative-writing-bench
sys.path.insert(0, "/project/jevans/maxzhuyt/creative-writing-bench")

from core.narrative_pipeline import (
    NarrativePipeline, WarmupStep, GenerateWarmupStep,
    GenerateWarmupFromTextStep, BeatSheetStep, StoryWriteStep, VanillaStep,
)


# ── Defaults ──────────────────────────────────────────────────────────────

TEMPLATES_PATH = "/project/jevans/maxzhuyt/creative-writing-bench/data/narrative_step0_templates.json"
NYER_SOURCES_DIR = "/project/jevans/maxzhuyt/narrative_ablation/data/nyer_sources"
PROMPTS_PATH = "/project/jevans/maxzhuyt/creative-writing-bench/data/creative_writing_prompts_v3.json"


def load_precomputed_templates():
    with open(TEMPLATES_PATH) as f:
        raw = json.load(f)
    return {e["source_id"]: e["step0"] for e in raw}


def load_prompts():
    with open(PROMPTS_PATH) as f:
        return json.load(f)


# ── Pipeline builder ──────────────────────────────────────────────────────

def add_pipeline_args(parser):
    """Add pipeline CLI arguments to an argparse parser."""

    g = parser.add_argument_group("Pipeline configuration")

    g.add_argument("--step0", type=str, default="title",
                   choices=["title", "fulltext", "precomputed", "none"],
                   help=("Step 0 warmup mode. "
                         "'title': generate beat sheet from famous story title (20 stories). "
                         "'fulltext': generate beat sheet from full NYer story text (20 stories). "
                         "'precomputed': use pre-generated templates from JSON. "
                         "'none': skip warmup."))

    g.add_argument("--step0-style", type=str, default="c1_specific",
                   choices=["baseline", "c1_specific", "c2_abstract"],
                   help="Step 0 prompt style (default: c1_specific)")

    g.add_argument("--step0-source", type=str, default=None,
                   help="Fix a specific source ID for step0 (e.g., 'A1' or '00015'). "
                        "If omitted, randomly sample each iteration.")

    g.add_argument("--step0-sources-dir", type=str, default=NYER_SOURCES_DIR,
                   help="Directory of .txt files for fulltext mode")

    g.add_argument("--step0-templates", type=str, default=TEMPLATES_PATH,
                   help="JSON file for precomputed mode")

    g.add_argument("--step1", type=str, default="default",
                   help=("Step 1 beat generation. "
                         "'default': standard beat adaptation prompt. "
                         "'none': skip beats (step2 becomes vanilla, step0 also skipped). "
                         "Or path to JSON file with custom step1 prompts."))

    g.add_argument("--step1-max-beats", type=int, default=None,
                   help="Limit step1 to N beats (e.g., 3 for short beat list)")

    return parser


def build_pipeline_from_args(args) -> NarrativePipeline:
    """Build a NarrativePipeline from parsed CLI args.

    Enforces: if step1 == "none" → step0 auto-skipped, step2 = vanilla.
    """
    step1_mode = args.step1
    step0_mode = args.step0

    # ── If step1 is ablated, everything collapses to vanilla ──────────
    if step1_mode == "none":
        return NarrativePipeline([VanillaStep()])

    steps = []

    # ── Step 0 ────────────────────────────────────────────────────────
    if step0_mode == "title":
        steps.append(GenerateWarmupStep(
            style=args.step0_style,
            reference_id=args.step0_source,
        ))

    elif step0_mode == "fulltext":
        steps.append(GenerateWarmupFromTextStep(
            style=args.step0_style,
            sources_dir=args.step0_sources_dir,
            fixed_source_id=args.step0_source,
        ))

    elif step0_mode == "precomputed":
        templates = load_precomputed_templates()
        steps.append(WarmupStep(
            templates=templates,
            source_id=args.step0_source,
        ))

    elif step0_mode == "none":
        pass  # no warmup step

    # ── Step 1 ────────────────────────────────────────────────────────
    if step1_mode == "default":
        steps.append(BeatSheetStep(max_beats=args.step1_max_beats))

    else:
        # step1 is a JSON file path with custom prompts
        if os.path.isfile(step1_mode):
            with open(step1_mode) as f:
                custom_prompts = json.load(f)
            # Use the first prompt as custom_instruction template
            # JSON format: list of prompt strings with {warmup} and {writing_prompt} placeholders
            if isinstance(custom_prompts, list) and len(custom_prompts) > 0:
                steps.append(BeatSheetStep(
                    custom_instruction=custom_prompts[0],
                    max_beats=args.step1_max_beats,
                ))
            else:
                raise ValueError(f"--step1 JSON must be a non-empty list of prompt strings, "
                                 f"got: {type(custom_prompts)}")
        else:
            raise ValueError(f"--step1 must be 'default', 'none', or a path to a JSON file. "
                             f"Got: {step1_mode!r}")

    # ── Step 2 (always present) ───────────────────────────────────────
    steps.append(StoryWriteStep())

    return NarrativePipeline(steps)


def describe_pipeline(args) -> str:
    """Human-readable description of the pipeline config."""
    if args.step1 == "none":
        return "vanilla (no pipeline)"

    parts = []

    # Step 0
    if args.step0 == "none":
        parts.append("step0=none")
    elif args.step0 == "title":
        src = args.step0_source or "random"
        parts.append(f"step0=title({args.step0_style}, src={src})")
    elif args.step0 == "fulltext":
        src = args.step0_source or "random"
        parts.append(f"step0=fulltext({args.step0_style}, src={src})")
    elif args.step0 == "precomputed":
        src = args.step0_source or "random"
        parts.append(f"step0=precomputed(src={src})")

    # Step 1
    if args.step1 == "default":
        beats = f"max={args.step1_max_beats}" if args.step1_max_beats else "default"
        parts.append(f"step1=beats({beats})")
    else:
        parts.append(f"step1=custom({os.path.basename(args.step1)})")

    # Step 2 always present
    parts.append("step2=story")

    return " → ".join(parts)
