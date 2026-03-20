"""
Pipeline configuration builder — all settings controlled via CLI flags.

Step logic:
  --idea-gen  (optional) Generate story ideas from taxonomy/tropes/priming.
              Runs first; overwrites the writing prompt for downstream steps.
  --step0     Warmup mode: "title", "fulltext", "precomputed", "none"
  --step1     Beat generation: "default", a JSON file path, or "none"
  --step2     Always present: takes beats (from step1) or is vanilla.

  If step1 is "none" (and no --idea-gen), step0 is auto-skipped → vanilla.

Idea-generation ablation:
  --no-genre-priming   Disable genre priming (requires --idea-gen)
  --no-taxonomy        Disable taxonomy profile; also disables priming
  --no-tropes          Disable trope selection

Usage from generate_stories.py:
    pipeline = build_pipeline_from_args(args)
"""

import os
import json
from typing import Optional, Dict

from core.narrative_pipeline import (
    NarrativePipeline, WarmupStep, GenerateWarmupStep,
    GenerateWarmupFromTextStep, BeatSheetStep, StoryWriteStep, VanillaStep,
)

# Optional import — only needed when --idea-gen is used
try:
    from core.idea_generation import IdeaGenerationStep
    from core.idea_generation import DEFAULT_TROPES_PATH as _DEFAULT_TROPES_PATH
    _HAS_IDEA_GEN = True
except ImportError as _e:
    _HAS_IDEA_GEN = False
    _IDEA_GEN_IMPORT_ERR = str(_e)
    _DEFAULT_TROPES_PATH = None


# ── Defaults ──────────────────────────────────────────────────────────────

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TEMPLATES_PATH = os.path.join(_DATA_DIR, "narrative_step0_templates.json")
NYER_SOURCES_DIR = os.path.join(_DATA_DIR, "nyer_sources")
PROMPTS_PATH = os.path.join(_DATA_DIR, "creative_writing_prompts_v3.json")


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

    g.add_argument("--vanilla-instruction", action="store_true", default=False,
                   help="When step1=none, wrap the idea in a writing instruction "
                        "matching StoryWriteStep's framing (instruction-vanilla mode).")

    g.add_argument("--step1-max-beats", type=int, default=None,
                   help="Limit step1 to N beats (e.g., 3 for short beat list)")

    # ── Idea generation ───────────────────────────────────────────
    ig = parser.add_argument_group(
        "Idea generation (--idea-gen)",
        description="Generate story ideas on-the-fly using genre taxonomy, "
                    "tropes, and structural priming.  The generated idea "
                    "replaces the writing prompt for all downstream steps."
    )

    ig.add_argument(
        "--idea-gen", action="store_true", default=False,
        help="Enable story idea generation (taxonomy + tropes + priming)."
    )
    ig.add_argument(
        "--no-genre-priming", action="store_true", default=False,
        help="Ablate genre priming from idea generation."
    )
    ig.add_argument(
        "--no-taxonomy", action="store_true", default=False,
        help="Ablate taxonomy profile.  Also disables genre priming. "
             "Tropes (if enabled) fall back to random selection."
    )
    ig.add_argument(
        "--no-tropes", action="store_true", default=False,
        help="Ablate trope selection from idea generation."
    )
    ig.add_argument(
        "--tropes-path", type=str, default=None,
        help="Path to tropes JSONL file (default: data/tropes_filtered.jsonl)"
    )

    return parser


def build_pipeline_from_args(args) -> NarrativePipeline:
    """Build a NarrativePipeline from parsed CLI args.

    Enforces:
      - step1 == "none" and no idea-gen → vanilla.
      - step1 == "none" with idea-gen   → idea_gen + vanilla.
      - --no-taxonomy forces --no-genre-priming and --no-tropes.
    """
    step1_mode = args.step1
    step0_mode = args.step0
    has_idea_gen = getattr(args, "idea_gen", False)

    # Enforce dependency: no taxonomy → no priming (priming uses taxonomy profiles)
    if getattr(args, "no_taxonomy", False):
        args.no_genre_priming = True

    # ── Pure vanilla (no pipeline, no idea gen) ───────────────────
    use_instr = getattr(args, "vanilla_instruction", False)
    if step1_mode == "none" and not has_idea_gen:
        return NarrativePipeline([VanillaStep(with_instruction=use_instr)])

    steps = []

    # ── Idea generation (always first when active) ────────────────
    if has_idea_gen:
        if not _HAS_IDEA_GEN:
            raise ImportError(
                f"--idea-gen requires core.idea_generation module. "
                f"Import failed: {_IDEA_GEN_IMPORT_ERR}"
            )
        tropes_path = (
            getattr(args, "tropes_path", None) or _DEFAULT_TROPES_PATH
        )
        steps.append(IdeaGenerationStep(
            use_genre_priming=not getattr(args, "no_genre_priming", False),
            use_taxonomy=not getattr(args, "no_taxonomy", False),
            use_tropes=not getattr(args, "no_tropes", False),
            tropes_path=tropes_path,
        ))

    # ── Idea-gen + vanilla (no beats) ─────────────────────────────
    if step1_mode == "none":
        steps.append(VanillaStep(with_instruction=use_instr))
        return NarrativePipeline(steps)

    # ── Step 0 ────────────────────────────────────────────────────
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

    # ── Step 1 ────────────────────────────────────────────────────
    if step1_mode == "default":
        steps.append(BeatSheetStep(max_beats=args.step1_max_beats))

    else:
        # step1 is a JSON file path with custom prompts
        if os.path.isfile(step1_mode):
            with open(step1_mode) as f:
                custom_prompts = json.load(f)
            if isinstance(custom_prompts, list) and len(custom_prompts) > 0:
                steps.append(BeatSheetStep(
                    custom_instruction=custom_prompts[0],
                    max_beats=args.step1_max_beats,
                ))
            else:
                raise ValueError(
                    f"--step1 JSON must be a non-empty list of prompt "
                    f"strings, got: {type(custom_prompts)}"
                )
        else:
            raise ValueError(
                f"--step1 must be 'default', 'none', or a path to a JSON "
                f"file. Got: {step1_mode!r}"
            )

    # ── Step 2 (always present) ───────────────────────────────────
    steps.append(StoryWriteStep())

    return NarrativePipeline(steps)


def describe_pipeline(args) -> str:
    """Human-readable description of the pipeline config."""
    has_idea_gen = getattr(args, "idea_gen", False)

    if args.step1 == "none" and not has_idea_gen:
        return "vanilla (no pipeline)"

    parts = []

    # ── Idea generation ───────────────────────────────────────────
    if has_idea_gen:
        flags = []
        if getattr(args, "no_genre_priming", False):
            flags.append("no_priming")
        if getattr(args, "no_taxonomy", False):
            flags.append("no_taxonomy")
        if getattr(args, "no_tropes", False):
            flags.append("no_tropes")
        tag = ", ".join(flags) if flags else "full"
        parts.append(f"idea_gen({tag})")

    if args.step1 == "none":
        parts.append("vanilla")
        return " → ".join(parts)

    # ── Step 0 ────────────────────────────────────────────────────
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

    # ── Step 1 ────────────────────────────────────────────────────
    if args.step1 == "default":
        beats = (
            f"max={args.step1_max_beats}" if args.step1_max_beats
            else "default"
        )
        parts.append(f"step1=beats({beats})")
    else:
        parts.append(f"step1=custom({os.path.basename(args.step1)})")

    # ── Step 2 (always present) ───────────────────────────────────
    parts.append("step2=story")

    return " → ".join(parts)