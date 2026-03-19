# core/narrative_pipeline.py

"""
Modular narrative generation pipeline.

Each "step" is a callable with signature:
    step(ctx: PipelineContext, api, model, **kwargs) -> str

PipelineContext holds the original prompt plus all prior step outputs,
so any step can reference any earlier output.

Usage:
    # Full 3-step pipeline
    pipeline = NarrativePipeline([
        WarmupStep(templates=step0_data),
        BeatSheetStep(),
        StoryWriteStep(),
    ])

    # Ablate warmup: skip step0, step1 gets no beat sheet reference
    pipeline = NarrativePipeline([
        BeatSheetStep(),
        StoryWriteStep(),
    ])

    # Ablate both planning steps: equivalent to vanilla model
    pipeline = NarrativePipeline([
        VanillaStep(),
    ])

    # Custom: short 3-point to-do list
    pipeline = NarrativePipeline([
        WarmupStep(templates=step0_data),
        BeatSheetStep(max_beats=3),
        StoryWriteStep(),
    ])

    # No pre-computed templates: generate warmup live via API call
    # (defaults to c1_specific style with famous short stories)
    pipeline = NarrativePipeline([
        GenerateWarmupStep(),                    # or style="baseline"
        BeatSheetStep(),
        StoryWriteStep(),
    ])
"""

import random
import logging
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class PipelineContext:
    """Accumulates outputs from each step so later steps can reference earlier ones."""

    def __init__(self, final_prompt: str):
        self.final_prompt = final_prompt   # the writing prompt with <SEED> replaced
        self.step_outputs: Dict[str, str] = {}  # step_name -> output text
        self.metadata: Dict[str, Any] = {}      # step_name -> arbitrary metadata

    def set(self, step_name: str, output: str, **meta):
        self.step_outputs[step_name] = output
        if meta:
            self.metadata[step_name] = meta

    def get(self, step_name: str) -> Optional[str]:
        return self.step_outputs.get(step_name)

    def last_output(self) -> Optional[str]:
        """Return the most recently added step output."""
        if not self.step_outputs:
            return None
        return list(self.step_outputs.values())[-1]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage in results JSON."""
        d = {}
        for name, output in self.step_outputs.items():
            d[f"narrative_{name}"] = output
        for name, meta in self.metadata.items():
            for k, v in meta.items():
                d[f"narrative_{name}_{k}"] = v
        return d


# ---------------------------------------------------------------------------
# Base step
# ---------------------------------------------------------------------------

class PipelineStep:
    """Base class for pipeline steps."""

    name: str = "base"

    def __call__(self, ctx: PipelineContext, api, model: str, **kwargs) -> str:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# Built-in steps
# ---------------------------------------------------------------------------

class WarmupStep(PipelineStep):
    """
    Step 0: Pick a pre-generated structural beat sheet from reference stories.
    No API call — just selects from pre-computed templates.

    Args:
        templates: dict of {source_id: step0_text}
        source_id: if set, always use this specific template (for ablation).
                   If None, randomly sample one.
    """
    name = "warmup"

    def __init__(self, templates: Dict[str, str], source_id: Optional[str] = None):
        self.templates = templates
        self.fixed_source_id = source_id

    def __call__(self, ctx: PipelineContext, api, model: str, **kwargs) -> str:
        if self.fixed_source_id:
            chosen = self.fixed_source_id
        else:
            chosen = random.choice(list(self.templates.keys()))

        output = self.templates[chosen]
        ctx.set(self.name, output, source_id=chosen)
        return output

    def __repr__(self):
        if self.fixed_source_id:
            return f"WarmupStep(source_id={self.fixed_source_id!r})"
        return f"WarmupStep(n_templates={len(self.templates)})"


# Default reference stories for GenerateWarmupStep (Type A: famous, no text needed)
DEFAULT_REFERENCES = [
    {"id": "A1", "ref": "The Ones Who Walk Away from Omelas by Ursula K. Le Guin"},
    {"id": "A2", "ref": "The Lottery by Shirley Jackson"},
    {"id": "A3", "ref": "Hills Like White Elephants by Ernest Hemingway"},
    {"id": "A4", "ref": "The Yellow Wallpaper by Charlotte Perkins Gilman"},
    {"id": "A5", "ref": "A Good Man Is Hard to Find by Flannery O'Connor"},
    {"id": "A6", "ref": "The Garden of Forking Paths by Jorge Luis Borges"},  # Metafiction/Structure
    {"id": "A7", "ref": "The Metamorphosis by Franz Kafka"},                 # Surrealism/Premise
    {"id": "A8", "ref": "Rashōmon by Ryūnosuke Akutagawa"},                  # Subjectivity/Perspective
    {"id": "A9", "ref": "The Necklace by Guy de Maupassant"},                # Irony/Pacing
    {"id": "A10", "ref": "The Overcoat by Nikolai Gogol"},                   # Pathos/Character
    {"id": "A11", "ref": "Girl by Jamaica Kincaid"},                         # Form/Prose Rhythm
    {"id": "A12", "ref": "Axolotl by Julio Cortázar"},                       # Magical Realism/Identity
    {"id": "A13", "ref": "Cathedral by Raymond Carver"},                     # Minimalism/Dirty Realism
    {"id": "A14", "ref": "The Lady with the Dog by Anton Chekhov"},           # Modern Realism
    {"id": "A15", "ref": "Araby by James Joyce"},                            
    {"id": "A16", "ref": "A Very Old Man with Enormous Wings by Gabriel García Márquez"}, # Imagery
    {"id": "A17", "ref": "Sonny's Blues by James Baldwin"},                  # Voice/Narrative Soul
    {"id": "A18", "ref": "The Fifth Story by Clarice Lispector"},            # Narrative Experimentation
    {"id": "A19", "ref": "A Madman's Diary by Lu Xun"},                      # Allegory/Social Critique
    {"id": "A20", "ref": "The Bear Came Over the Mountain by Alice Munro"}    # Handling of Time/Memory
]


class GenerateWarmupStep(PipelineStep):
    """
    Step 0 (live, title mode): Generate a structural beat sheet via an API call
    using only a famous story title (the LLM relies on its training knowledge).

    Args:
        style: prompt style — "c1_specific" (default), "baseline", or "c2_abstract".
        references: list of {"id": str, "ref": str} dicts. One is randomly
                    selected per call. Defaults to 20 famous short stories.
        reference_id: if set, always use this specific reference (for ablation).
        custom_prompt: override the prompt entirely. Use {ref} as placeholder
                       for the reference story name.
    """
    name = "warmup"

    def __init__(
        self,
        style: str = "c1_specific",
        references: Optional[List[Dict[str, str]]] = None,
        reference_id: Optional[str] = None,
        custom_prompt: Optional[str] = None,
    ):
        self.style = style
        self.references = references or DEFAULT_REFERENCES
        self.fixed_reference_id = reference_id
        self.custom_prompt = custom_prompt
        self._ref_map = {r["id"]: r["ref"] for r in self.references}

    def _build_prompt(self, ref_name: str) -> str:
        if self.custom_prompt:
            return self.custom_prompt.format(ref=ref_name)

        if self.style == "baseline":
            return (
                f'Create a 5-point structural beat sheet for "{ref_name}". '
                "For each beat, give it a short name and describe the narrative function "
                "it serves — what the reader experiences and why."
            )
        elif self.style == "c1_specific":
            return (
                f'Using "{ref_name}" as your reference, extract 7 scene-level beats. '
                "For each beat describe: (1) the concrete action, (2) the specific narrative "
                "technique used, and (3) identify what made it memorable. What narrative trick, "
                "thematic resonance, or structural choice made each story work?"
            )
        elif self.style == "c2_abstract":
            return (
                f'Using "{ref_name}" as your reference, identify 3 universal '
                "narrative moves that make this story work. Be fully abstract and archetypal — "
                "do not mention any specific characters, plot events, or details from the story. "
                "Each move should be described in terms directly transferable to any story in "
                "any genre."
            )
        else:
            raise ValueError(f"Unknown style: {self.style}")

    def __call__(self, ctx: PipelineContext, api, model: str, **kwargs) -> str:
        if self.fixed_reference_id:
            chosen_id = self.fixed_reference_id
        else:
            chosen_id = random.choice(list(self._ref_map.keys()))

        ref_name = self._ref_map[chosen_id]
        prompt = self._build_prompt(ref_name)

        output = api.generate(
            model, prompt,
            temperature=0.7, max_tokens=4000, min_p=0.1, include_seed=False
        )
        ctx.set(self.name, output, source_id=chosen_id, style=self.style)
        return output

    def __repr__(self):
        parts = [f"style={self.style!r}"]
        if self.fixed_reference_id:
            parts.append(f"reference_id={self.fixed_reference_id!r}")
        if self.custom_prompt:
            parts.append("custom_prompt=...")
        return f"GenerateWarmupStep({', '.join(parts)})"


class GenerateWarmupFromTextStep(PipelineStep):
    """
    Step 0 (live, full-text mode): Generate a structural beat sheet from the
    full text of a reference story (e.g., NYer corpus stories).

    The LLM reads the complete story and extracts structural patterns.

    Args:
        style: prompt style — "c1_specific" (default), "baseline", or "c2_abstract".
        sources_dir: directory containing .txt story files.
        source_ids: list of filename stems to use (e.g., ["00015", "00066"]).
                    If None, uses all .txt files in sources_dir.
        fixed_source_id: if set, always use this specific source.
    """
    name = "warmup"

    def __init__(
        self,
        style: str = "c1_specific",
        sources_dir: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        fixed_source_id: Optional[str] = None,
    ):
        import os
        self.style = style
        self.fixed_source_id = fixed_source_id
        self.sources_dir = sources_dir

        # Load story texts
        self.texts: Dict[str, str] = {}
        if sources_dir and os.path.isdir(sources_dir):
            for fname in sorted(os.listdir(sources_dir)):
                if not fname.endswith(".txt"):
                    continue
                sid = fname.replace(".txt", "")
                if source_ids and sid not in source_ids:
                    continue
                fpath = os.path.join(sources_dir, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    self.texts[sid] = f.read().strip().lstrip("\ufeff")

        if not self.texts:
            raise ValueError(f"No story texts found in {sources_dir}")

    def _build_prompt(self, story_text: str) -> str:
        preamble = f"Here is a short story:\n\n---\n{story_text}\n---\n\n"

        if self.style == "baseline":
            return (
                f"{preamble}Create a 5-point structural beat sheet for this story. "
                "For each beat, give it a short name and describe the narrative function "
                "it serves — what the reader experiences and why."
            )
        elif self.style == "c1_specific":
            return (
                f"{preamble}Using this story as your reference, extract 7 scene-level beats. "
                "For each beat describe: (1) the concrete action, (2) the specific narrative "
                "technique used, and (3) identify what made it memorable. What narrative trick, "
                "thematic resonance, or structural choice made each story work?"
            )
        elif self.style == "c2_abstract":
            return (
                f"{preamble}Using this story as your reference, identify 3 universal "
                "narrative moves that make this story work. Be fully abstract and archetypal — "
                "do not mention any specific characters, plot events, or details from the story. "
                "Each move should be described in terms directly transferable to any story in "
                "any genre."
            )
        else:
            raise ValueError(f"Unknown style: {self.style}")

    def __call__(self, ctx: PipelineContext, api, model: str, **kwargs) -> str:
        if self.fixed_source_id:
            chosen_id = self.fixed_source_id
        else:
            chosen_id = random.choice(list(self.texts.keys()))

        story_text = self.texts[chosen_id]
        prompt = self._build_prompt(story_text)

        output = api.generate(
            model, prompt,
            temperature=0.7, max_tokens=4000, min_p=0.1, include_seed=False
        )
        ctx.set(self.name, output, source_id=chosen_id, style=self.style,
                mode="full_text")
        return output

    def __repr__(self):
        parts = [f"style={self.style!r}", f"n_sources={len(self.texts)}"]
        if self.fixed_source_id:
            parts.append(f"fixed={self.fixed_source_id!r}")
        return f"GenerateWarmupFromTextStep({', '.join(parts)})"


class BeatSheetStep(PipelineStep):
    """
    Step 1: Generate adapted structural beats from a warmup template + writing prompt.

    If no warmup output exists in context, generates beats from scratch
    (i.e., warmup was ablated).

    Args:
        max_beats: if set, instruct the model to limit to N beats.
        custom_instruction: override the default prompt template entirely.
    """
    name = "beats"

    def __init__(self, max_beats: Optional[int] = None, custom_instruction: Optional[str] = None):
        self.max_beats = max_beats
        self.custom_instruction = custom_instruction

    def __call__(self, ctx: PipelineContext, api, model: str, **kwargs) -> str:
        warmup = ctx.get("warmup")

        if self.custom_instruction:
            prompt = self.custom_instruction.format(
                warmup=warmup or "",
                writing_prompt=ctx.final_prompt,
                max_beats=self.max_beats or "",
            )
        elif warmup:
            beats_constraint = ""
            if self.max_beats:
                beats_constraint = f" Limit to exactly {self.max_beats} beats."

            prompt = (
                "Here is a structural beat sheet extracted from a reference story:\n\n"
                "---\n"
                f"{warmup}\n"
                "---\n\n"
                "Using the above as a template, create an adapted to-do list "
                "of the most important structural beats needed to tell THIS story:\n\n"
                f"{ctx.final_prompt}"
                f"{beats_constraint}"
            )
        else:
            # No warmup available — generate beats from scratch
            beats_constraint = ""
            if self.max_beats:
                beats_constraint = f" Limit to exactly {self.max_beats} beats."

            prompt = (
                "Create a structural to-do list of the most important "
                "beats needed to tell THIS story:\n\n"
                f"{ctx.final_prompt}"
                f"{beats_constraint}"
            )

        output = api.generate(
            model, prompt,
            temperature=0.7, max_tokens=4000, min_p=0.1, include_seed=False
        )
        ctx.set(self.name, output)
        return output

    def __repr__(self):
        parts = []
        if self.max_beats:
            parts.append(f"max_beats={self.max_beats}")
        if self.custom_instruction:
            parts.append("custom_instruction=...")
        return f"BeatSheetStep({', '.join(parts)})"


class StoryWriteStep(PipelineStep):
    """
    Step 2: Write the story from structural beats.

    If no beats exist in context, falls back to vanilla generation.

    Args:
        custom_instruction: override the default prompt template.
    """
    name = "story"

    def __init__(self, custom_instruction: Optional[str] = None):
        self.custom_instruction = custom_instruction

    def __call__(self, ctx: PipelineContext, api, model: str, **kwargs) -> str:
        beats = ctx.get("beats")

        if self.custom_instruction:
            prompt = self.custom_instruction.format(
                beats=beats or "",
                writing_prompt=ctx.final_prompt,
            )
        elif beats:
            prompt = (
                "Here is a structural to-do list for a short story:\n\n"
                "---\n"
                f"{beats}\n"
                "---\n\n"
                "Write the story now. Follow the structural beats closely. "
                "Write only the story — no commentary."
            )
        else:
            # No beats — fall back to vanilla
            prompt = ctx.final_prompt

        output = api.generate(
            model, prompt,
            temperature=0.7, max_tokens=4000, min_p=0.1, include_seed=False
        )
        ctx.set(self.name, output)
        return output


class VanillaStep(PipelineStep):
    """Single-shot generation — equivalent to the original benchmark behavior."""
    name = "story"

    def __call__(self, ctx: PipelineContext, api, model: str, **kwargs) -> str:
        output = api.generate(
            model, ctx.final_prompt,
            temperature=0.7, max_tokens=4000, min_p=0.1, include_seed=False
        )
        ctx.set(self.name, output)
        return output


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

class NarrativePipeline:
    """
    Runs a sequence of PipelineStep objects, threading context through them.

    The final step's output is the model_response submitted for judging.
    """

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def run(self, final_prompt: str, api, model: str) -> tuple:
        """
        Execute all steps in sequence.

        Returns:
            (final_output, context_dict) where context_dict has all
            intermediate outputs for storage/debugging.
        """
        ctx = PipelineContext(final_prompt)

        for step in self.steps:
            logger.debug(f"Running pipeline step: {step.name}")
            step(ctx, api, model)

        final_output = ctx.last_output() or ""
        return final_output, ctx.to_dict()

    def __repr__(self):
        step_strs = ", ".join(repr(s) for s in self.steps)
        return f"NarrativePipeline([{step_strs}])"


# ---------------------------------------------------------------------------
# Preset configurations (convenience)
# ---------------------------------------------------------------------------

def make_full_pipeline(step0_data: Dict[str, str], source_id: Optional[str] = None,
                       max_beats: Optional[int] = None) -> NarrativePipeline:
    """Full 3-step: warmup -> beats -> story."""
    return NarrativePipeline([
        WarmupStep(templates=step0_data, source_id=source_id),
        BeatSheetStep(max_beats=max_beats),
        StoryWriteStep(),
    ])


def make_no_warmup_pipeline(max_beats: Optional[int] = None) -> NarrativePipeline:
    """Ablate step0: beats from scratch -> story."""
    return NarrativePipeline([
        BeatSheetStep(max_beats=max_beats),
        StoryWriteStep(),
    ])


def make_no_beats_pipeline(step0_data: Dict[str, str], source_id: Optional[str] = None) -> NarrativePipeline:
    """Ablate step1: warmup only, then vanilla story (no structural beats)."""
    return NarrativePipeline([
        WarmupStep(templates=step0_data, source_id=source_id),
        StoryWriteStep(),
    ])


def make_live_warmup_pipeline(style: str = "c1_specific",
                              reference_id: Optional[str] = None,
                              max_beats: Optional[int] = None) -> NarrativePipeline:
    """Full 3-step with live step0 generation from title (no pre-computed templates)."""
    return NarrativePipeline([
        GenerateWarmupStep(style=style, reference_id=reference_id),
        BeatSheetStep(max_beats=max_beats),
        StoryWriteStep(),
    ])


def make_fulltext_warmup_pipeline(style: str = "c1_specific",
                                  sources_dir: Optional[str] = None,
                                  fixed_source_id: Optional[str] = None,
                                  max_beats: Optional[int] = None) -> NarrativePipeline:
    """Full 3-step with live step0 from full story text."""
    return NarrativePipeline([
        GenerateWarmupFromTextStep(style=style, sources_dir=sources_dir,
                                   fixed_source_id=fixed_source_id),
        BeatSheetStep(max_beats=max_beats),
        StoryWriteStep(),
    ])


def make_vanilla_pipeline() -> NarrativePipeline:
    """No pipeline — equivalent to default benchmark behavior."""
    return NarrativePipeline([
        VanillaStep(),
    ])
