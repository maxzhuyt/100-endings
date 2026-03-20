# core/idea_generation.py
"""
Story idea generation using genre taxonomy, tropes, and structural priming.

Adapted from generate_story_ideas_from_tropes.py for the ablation pipeline.
Always uses genres_with_explanations (detailed definitions).

Components (independently ablatable):
  - Genre Priming:    LLM analyses random taxonomy profiles to prime narrative
                      reasoning.  Depends on taxonomy being active.
  - Taxonomy Profile: Random structural constraints from the Screenwriters
                      Taxonomy (7 levels: type, super-genre, macro/micro,
                      voice, pathway, POV).
  - Trope Selection:  Sample tropes by aspect, then LLM-select the 2 most
                      relevant for the current profile.

Pipeline integration:
  IdeaGenerationStep runs first, generates a story idea, and overwrites
  ctx.final_prompt so downstream steps (warmup → beats → story) operate
  on the generated idea.  The original prompt is preserved in metadata.
"""

import os
import json
import random
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any

from core.narrative_pipeline import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)

# ─── Directory layout ─────────────────────────────────────────────────────
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_MODULE_DIR)
_DATA_DIR = os.path.join(_REPO_DIR, "data")
DEFAULT_TROPES_PATH = os.path.join(_DATA_DIR, "tropes_filtered.jsonl")


# ─── Genre module (always with explanations) ──────────────────────────────
_genres = None


def _load_genres():
    """Lazily load genres_with_explanations from several search paths."""
    global _genres
    if _genres is not None:
        return _genres

    # 1. Direct import (repo root or sys.path)
    try:
        import genres_with_explanations as mod
        _genres = mod
        return _genres
    except ImportError:
        pass

    # 2. File-based import from data/ or repo root
    import importlib.util
    for search_dir in (_DATA_DIR, _REPO_DIR):
        path = os.path.join(search_dir, "genres_with_explanations.py")
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location(
                "genres_with_explanations", path
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _genres = mod
            return _genres

    raise ImportError(
        "Cannot find genres_with_explanations.py. "
        "Place it in the repo root or data/ directory."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Trope helpers
# ═══════════════════════════════════════════════════════════════════════════

TROPE_ASPECTS = [
    "character", "event", "conflict", "setting", "relationship",
    "dialogue", "emotion", "object", "action", "situation",
]


def load_tropes_by_aspect(filepath: str) -> Dict[str, List[Dict]]:
    """Load tropes from JSONL grouped by aspect."""
    tropes: Dict[str, List[Dict]] = {a: [] for a in TROPE_ASPECTS}
    if not os.path.exists(filepath):
        logger.warning("Tropes file not found: %s", filepath)
        return tropes
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            aspect = data.get("aspect")
            if aspect in tropes:
                tropes[aspect].append(data)
    total = sum(len(v) for v in tropes.values())
    logger.info("Loaded %d tropes from %s", total, filepath)
    return tropes


def sample_random_tropes(
    tropes_by_aspect: Dict[str, List[Dict]],
    n_per_aspect: int,
) -> List[Dict]:
    """Randomly sample up to *n_per_aspect* tropes from each aspect."""
    sampled: List[Dict] = []
    for available in tropes_by_aspect.values():
        if available:
            sampled.extend(
                random.sample(available, min(n_per_aspect, len(available)))
            )
    return sampled


# ═══════════════════════════════════════════════════════════════════════════
# Taxonomy helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_random_taxonomy_profile() -> Dict[str, str]:
    """Build a random 7-level narrative profile (Screenwriters Taxonomy)."""
    g = _load_genres()

    type_cat = random.choice(list(g.TYPES.keys()))
    specific_type = random.choice(g.TYPES[type_cat])
    super_genre = random.choice(g.SUPER_GENRES)
    macro = random.choice(list(g.MACRO_MICRO_GENRES.keys()))
    micro = random.choice(g.MACRO_MICRO_GENRES[macro])
    voice_cat = random.choice(list(g.VOICE.keys()))
    voice_val = random.choice(g.VOICE[voice_cat])
    pathway = random.choice(g.PATHWAYS)
    pov = random.choice(g.POV)

    return {
        "Type": f"{type_cat} ({specific_type})",
        "Super Genre": super_genre,
        "Macro/Micro Genre": f"{macro} -> {micro}",
        "Voice": f"{voice_cat}: {voice_val}",
        "Pathway": pathway,
        "POV": pov,
    }


def format_profile(profile: Dict[str, str]) -> str:
    """Render a taxonomy profile as a newline-separated key: value string."""
    return "\n".join(f"{k}: {v}" for k, v in profile.items())


# ═══════════════════════════════════════════════════════════════════════════
# Genre priming
# ═══════════════════════════════════════════════════════════════════════════

def build_priming_prompt(n_profiles: int = 5) -> Tuple[str, List[Dict]]:
    """
    Build the genre-priming prompt from *n_profiles* random taxonomy profiles.

    Returns (prompt_text, list_of_profiles).
    """
    profiles = [get_random_taxonomy_profile() for _ in range(n_profiles)]

    formatted = ""
    for i, p in enumerate(profiles):
        formatted += f"PROFILE {i + 1}:\n"
        formatted += "\n".join(f"- {k}: {v}" for k, v in p.items())
        formatted += "\n\n"

    prompt = (
        "You are a literary theorist specializing in narrative theory and "
        "structural analysis.\n\n"
        f"I am providing you with {n_profiles} complex narrative profiles "
        "based on the Screenwriters Taxonomy.\n\n"
        f"{formatted}"
        "For EACH profile:\n"
        "1. Identify ONE famous short story that fits this specific ontology.\n"
        "2. Briefly analyze the structure: consider how the **Pathway** and "
        "**POV** specifically serve the **Macro/Micro Genre** in that story; "
        "discuss how the **Voice** choice changes the feel of the "
        "**Super Genre**.\n"
        "3. Explain in 1-2 sentences what narrative technique or structural "
        "choice made it memorable. Focus on specific craft elements: how did "
        "the author create tension, develop character, and subvert "
        "expectations? What makes the ending particularly memorable or "
        "surprising?\n"
        "(Note: The profile includes specific definitions for the genres. "
        "You must adhere to the *definition* provided, not just the general "
        "vibe of the genre name.)\n\n"
        "Your goal is to activate your knowledge of deep narrative mechanics."
    )
    return prompt, profiles


# ═══════════════════════════════════════════════════════════════════════════
# Trope selection prompt
# ═══════════════════════════════════════════════════════════════════════════

def build_trope_selection_prompt(
    profile: Dict[str, str],
    tropes: List[Dict],
) -> str:
    """Prompt to LLM-select the 2 most relevant tropes for a profile."""
    profile_str = format_profile(profile)
    tropes_list = "\n".join(
        f"- {t['trope']}: {t['summary']}" for t in tropes
    )
    return (
        "You are a helpful assistant that selects relevant narrative "
        "elements.\n\n"
        f"Here is a Narrative Taxonomy Profile:\n{profile_str}\n\n"
        f"Here is a list of randomly selected tropes:\n{tropes_list}\n\n"
        "Identify the 2 most relevant tropes from the list above that "
        "would best fit this profile. Return ONLY those 2 tropes with "
        "their summaries, exactly as they appear in the list. Do not "
        "include any other text."
    )


# ═══════════════════════════════════════════════════════════════════════════
# Idea prompt builder
# ═══════════════════════════════════════════════════════════════════════════

def build_idea_prompt(
    profile: Optional[Dict[str, str]] = None,
    selected_tropes: Optional[str] = None,
    priming_context: Optional[str] = None,
) -> str:
    """
    Build the full single-string idea-generation prompt.

    Adapts the prompt body to whichever components are active so every
    ablation condition gets a coherent instruction.

    NOTE: The original code used chat messages with an ``assistant``
    pre-fill for priming context.  Here priming is embedded in the
    prompt body (works with any single-string API).
    """
    # ── system-level header ───────────────────────────────────────
    header = (
        "You are an expert creative writer who has deeply analyzed what "
        "makes great stories work. You do not offer advice, you only "
        "generate story concepts.\n"
    )

    # ── optional priming preamble ─────────────────────────────────
    preamble = ""
    if priming_context:
        preamble = (
            "\nHere is your previous analysis of how narrative structures "
            "work in practice:\n\n"
            f"{priming_context}\n\n---\n"
        )

    # ── task body (varies by active components) ───────────────────
    body_parts: List[str] = []

    if profile and selected_tropes:
        body_parts.append(
            "Task: Generate a high-concept story idea. You must strictly "
            "adhere to this Narrative Taxonomy Profile.\n"
            "(Note: The profile below includes specific definitions for "
            "the genres. You must adhere to the *definition* provided, "
            "not just the general vibe of the genre name.)"
        )
        body_parts.append(format_profile(profile))
        body_parts.append(f"Here are 2 relevant tropes:\n{selected_tropes}")
        body_parts.append(
            "Use them to generate a compelling story idea (3-4 sentences). "
            "Before you generate, think about: what is the core tension "
            "that the story is situated in? What makes this story "
            "impossible to put down?\n\n"
            "Your response should include the story idea only. No intro, "
            "no outro. Be specific. Avoid generic tropes. Lean into the "
            "weird combinations of the taxonomy (e.g., if asked for a "
            "Horror Comedy with a Slasher subject but a 'Family Friendly' "
            "tone, make that contradiction work)."
        )

    elif profile:                       # taxonomy only, no tropes
        body_parts.append(
            "Task: Generate a high-concept story idea. You must strictly "
            "adhere to this Narrative Taxonomy Profile.\n"
            "(Note: The profile below includes specific definitions. "
            "Adhere to the *definition*, not just the name.)"
        )
        body_parts.append(format_profile(profile))
        body_parts.append(
            "Generate a compelling story idea (3-4 sentences). Think "
            "about: what is the core tension? What makes this story "
            "impossible to put down?\n\n"
            "Your response should include the story idea only. No intro, "
            "no outro. Be specific. Lean into the weird combinations of "
            "the taxonomy."
        )

    elif selected_tropes:               # tropes only, no taxonomy
        body_parts.append(
            "Task: Generate a high-concept story idea inspired by "
            "these tropes:"
        )
        body_parts.append(selected_tropes)
        body_parts.append(
            "Generate a compelling story idea (3-4 sentences). Think "
            "about: what is the core tension? What makes this story "
            "impossible to put down?\n\n"
            "Your response should include the story idea only. No intro, "
            "no outro. Be specific, surprising, and inventive."
        )

    else:                               # everything ablated
        body_parts.append(
            "Task: Generate a high-concept, original story idea (3-4 "
            "sentences). Think about: what is the core tension? What "
            "makes this story impossible to put down? Be specific, "
            "surprising, and inventive.\n\n"
            "Your response should include the story idea only. No intro, "
            "no outro."
        )

    body = "\n\n".join(body_parts)
    return f"{header}{preamble}\n{body}"


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline step
# ═══════════════════════════════════════════════════════════════════════════

class IdeaGenerationStep(PipelineStep):
    """
    Pipeline step: generate a story idea from taxonomy / tropes / priming.

    The generated idea **replaces** ``ctx.final_prompt`` so every downstream
    step (warmup → beats → story) automatically operates on the new idea.
    The original prompt is preserved in step metadata.

    Ablation knobs
    ~~~~~~~~~~~~~~
    ``use_genre_priming``
        LLM analyses random taxonomy profiles to prime narrative reasoning.
        Hard-depends on ``use_taxonomy``; turning taxonomy off also disables
        priming.

    ``use_taxonomy``
        Include a random Screenwriters-Taxonomy profile as a structural
        constraint in the idea prompt.

    ``use_tropes``
        Sample tropes from 10 aspect categories, then LLM-select the 2
        most relevant (or pick 2 randomly when taxonomy is off).

    Implementation notes
    ~~~~~~~~~~~~~~~~~~~~
    * Priming is generated **once** per pipeline instance and cached
      (thread-safe).  The original script re-primed every 5 ideas;
      single priming is better for controlled ablation (lower variance).
    * Priming context is embedded in the prompt body rather than as a
      chat ``assistant`` pre-fill, because ``api.generate`` takes a
      single string.  If you need exact chat fidelity, extend the API
      client with a ``generate_chat`` method.
    """

    name = "idea_gen"

    def __init__(
        self,
        use_genre_priming: bool = True,
        use_taxonomy: bool = True,
        use_tropes: bool = True,
        tropes_path: str = DEFAULT_TROPES_PATH,
        n_tropes_per_aspect: int = 2,
        n_priming_profiles: int = 5,
    ):
        # Enforce dependency: no taxonomy → no priming
        if not use_taxonomy:
            use_genre_priming = False

        self.use_genre_priming = use_genre_priming
        self.use_taxonomy = use_taxonomy
        self.use_tropes = use_tropes
        self.n_tropes_per_aspect = n_tropes_per_aspect
        self.n_priming_profiles = n_priming_profiles

        # Lazy-loaded resources
        self.tropes_by_aspect: Dict[str, List[Dict]] = {}
        if self.use_tropes:
            self.tropes_by_aspect = load_tropes_by_aspect(tropes_path)

        # Priming cache (generated once on first call, then reused)
        self._priming_text: Optional[str] = None
        self._priming_lock = threading.Lock()

    # ── internal helpers ──────────────────────────────────────────

    def _ensure_priming(self, api, model: str) -> None:
        """Generate and cache the priming analysis (thread-safe)."""
        if not self.use_genre_priming:
            return
        with self._priming_lock:
            if self._priming_text is not None:
                return
            prompt, _profiles = build_priming_prompt(self.n_priming_profiles)
            logger.info(
                "Generating genre priming (%d profiles)…",
                self.n_priming_profiles,
            )
            self._priming_text = api.generate(
                model,
                prompt,
                temperature=0.7,
                max_tokens=4000,
                min_p=0.1,
                include_seed=False,
            )
            logger.info("Genre priming generated and cached.")

    def _select_tropes(
        self,
        sampled: List[Dict],
        profile: Optional[Dict[str, str]],
        api,
        model: str,
    ) -> str:
        """LLM-pick 2 most relevant tropes, or random 2 if no profile."""
        if profile:
            sel_prompt = build_trope_selection_prompt(profile, sampled)
            return api.generate(
                model,
                sel_prompt,
                temperature=0.3,
                max_tokens=500,
                min_p=0.1,
                include_seed=False,
            )
        # No profile → pick 2 at random (no LLM curation possible)
        logger.info("No taxonomy profile — selecting 2 tropes randomly.")
        chosen = random.sample(sampled, min(2, len(sampled)))
        return "\n".join(
            f"- {t['trope']}: {t['summary']}" for t in chosen
        )

    # ── main entry point ──────────────────────────────────────────

    def __call__(
        self, ctx: PipelineContext, api, model: str, **kw
    ) -> str:
        # 1. Genre priming (cached after first call)
        priming: Optional[str] = None
        if self.use_genre_priming:
            self._ensure_priming(api, model)
            priming = self._priming_text

        # 2. Taxonomy profile
        profile: Optional[Dict[str, str]] = None
        if self.use_taxonomy:
            profile = get_random_taxonomy_profile()

        # 3. Trope selection
        selected_tropes_text: Optional[str] = None
        sampled_tropes: List[Dict] = []
        if self.use_tropes and self.tropes_by_aspect:
            sampled_tropes = sample_random_tropes(
                self.tropes_by_aspect, self.n_tropes_per_aspect
            )
            if sampled_tropes:
                selected_tropes_text = self._select_tropes(
                    sampled_tropes, profile, api, model
                )

        # 4. Build prompt and generate idea
        full_prompt = build_idea_prompt(
            profile=profile,
            selected_tropes=selected_tropes_text,
            priming_context=priming,
        )
        idea_text = api.generate(
            model,
            full_prompt,
            temperature=0.9,
            max_tokens=2000,
            min_p=0.1,
            include_seed=False,
        )

        # 5. Store metadata and update downstream prompt
        meta: Dict[str, Any] = {
            "original_prompt": ctx.final_prompt,
            "use_genre_priming": self.use_genre_priming,
            "use_taxonomy": self.use_taxonomy,
            "use_tropes": self.use_tropes,
        }
        if profile:
            meta["taxonomy_profile"] = profile
        if sampled_tropes:
            meta["sampled_tropes"] = [t["trope"] for t in sampled_tropes]
        if selected_tropes_text:
            meta["selected_tropes"] = selected_tropes_text

        ctx.set(self.name, idea_text, **meta)
        ctx.final_prompt = idea_text  # downstream steps see the generated idea
        return idea_text

    def __repr__(self):
        flags = []
        if not self.use_genre_priming:
            flags.append("no_priming")
        if not self.use_taxonomy:
            flags.append("no_taxonomy")
        if not self.use_tropes:
            flags.append("no_tropes")
        tag = ", ".join(flags) if flags else "full"
        return f"IdeaGenerationStep({tag})"