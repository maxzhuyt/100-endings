# 100-Endings: Narrative Pipeline Ablation via Ending Geometry

Evaluates creative writing pipeline configurations by measuring **how predictable and coherent generated stories are** using 100-ending geometry — a method that generates 100 possible continuations at each position in a story and analyzes their geometric properties in embedding space.

## How It Works

```
Stage 1                    Stage 2                   Stage 3                Stage 4
Generate Stories           Generate 100 Endings      Embedding Geometry     LLM Judge
(API: Opus 4.6)          (vLLM: Qwen3-32B)        (Qwen3-Embedding)      (Qwen3-8B)

 prompt + pipeline  ─►  story text  ─►  100 endings  ─►  dist_to_actual   ─►  yes_rate
 config (CLI flags)     per position     per position     intra_mean          per ending
                                                          effective_rank
```

**Core idea:** A well-structured story should constrain what comes next. By comparing 100 possible endings at each point in a story, we measure:
- **dist_to_actual** — How close are generated endings to the true continuation? (lower = more predictable)
- **intra_mean** — How similar are the 100 endings to each other? (lower = more constrained)
- **effective_rank** — Spectral diversity: ~1 means all endings are alike, ~100 means maximally diverse
- **yes_rate** — What fraction of endings match events that actually occur? (higher = more on-target)

## Prerequisites

- **This repo** (100-endings evaluation pipeline)
- **[creative-writing-bench](https://github.com/maxzhuyt/creative-writing-bench)** (cloned at `/project/jevans/maxzhuyt/creative-writing-bench`) — provides the narrative pipeline classes and API client
- Python environment with `requests`, `python-dotenv`, `numpy`, `torch`, `transformers`
- vLLM environment for Stages 2-4 (GPU required)
- API key for OpenRouter/Anthropic in `creative-writing-bench/.env`

## Quick Start

### 1. Generate stories (CPU, ~$5-15 per config)

```bash
# Full pipeline: title warmup → beats → story
python generate_stories.py --run-id full --step0 title --step1 default

# Vanilla baseline: single-shot generation
python generate_stories.py --run-id vanilla --step1 none
```

### 2. Generate 100 endings per position (GPU, Qwen3-32B)

```bash
python generate_endings.py                         # all configs
python generate_endings.py --config full,vanilla   # specific configs
```

### 3. Compute geometry + judge (GPU)

```bash
python compute_geometry.py    # embedding distances + spectral metrics
python compute_judge.py       # LLM binary judge (Y/N per ending)
```

### On SLURM

```bash
# Stage 1: CPU node (API calls only)
sbatch sbatch/generate_stories.sbatch

# Stage 2: GPU node (vLLM, Qwen3-32B)
sbatch sbatch/generate_endings.sbatch

# Stages 3 & 4: GPU node (can run in parallel after Stage 2)
sbatch sbatch/compute_geometry.sbatch
sbatch sbatch/compute_judge.sbatch
```

## Pipeline Configuration (CLI Flags)

**All ablation settings are command-line flags** — no code edits needed.

### Step 0: Warmup (`--step0`)

Controls how the structural beat sheet reference is generated.

| Flag | Mode | Description |
|---|---|---|
| `--step0 title` | Title (default) | LLM generates beats from a famous story title (20 stories in pool) |
| `--step0 fulltext` | Full text | LLM reads entire NYer story text and extracts beats (20 stories) |
| `--step0 precomputed` | Pre-computed | No API call; selects from 6 pre-generated templates |
| `--step0 none` | Skip | No warmup; beats generated from scratch in step 1 |

**Step 0 options:**
- `--step0-style {baseline,c1_specific,c2_abstract}` — Prompt style (default: `c1_specific`)
- `--step0-source ID` — Fix a specific source (e.g., `A1` for Le Guin, `00015` for NYer). Omit to randomly sample.
- `--step0-sources-dir PATH` — Directory of .txt files for `fulltext` mode

### Step 1: Beat Generation (`--step1`)

Controls how structural beats are created for the writing prompt.

| Flag | Mode | Description |
|---|---|---|
| `--step1 default` | Standard (default) | Adapt warmup beats to the writing prompt |
| `--step1 path/to/prompts.json` | Custom | Use custom prompt templates from a JSON file |
| `--step1 none` | Skip | **No beats → step 0 auto-skipped → step 2 becomes vanilla** |

**Step 1 options:**
- `--step1-max-beats N` — Limit to N beats (e.g., `3` for short beat list)

### Step 2: Story Writing

Always present. If beats exist (from step 1), writes from beats. If no beats, writes from the raw prompt (vanilla).

### Key Constraint

**`--step1 none` forces vanilla mode.** Step 0 is automatically skipped because warmup without beats is meaningless. This is enforced by the code.

## Ablation Recipes

```bash
# ── Full pipeline variants ────────────────────────────────────────
python generate_stories.py --run-id full            --step0 title    --step1 default
python generate_stories.py --run-id fulltext        --step0 fulltext --step1 default
python generate_stories.py --run-id precomputed     --step0 precomputed --step1 default

# ── Ablate step 0 (no warmup) ────────────────────────────────────
python generate_stories.py --run-id no_warmup       --step0 none     --step1 default

# ── Ablate step 1 (no beats → vanilla) ───────────────────────────
python generate_stories.py --run-id vanilla         --step1 none

# ── Modify step 0 style ──────────────────────────────────────────
python generate_stories.py --run-id baseline_style  --step0-style baseline
python generate_stories.py --run-id abstract_style  --step0-style c2_abstract

# ── Short beats ───────────────────────────────────────────────────
python generate_stories.py --run-id short_3beats    --step1-max-beats 3

# ── Fixed warmup source ──────────────────────────────────────────
python generate_stories.py --run-id fixed_leguin    --step0-source A1
python generate_stories.py --run-id fixed_nyer15    --step0 fulltext --step0-source 00015

# ── Custom step 1 prompts ────────────────────────────────────────
python generate_stories.py --run-id custom          --step1 my_prompts.json
```

### Custom Step 1 Prompts (JSON)

Create a JSON file with a list of prompt template strings. Use `{warmup}` and `{writing_prompt}` as placeholders:

```json
[
  "Reference beats:\n---\n{warmup}\n---\n\nCreate a minimalist 3-act structure for:\n{writing_prompt}"
]
```

## Step 0 Reference Stories

### Title Mode (20 famous stories)

| ID | Story | Technique Focus |
|---|---|---|
| A1 | The Ones Who Walk Away from Omelas (Le Guin) | Moral philosophy |
| A2 | The Lottery (Jackson) | Suspense/Reveal |
| A3 | Hills Like White Elephants (Hemingway) | Subtext/Dialogue |
| A4 | The Yellow Wallpaper (Gilman) | Unreliable narrator |
| A5 | A Good Man Is Hard to Find (O'Connor) | Violence/Grace |
| A6 | The Garden of Forking Paths (Borges) | Metafiction/Structure |
| A7 | The Metamorphosis (Kafka) | Surrealism/Premise |
| A8 | Rashomon (Akutagawa) | Subjectivity/Perspective |
| A9 | The Necklace (Maupassant) | Irony/Pacing |
| A10 | The Overcoat (Gogol) | Pathos/Character |
| A11 | Girl (Kincaid) | Form/Prose Rhythm |
| A12 | Axolotl (Cortazar) | Magical Realism/Identity |
| A13 | Cathedral (Carver) | Minimalism/Dirty Realism |
| A14 | The Lady with the Dog (Chekhov) | Modern Realism |
| A15 | Araby (Joyce) | Epiphany/Coming-of-age |
| A16 | A Very Old Man with Enormous Wings (Marquez) | Imagery |
| A17 | Sonny's Blues (Baldwin) | Voice/Narrative Soul |
| A18 | The Fifth Story (Lispector) | Narrative Experimentation |
| A19 | A Madman's Diary (Lu Xun) | Allegory/Social Critique |
| A20 | The Bear Came Over the Mountain (Munro) | Handling of Time/Memory |

### Full-Text Mode (20 New Yorker stories)

Located in `data/nyer_sources/`. These are short stories (1000-2600 words) from the New Yorker corpus. The full text is sent to the LLM for structural extraction.

## Output Structure

```
results/
├── stories/{run_id}/              # Stage 1 output
│   ├── p01_iter1.txt              # Story text
│   ├── p01_iter1_meta.json        # Pipeline intermediates + metadata
│   ├── p01_iter2.txt
│   ├── ...
│   └── _config.json               # CLI config snapshot for reproducibility
├── endings/{run_id}/              # Stage 2 output
│   ├── p01_iter1_endings.json     # 100 endings per position
│   └── ...
├── geometry/{run_id}_geometry.json # Stage 3 output
│   # Per-position: dist_to_actual, intra_mean, intra_std,
│   #               top3_eigenvalues, effective_rank
└── judge/{run_id}_judge.json      # Stage 4 output
    # Per-position: yes_rate, judgments (100 binary values)
```

Each run saves a `_config.json` with the exact CLI flags used, so runs are fully reproducible.

## Metrics

| Metric | Source | What it measures |
|---|---|---|
| **dist_to_actual** | Geometry | Mean cosine distance from 100 endings to true continuation |
| **intra_mean** | Geometry | Mean pairwise cosine distance among 100 endings |
| **intra_std** | Geometry | Std of pairwise distances (diversity variance) |
| **effective_rank** | Geometry | Spectral entropy: 1 = all endings alike, 100 = maximally diverse |
| **top3_eigenvalues** | Geometry | First 3 eigenvalues of the similarity matrix |
| **yes_rate** | Judge | Fraction of 100 endings matching events in the actual continuation |

## Estimated Costs

| Stage | Resource | Approx. Cost / Time |
|---|---|---|
| Stage 1 (per config) | API (Opus 4.6) | ~$5-15 for 32 prompts × 3 iterations |
| Stage 2 (per config) | GPU (H200, Qwen3-32B) | ~4-8 hours |
| Stage 3 (per config) | GPU (H200, Qwen3-Embedding) | ~1-2 hours |
| Stage 4 (per config) | GPU (H200, Qwen3-8B) | ~2-4 hours |

## Relationship to creative-writing-bench

This project imports pipeline classes from [creative-writing-bench](https://github.com/maxzhuyt/creative-writing-bench):
- `core/narrative_pipeline.py` — step classes (`WarmupStep`, `BeatSheetStep`, etc.)
- `utils/api.py` — OpenRouter/Anthropic API client
- `data/creative_writing_prompts_v3.json` — the 32 writing prompts

The creative-writing-bench evaluates stories via **rubric scoring + Elo pairwise matchups**. This project (100-endings) evaluates via **geometric coherence + semantic accuracy**. They are complementary evaluation approaches for the same pipeline.
