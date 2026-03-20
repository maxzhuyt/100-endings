# 100-Endings: Narrative Pipeline Ablation via Ending Geometry

Evaluates creative writing pipeline configurations by measuring **how predictable and coherent generated stories are** using 100-ending geometry — a method that generates 100 possible continuations at each position in a story and analyzes their geometric properties in embedding space.

## How It Works

```
Stage 1                    Stage 2                   Stage 3
Generate Stories           Generate 100 Endings      LLM Judge
(API: Opus 4.6)           (vLLM: Qwen3-32B)        (Qwen3-8B)

 prompt + pipeline  ►  story text  ►  100 endings  ►  yes_rate (no_rate)
 config (CLI flags)     per position     per position    per ending
```

**Core metric — no_rate:** At each sentence position in a story, we generate 100 possible endings and ask an LLM judge: "Does this ending match what actually happens?" The **no_rate** (fraction of endings that do NOT match) measures narrative unpredictability. Higher no_rate = less predictable story = more human-like narrative structure.

**Reference anchors:**
- Top-10 LLMs (0-shot, EQ-Bench 32 prompts): ~0.71 no_rate
- New Yorker fiction (human): ~0.87 no_rate
- Our pipeline configs: 0.83–0.88 no_rate

## Full Ablation Design

### Phase 1 — Structural Pipeline Ablations

**Controlled for:** Fixed 15 story ideas (3 iterations each, best qualified selected) | Model: Opus 4.6

Tests the core pipeline components: warmup modes and beat generation.

| Run ID | Step 0 (Warmup) | Step 1 (Beats) | Step 2 (Story) | CLI |
|---|---|---|---|---|
| `vanilla` | — | — | raw idea (no instruction) | `--step1 none` |
| `instruction_vanilla` | — | — | idea + writing instruction | `--step1 none --vanilla-instruction` |
| `full_title` | title warmup | full beats | write from beats | `--step0 title` |
| `full_fulltext` | NYer fulltext warmup | full beats | write from beats | `--step0 fulltext` |
| `no_warmup` | — | beats from scratch | write from beats | `--step0 none` |
| `short_beats` | title warmup | 3 beats only | write from beats | `--step0 title --step1-max-beats 3` |

### Phase 2 — Beat Count Sweep

**Controlled for:** Fixed 15 story ideas (1 iteration) | Model: Opus 4.6 | Step 0: fulltext warmup (c1_specific)

Holds step0=fulltext constant, varies beat count.

| Run ID | Max Beats | CLI |
|---|---|---|
| `full_fulltext` | unlimited | *(from Phase 1)* |
| `fulltext_beats10` | 10 | `--step0 fulltext --step1-max-beats 10` |
| `fulltext_beats5` | 5 | `--step0 fulltext --step1-max-beats 5` |
| `fulltext_beats3` | 3 | `--step0 fulltext --step1-max-beats 3` |

### Phase 3 — Idea Generation Ablations

**Controlled for:** Model: Opus 4.6 | Step 0: fulltext warmup | Step 1: full beats (except vanilla configs)

Tests whether LLM-generated story ideas (from taxonomy + tropes + genre priming) produce better narrative structure than fixed ideas.

| Run ID | Taxonomy | Tropes | Priming | Step 1 | CLI |
|---|---|---|---|---|---|
| `idea_full` | ✓ | ✓ (LLM-curated) | ✓ | full beats | `--idea-gen` |
| `idea_no_priming` | ✓ | ✓ (LLM-curated) | ✗ | full beats | `--idea-gen --no-genre-priming` |
| `idea_no_tropes` | ✓ | ✗ | ✓ | full beats | `--idea-gen --no-tropes` |
| `idea_tax_only` | ✓ | ✗ | ✗ | full beats | `--idea-gen --no-genre-priming --no-tropes` |
| `idea_tropes_only` | ✗ | ✓ (random 2) | ✗ | full beats | `--idea-gen --no-taxonomy` |
| `idea_bare` | ✗ | ✗ | ✗ | full beats | `--idea-gen --no-taxonomy --no-tropes` |
| `idea_vanilla` | ✓ | ✓ | ✓ | **none** | `--idea-gen --step1 none` |
| `idea_instruction_vanilla` | ✓ | ✓ | ✓ | **none (with instr)** | `--idea-gen --step1 none --vanilla-instruction` |

**Idea generation components:**
- **Taxonomy**: Random 7-level narrative profile from the Screenwriters Taxonomy (type, super-genre, macro/micro genre, voice, pathway, POV)
- **Tropes**: Sample tropes from 10 aspect categories, then LLM selects 2 most relevant for the taxonomy profile
- **Genre Priming**: LLM analyzes 5 random taxonomy profiles to activate narrative reasoning (cached once per run, shared across all 15 stories)

**Dependencies:** Taxonomy off → priming also off. Taxonomy off + tropes on → tropes randomly selected (no profile to match against).

## Prerequisites

- Python environment with `requests`, `python-dotenv`, `numpy`, `torch`, `transformers`
- vLLM environment for Stages 2-3 (GPU required, H200 recommended)
- API key for OpenRouter in `.env`:
  ```
  TEST_API_KEY=sk-or-...
  TEST_API_URL=https://openrouter.ai/api/v1/chat/completions
  ```

## Quick Start

### 1. Generate stories (CPU, API calls)

```bash
# Full pipeline: fulltext warmup → beats → story
python generate_stories.py --run-id full_fulltext --step0 fulltext \
    --ideas data/story_ideas.json --iterations 1 --threads 6 \
    --model anthropic/claude-opus-4-6

# Vanilla with writing instruction
python generate_stories.py --run-id instruction_vanilla --step1 none --vanilla-instruction \
    --ideas data/story_ideas.json --iterations 1 --threads 6 \
    --model anthropic/claude-opus-4-6

# Idea generation (full)
python generate_stories.py --run-id idea_full --idea-gen \
    --ideas data/story_ideas.json --iterations 1 --threads 6 \
    --model anthropic/claude-opus-4-6
```

### 2. Retry unqualified stories (outside 2000-3000 words)

```bash
# Dry run: see what needs retrying
python retry_unqualified.py --dry-run

# Run retries (Phase 2: re-runs step2; Phase 3: generates new stories)
python retry_unqualified.py --phase 2 --phase 3
```

### 3. Generate 100 endings (GPU, Qwen3-32B on H200)

```bash
python generate_endings.py --config full_fulltext,vanilla --batch-size 64
# or via SLURM:
sbatch sbatch/endings_ph3_gpu1.sbatch
```

### 4. LLM Judge (GPU, Qwen3-8B)

```bash
python compute_judge.py --config full_fulltext,vanilla --batch-size 128
# or via SLURM (chain after endings):
sbatch --dependency=afterok:<endings_job_id> sbatch/judge_ph3_gpu1.sbatch
```

### 5. Analyze results

Open `notebooks/norate_by_phase.ipynb` — produces one bar chart per phase with 95% CI and reference baselines.

## Pipeline Configuration (CLI Flags)

All ablation settings are command-line flags — no code edits needed.

### Step 0: Warmup (`--step0`)

| Flag | Description |
|---|---|
| `--step0 title` | LLM generates beats from a famous story title (20 in pool, default) |
| `--step0 fulltext` | LLM reads entire NYer story text and extracts beats (20 stories) |
| `--step0 precomputed` | Use 6 pre-generated templates (no API call) |
| `--step0 none` | Skip warmup; beats generated from scratch in step 1 |
| `--step0-style {baseline,c1_specific,c2_abstract}` | Prompt style (default: `c1_specific`) |
| `--step0-source ID` | Fix a specific source (e.g., `A1`, `00015`). Omit for random. |

### Step 1: Beat Generation (`--step1`)

| Flag | Description |
|---|---|
| `--step1 default` | Standard beat adaptation prompt (default) |
| `--step1 path.json` | Custom prompt templates from JSON |
| `--step1 none` | Skip beats → step 0 auto-skipped → step 2 becomes vanilla |
| `--step1-max-beats N` | Limit to N beats (e.g., 3 for short beat list) |

### Step 2: Story Writing

Always present. Writes from beats if available, or directly from the idea (vanilla mode).

- `--vanilla-instruction` — When step1=none, wraps the idea in "Write the story now. Write only the story — no commentary." matching the instruction used by `StoryWriteStep`.

### Idea Generation (`--idea-gen`)

| Flag | Description |
|---|---|
| `--idea-gen` | Enable story idea generation (taxonomy + tropes + priming) |
| `--no-genre-priming` | Disable genre priming |
| `--no-taxonomy` | Disable taxonomy profile (also disables priming) |
| `--no-tropes` | Disable trope selection |
| `--tropes-path PATH` | Custom tropes JSONL file |

## Step 0 Reference Stories

### Title Mode (20 famous stories)

| ID | Story | Technique |
|---|---|---|
| A1 | The Ones Who Walk Away from Omelas (Le Guin) | Moral philosophy |
| A2 | The Lottery (Jackson) | Suspense/Reveal |
| A3 | Hills Like White Elephants (Hemingway) | Subtext/Dialogue |
| A4 | The Yellow Wallpaper (Gilman) | Unreliable narrator |
| A5 | A Good Man Is Hard to Find (O'Connor) | Violence/Grace |
| A6 | The Garden of Forking Paths (Borges) | Metafiction |
| A7 | The Metamorphosis (Kafka) | Surrealism |
| A8 | Rashomon (Akutagawa) | Perspective |
| A9 | The Necklace (Maupassant) | Irony |
| A10 | The Overcoat (Gogol) | Pathos |
| A11 | Girl (Kincaid) | Form/Rhythm |
| A12 | Axolotl (Cortazar) | Magical Realism |
| A13 | Cathedral (Carver) | Minimalism |
| A14 | The Lady with the Dog (Chekhov) | Realism |
| A15 | Araby (Joyce) | Epiphany |
| A16 | A Very Old Man with Enormous Wings (Marquez) | Imagery |
| A17 | Sonny's Blues (Baldwin) | Voice |
| A18 | The Fifth Story (Lispector) | Experimentation |
| A19 | A Madman's Diary (Lu Xun) | Allegory |
| A20 | The Bear Came Over the Mountain (Munro) | Time/Memory |

### Full-Text Mode (20 New Yorker stories)

Located in `data/nyer_sources/`. Short stories (1000-2600 words) from the New Yorker corpus. Full text sent to the LLM for structural extraction.

## Output Structure

```
results/
├── stories/{run_id}/              # Stage 1: generated stories
│   ├── t1_001_iter1.txt           # Story text
│   ├── t1_001_iter1_meta.json     # Pipeline intermediates + metadata
│   └── _config.json               # CLI config snapshot
├── endings/{run_id}/              # Stage 2: 100 endings per position
│   └── t1_001_iter1_endings.json
└── judge/{run_id}_judge.json      # Stage 3: yes_rate + binary judgments
```

## Word Count Filtering

Stories outside the 2000-3000 word range are filtered from analysis (but kept on disk). The `retry_unqualified.py` script regenerates unqualified stories:
- **Phase 1** (3 iterations): selects best qualified iteration per idea → n=15
- **Phase 2** (fixed ideas): re-runs step 2 only (reuses warmup + beats from metadata)
- **Phase 3** (idea-gen): generates entirely new stories to fill gaps

## SLURM Batch Jobs

Endings require H200 GPUs (Qwen3-32B OOMs on H100 80GB).

```bash
# Endings (H200, jevans-gpu)
sbatch sbatch/endings_ph3_gpu1.sbatch    # idea_full, idea_no_priming, idea_no_tropes
sbatch sbatch/endings_ph3_gpu2.sbatch    # idea_no_taxonomy, idea_tax_only, idea_tropes_only
sbatch sbatch/endings_ph3_gpu3.sbatch    # idea_bare, idea_vanilla

# Judge (H200, jevans-gpu — chain after endings)
sbatch --dependency=afterok:<id> sbatch/judge_ph3_gpu1.sbatch
sbatch --dependency=afterok:<id> sbatch/judge_ph3_gpu2.sbatch
sbatch --dependency=afterok:<id> sbatch/judge_ph3_gpu3.sbatch
```

## Relationship to creative-writing-bench

[creative-writing-bench](https://github.com/maxzhuyt/creative-writing-bench) evaluates stories via **rubric scoring + Elo pairwise matchups** (EQ-Bench Creative Writing v3). This project (100-endings) evaluates via **geometric coherence + semantic accuracy**. They share the same narrative pipeline architecture (`core/narrative_pipeline.py`) and can use the same writing prompts, but measure different aspects of story quality.
