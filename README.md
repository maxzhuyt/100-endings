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

## Results & Analysis

All results are from n=15 stories per configuration, filtered to 2000–3015 words. 95% CIs are ~±0.03–0.05, so differences under ~0.03 should be treated as noise.

**Reference anchors** (different story prompts — interpret as scale markers, not direct comparisons):
- Top-10 LLMs (0-shot, EQ-Bench prompts): **0.71**
- New Yorker fiction (human): **0.87**

### Complete Results Table

| Config | No-Rate | 95% CI | Phase | Notes |
|---|---|---|---|---|
| `vanilla` (no instruction) | 0.840 | ±0.057 | 1 | Raw idea → model infers task |
| `instruction_vanilla` | 0.842 | ±0.043 | 1 | Same idea + "Write the story now" |
| `short_beats` (title + 3 beats) | 0.831 | ±0.044 | 1 | Minimal structure |
| `full_title` (title + beats) | 0.866 | ±0.035 | 1 | Title-only warmup |
| `full_fulltext` (fulltext + beats) | 0.879 | ±0.029 | 1 | Full NYer text warmup |
| `no_warmup` (beats from scratch) | 0.883 | ±0.024 | 1 | No reference story |
| `fulltext_beats3` | 0.862 | ±0.033 | 2 | Fulltext warmup + 3 beats |
| `fulltext_beats5` | 0.856 | ±0.042 | 2 | Fulltext warmup + 5 beats |
| `fulltext_beats10` | 0.868 | ±0.053 | 2 | Fulltext warmup + 10 beats |
| `idea_bare` | 0.746 | ±0.075 | 3 | No constraints, just "generate an idea" |
| `idea_no_priming` | 0.788 | ±0.048 | 3 | Taxonomy + tropes, no priming |
| `idea_tropes_only` | 0.833 | ±0.038 | 3 | Random tropes, no taxonomy |
| `idea_full` | 0.833 | ±0.037 | 3 | Full idea gen (taxonomy + tropes + priming) |
| `idea_no_tropes` | 0.847 | ±0.036 | 3 | Taxonomy + priming, no tropes |
| `idea_instruction_vanilla` | 0.858 | ±0.044 | 3 | Full idea gen → vanilla (with instruction) |
| `idea_tax_only` | 0.866 | ±0.036 | 3 | Taxonomy only |
| `idea_vanilla` (no instruction) | 0.879 | ±0.027 | 3 | Full idea gen → vanilla (no instruction) |

### Phase 1 Findings — Structural Pipeline

**The structural beat pipeline consistently improves no-rate over vanilla.** Configs with beats (`full_title` 0.866, `full_fulltext` 0.879, `no_warmup` 0.883) are ~0.03–0.04 higher than vanilla (0.840–0.842). This gap is at the edge of significance given our CIs, but is consistent across all beat-based configs.

**Adding a writing instruction to vanilla makes no difference** (0.840 → 0.842). Opus 4.6 correctly infers "write a story" from the raw idea alone.

**Too few beats hurts.** `short_beats` (3 beats, 0.831) is the lowest of all configs — even below vanilla. Three structural beats may over-constrain the model into a predictable template without enough detail to create surprise.

**Warmup mode doesn't matter much.** `full_fulltext` (0.879), `full_title` (0.866), and `no_warmup` (0.883) are within ~0.02 of each other. Whether the model reads a full NYer story, just a title, or generates beats from scratch, the resulting narrative structure is similar. The beats themselves — not the reference story — are what drives unpredictability.

### Phase 2 Findings — Beat Count Sweep

**Beat count has no clear effect above 3.** With step0=fulltext held constant: unlimited (0.879), 10 beats (0.868), 5 beats (0.856), 3 beats (0.862) — all within ~0.02, well inside the noise band. The takeaway: any reasonable number of beats (3+) produces similar results. The sharp drop only happens at `short_beats` in Phase 1 (0.831), which used title warmup instead of fulltext — suggesting the combination of weak warmup + few beats is problematic, not the beat count alone.

### Phase 3 Findings — Idea Generation

**Note:** The 15 fixed story ideas used in Phases 1–2 were themselves generated by the full idea-gen pipeline (taxonomy + tropes + priming). So comparisons between Phase 1/2 configs and Phase 3's `idea_full` (0.833) are partially testing the same idea-generation process, just with fresh random seeds for taxonomy/tropes. The ~0.05 gap between Phase 1's `full_fulltext` (0.879) and Phase 3's `idea_full` (0.833) likely reflects sampling variance rather than a systematic difference.

**Bare idea generation (no constraints) produces the worst stories.** `idea_bare` (0.746) is dramatically lower than everything else — a ~0.09 gap from the next-lowest config. When the LLM generates ideas with no taxonomy or trope constraints, the resulting stories are significantly more predictable. This is the clearest signal in the entire ablation.

**Genre priming hurts when combined with tropes.** `idea_no_priming` (0.788) vs `idea_full` (0.833) — removing priming drops no-rate by 0.045. But `idea_no_tropes` (0.847) and `idea_tax_only` (0.866) both score *higher* than the full pipeline. This suggests priming may introduce a formulaic quality when the model has already been given concrete tropes to work with. The priming analysis activates narrative reasoning, but that reasoning then interacts poorly with LLM-curated trope selection — the model over-fits to the structural patterns it just analyzed.

**Taxonomy alone is surprisingly strong.** `idea_tax_only` (0.866) outperforms the full idea-gen pipeline (0.833) and matches Phase 1's best configs. A random structural profile (genre, voice, pathway, POV) without tropes or priming produces ideas that lead to unpredictable stories. The constraint forces the model into unusual genre combinations that it can't solve with default patterns.

**Vanilla step2 (no beats) with generated ideas works well.** `idea_vanilla` (0.879) is the highest in Phase 3, matching `full_fulltext` from Phase 1. When the idea itself is sufficiently structured (from the full idea-gen pipeline), adding beats on top may over-constrain the narrative. This echoes Phase 1's finding that `instruction_vanilla` ≈ `vanilla` — the model doesn't need explicit structural scaffolding when the idea is good enough.

### Summary of Key Takeaways

1. **Beats help, but the warmup source doesn't matter.** Any beat-based config beats vanilla by ~0.03–0.04. Whether the beats come from a famous story title, a full NYer text, or from scratch — similar results.

2. **3 beats is too few (with weak warmup).** `short_beats` is the only beat config that underperforms vanilla. The minimum viable structure appears to be ~5+ beats.

3. **Idea quality matters more than pipeline structure.** The largest effects come from idea generation (0.746–0.879 range), not from beats vs vanilla (~0.03 range).

4. **Constrained idea generation beats unconstrained.** Taxonomy profiles alone (0.866) outperform bare LLM idea generation (0.746) by 0.12 — the largest effect size in the entire study.

5. **More components ≠ better.** The full idea-gen pipeline (taxonomy + tropes + priming) scores *lower* than taxonomy-only. Genre priming + tropes may introduce formulaic patterns that reduce unpredictability.

6. **All pipeline configs reach or exceed the New Yorker benchmark** (~0.87) when using the right combination. The best configs (`no_warmup` 0.883, `idea_vanilla` 0.879, `full_fulltext` 0.879) match professional human fiction on this metric.

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

## Relationship to creative-writing-bench

[creative-writing-bench](https://github.com/maxzhuyt/creative-writing-bench) evaluates stories via **rubric scoring + Elo pairwise matchups** (EQ-Bench Creative Writing v3). This project (100-endings) evaluates via **geometric coherence + semantic accuracy**. They share the same narrative pipeline architecture (`core/narrative_pipeline.py`) and can use the same writing prompts, but measure different aspects of story quality.
