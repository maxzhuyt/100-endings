#!/usr/bin/env python3
"""
Stage 3: Compute embedding geometry metrics for generated endings.

For each story in results/endings/{config_id}/, loads the original story text,
splits into sentences, then for each position:
  1. Embeds actual continuation (sentences[pos:]) and all 100 generated endings
  2. Computes dist_to_actual (mean cosine distance from endings to actual)
  3. Computes intra_mean/std (pairwise cosine distances among endings)
  4. Computes top-3 eigenvalues + effective rank of similarity matrix

Output: results/geometry/{config_id}_geometry.json

Usage:
    python compute_geometry.py                      # all configs
    python compute_geometry.py --config full,vanilla # specific configs
"""

import os
import sys
import re
import json
import gc
import time
import argparse
import logging
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = "/project/jevans/maxzhuyt/narrative_ablation"
STORIES_DIR = os.path.join(BASE_DIR, "results", "stories")
ENDINGS_DIR = os.path.join(BASE_DIR, "results", "endings")
GEOMETRY_DIR = os.path.join(BASE_DIR, "results", "geometry")

EMB_MODEL = "/project/jevans/maxzhuyt/models/Qwen3-Embedding-0.6B"
PCT_MIN, PCT_MAX = 10, 99


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_sentences(text):
    boundaries = [0]
    for m in re.finditer(r'[.!?]["\u201d\u2019)]*\s+(?=[A-Z\u201c"(\[])', text):
        boundaries.append(m.end())
    parts = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            parts.append(chunk)
    return parts


def token_pcts(sentences, positions, tokenizer):
    sent_lens = [len(tokenizer.encode(s, add_special_tokens=False))
                 for s in sentences]
    cum = np.cumsum([0] + sent_lens)
    total = cum[-1]
    max_idx = len(cum) - 1
    if total == 0:
        return {p: 0.0 for p in positions}
    return {p: round(float(cum[min(p, max_idx)]) / total * 100, 1)
            for p in positions}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Comma-separated config IDs (default: all)")
    args = parser.parse_args()

    os.makedirs(GEOMETRY_DIR, exist_ok=True)

    # Discover configs
    if args.config:
        config_ids = [c.strip() for c in args.config.split(",")]
    else:
        config_ids = sorted(d for d in os.listdir(ENDINGS_DIR)
                            if os.path.isdir(os.path.join(ENDINGS_DIR, d)))

    log.info(f"Configs: {config_ids}")

    # Load embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Loading embedding model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(EMB_MODEL, trust_remote_code=True,
                                       dtype=torch.float16).to(device).eval()

    def encode(texts, batch_size=256):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
        return np.vstack(all_embs)

    # Process each config
    for config_id in config_ids:
        endings_dir = os.path.join(ENDINGS_DIR, config_id)
        if not os.path.isdir(endings_dir):
            log.warning(f"[{config_id}] No endings dir, skipping")
            continue

        output_path = os.path.join(GEOMETRY_DIR, f"{config_id}_geometry.json")

        # Load checkpoint
        if os.path.exists(output_path):
            geometry = json.load(open(output_path))
            log.info(f"[{config_id}] Resuming from {len(geometry)} entries")
        else:
            geometry = {}

        endings_files = sorted(f for f in os.listdir(endings_dir)
                               if f.endswith("_endings.json"))
        log.info(f"\n[{config_id}] {len(endings_files)} ending files")

        for fi, fname in enumerate(endings_files):
            story_id = fname.replace("_endings.json", "")
            if story_id in geometry:
                continue

            # Load endings
            d = json.load(open(os.path.join(endings_dir, fname)))
            positions = d["positions"]

            # Load original story text
            story_path = os.path.join(STORIES_DIR, config_id, f"{story_id}.txt")
            if not os.path.exists(story_path):
                log.warning(f"  [{story_id}] story text missing, skipping")
                continue

            story_text = open(story_path, encoding="utf-8").read().strip()
            sentences = split_sentences(story_text)
            n_sents = len(sentences)
            pos_list = [p["position"] for p in positions]
            tok_map = token_pcts(sentences, pos_list, tokenizer)

            log.info(f"  [{fi+1}/{len(endings_files)}] {story_id}: "
                     f"{len(positions)} positions, {n_sents} sents")

            position_results = []
            t0 = time.time()

            for pi, p in enumerate(positions):
                pos = p["position"]
                endings = p["endings"]
                n_endings = len(endings)
                pct = tok_map.get(pos, 0.0)

                if pct < PCT_MIN or pct > PCT_MAX:
                    continue

                # Actual continuation
                continuation = " ".join(sentences[min(pos, len(sentences)-1):])

                # Embed
                cont_emb = encode([continuation])
                end_embs = encode(endings)

                # dist_to_actual
                sims = end_embs @ cont_emb.T
                dist_to_actual = float(np.mean(1.0 - sims[:, 0]))

                # Intra-ending pairwise distances
                sim_matrix = end_embs @ end_embs.T
                triu_idx = np.triu_indices(n_endings, k=1)
                pairwise_dists = 1.0 - sim_matrix[triu_idx]
                intra_mean = float(np.mean(pairwise_dists))
                intra_std = float(np.std(pairwise_dists))

                # Eigenvalues + effective rank
                try:
                    eigenvalues = np.linalg.eigvalsh(sim_matrix)
                    eigenvalues = np.sort(eigenvalues)[::-1]
                    top3 = eigenvalues[:3].tolist()
                    pos_eig = eigenvalues[eigenvalues > 0]
                    if len(pos_eig) > 0:
                        p_eig = pos_eig / pos_eig.sum()
                        eff_rank = float(np.exp(-np.sum(
                            p_eig * np.log(p_eig + 1e-12))))
                    else:
                        eff_rank = 1.0
                except Exception:
                    top3 = [0, 0, 0]
                    eff_rank = 1.0

                position_results.append({
                    "position": pos,
                    "pct": pct,
                    "dist_to_actual": dist_to_actual,
                    "intra_mean": intra_mean,
                    "intra_std": intra_std,
                    "top3_eigenvalues": top3,
                    "effective_rank": eff_rank,
                })

            geometry[story_id] = {
                "config_id": config_id,
                "positions": position_results,
            }

            # Checkpoint every 5 stories
            if (fi + 1) % 5 == 0:
                with open(output_path, "w") as f:
                    json.dump(geometry, f)
                log.info(f"    checkpoint ({len(geometry)} stories)")

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Final save for this config
        with open(output_path, "w") as f:
            json.dump(geometry, f)
        log.info(f"[{config_id}] Done. {len(geometry)} stories → {output_path}")

    log.info("\nAll geometry computation complete.")


if __name__ == "__main__":
    main()
