#!/bin/bash
# Generate instruction-vanilla and idea_instruction-vanilla stories
# with automatic retry for unqualified word counts.

set -e
cd /project/jevans/maxzhuyt/narrative_ablation
source ~/.bashrc
conda activate /project/jevans/maxzhuyt/honest_llama_env

export TEST_API_KEY=$(cat /project/jevans/maxzhuyt/credential)
export TEST_API_URL="https://openrouter.ai/api/v1/chat/completions"
export PYTHONUNBUFFERED=1

echo "=== instruction-vanilla (fixed ideas) ==="
python generate_stories.py \
    --run-id instruction_vanilla \
    --step1 none --vanilla-instruction \
    --ideas data/story_ideas.json \
    --iterations 1 --threads 6 \
    --model anthropic/claude-opus-4-6

echo "=== idea_instruction_vanilla (generated ideas) ==="
python generate_stories.py \
    --run-id idea_instruction_vanilla \
    --idea-gen --step1 none --vanilla-instruction \
    --ideas data/story_ideas.json \
    --iterations 1 --threads 6 \
    --model anthropic/claude-opus-4-6

echo "=== Retrying unqualified stories ==="
# Retry Phase 3 for idea_instruction_vanilla
python retry_unqualified.py --phase 3 \
    --model anthropic/claude-opus-4-6

echo "=== Done ==="
