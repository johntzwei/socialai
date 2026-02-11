# 02 — Anthropomorphic Behavior Judge

Detect anthropomorphic behavior in WildChat chit-chat conversations using LLM-as-a-judge.

## Prerequisites

Run experiment 01 first to produce chit-chat labels.

## Steps

1. **Prepare**: Merge exp 01 labels with original data → `results/chit_chat_conversations.jsonl`
2. **Judge**: Run AnthropomorphicJudge → `results/anthropomorphic_labels_part_N.jsonl`

## Usage

```bash
# Prepare input (CPU)
uv run python experiments/02_anthropomorphic_judge/run.py --prepare

# Judge first turn only (default)
uv run python experiments/02_anthropomorphic_judge/run.py --judge

# Judge full conversation
uv run python experiments/02_anthropomorphic_judge/run.py --judge --all_turns

# SLURM array
sbatch --array=0-4 slurm/run_gpu.sbatch experiments/02_anthropomorphic_judge/run.py --judge
```

## Output Schema

```json
{"conversation_hash": "...", "anthropomorphic": true, "reasoning": "...", "categories": [1, 5], "raw_response": "...", "model": "...", "judge_type": "anthropomorphic"}
```

Categories: 1=emotions, 2=experiences, 3=physical, 4=identity, 5=relationships, 6=consciousness, 7=first-person narratives
