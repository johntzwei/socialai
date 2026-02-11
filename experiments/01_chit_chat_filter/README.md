# 01 — Chit-Chat Filter

Filter WildChat-1M conversations to identify chit-chat using LLM-as-a-judge (Qwen3-VL-8B via vLLM).

## Steps

1. **Preprocess**: Download WildChat, filter to English, deduplicate → `results/wildchat_en.jsonl`
2. **Judge**: Run ChitChatJudge on each conversation's first user message → `results/chit_chat_labels_part_N.jsonl`

## Usage

```bash
# Preprocess (CPU, one-time)
uv run python experiments/01_chit_chat_filter/run.py --preprocess

# Judge (needs vLLM server on GPU)
# Start vLLM server first: vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8001
uv run python experiments/01_chit_chat_filter/run.py --judge

# SLURM array (5 shards)
sbatch --array=0-4 slurm/run_gpu.sbatch experiments/01_chit_chat_filter/run.py --judge
```

## Output Schema

Each line in the output JSONL:
```json
{"conversation_hash": "...", "keep": true, "reasoning": "...", "raw_response": "...", "model": "...", "judge_type": "chit_chat"}
```
