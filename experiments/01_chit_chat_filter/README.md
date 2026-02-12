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
# Start vLLM server first (--max-model-len is required for this model):
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --port ${VLLM_PORT:-8001} \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 16384
uv run python experiments/01_chit_chat_filter/run.py --judge

# SLURM array (5 shards)
sbatch --array=0-4 slurm/run_gpu.sbatch experiments/01_chit_chat_filter/run.py --judge
```

## Output Schema

Each line in the output JSONL:
```json
{"conversation_hash": "...", "keep": true, "reasoning": "...", "raw_response": "...", "model": "...", "judge_type": "chit_chat"}
```

## Results (200-example pilot)

- **200** English deduplicated conversations from WildChat-1M
- **9 skipped** by `format_conversation` (first user message < 5 characters)
- **191 judged**, 0 parse errors

| Label | Count | % |
| --- | --- | --- |
| Chit-chat (`keep=true`) | 18 | 9.4% |
| Not chit-chat (`keep=false`) | 173 | 90.6% |

### Observations

- The vast majority (~91%) of WildChat conversations are task-oriented (coding, writing, factual Q&A), not chit-chat. This is consistent with WildChat being drawn from ChatGPT users who typically have a goal in mind.
- Chit-chat examples are mostly simple greetings ("Hi", "Hello there", "How are you?") with no follow-up task.
- The judge produces structured JSON reliably — 0/191 parse failures with `temperature=0`.
- The 9 skipped rows had trivially short first messages (< 5 chars), which is a reasonable filter to avoid ambiguous single-token inputs.
