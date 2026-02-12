"""Experiment 01: Chit-chat filtering of WildChat.

Preprocessing: loads WildChat-1M, filters to English, deduplicates, writes JSONL.
Judging: runs ChitChatJudge to classify each conversation as chit-chat or not.

Usage:
  # Step 1: Preprocess (only needs to run once, can run on CPU)
  uv run python experiments/01_chit_chat_filter/run.py --preprocess

  # Step 2: Judge (needs vLLM server running)
  uv run python experiments/01_chit_chat_filter/run.py --judge

  # Or both:
  uv run python experiments/01_chit_chat_filter/run.py --preprocess --judge
"""

import argparse
import asyncio
import json
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
DATA_FILE = os.path.join(RESULTS_DIR, "wildchat_en.jsonl")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "chit_chat_labels.jsonl")

CACHE_DIR = "/project2/robinjia_875/wangzhu/eric_huang/.cache/huggingface"
DATASET_NAME = "allenai/WildChat-1M"


def preprocess(limit: int | None = None):
    """Download WildChat, filter to English, deduplicate, write JSONL."""
    from datasets import load_dataset

    print(f"Loading {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train", cache_dir=CACHE_DIR)

    # English filter
    dataset = dataset.filter(lambda x: x["language"] == "English")

    # Deduplicate by conversation_hash
    seen = set()
    unique_indices = []
    for i, h in enumerate(dataset["conversation_hash"]):
        if h not in seen:
            seen.add(h)
            unique_indices.append(i)
    dataset = dataset.select(unique_indices)
    print(f"After English + dedup: {len(dataset)} conversations")

    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Limiting to first {len(dataset)} conversations")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        for row in dataset:
            record = {
                "conversation_hash": row["conversation_hash"],
                "model": row["model"],
                "conversation": row["conversation"],
                "timestamp": str(row.get("timestamp", "")),
            }
            f.write(json.dumps(record, default=str) + "\n")
    print(f"Wrote {DATA_FILE}")


def judge(args):
    """Run ChitChatJudge on the preprocessed data."""
    from pipeline.chit_chat import ChitChatJudge
    from pipeline.judge import JudgeConfig

    shard_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    output = OUTPUT_FILE.replace(".jsonl", f"_part_{shard_id}.jsonl")

    config = JudgeConfig(
        model_name=args.model_name,
        input_path=DATA_FILE,
        output_path=output,
    )
    judge = ChitChatJudge(config)
    asyncio.run(judge.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action="store_true", help="Run preprocessing step")
    parser.add_argument("--judge", action="store_true", help="Run judge step")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--limit", type=int, default=None, help="Limit preprocessing to first N examples")
    args = parser.parse_args()

    if not args.preprocess and not args.judge:
        parser.error("Specify --preprocess and/or --judge")

    if args.preprocess:
        preprocess(limit=args.limit)
    if args.judge:
        judge(args)
