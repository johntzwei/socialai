"""Experiment 02: Anthropomorphic behavior detection in WildChat.

Takes chit-chat-positive conversations from experiment 01 and judges whether
the AI assistant exhibits anthropomorphic behavior.

Usage:
  # First, filter experiment 01 output to chit-chat-only conversations
  uv run python experiments/02_anthropomorphic_judge/run.py --prepare

  # Then run the anthropomorphic judge (needs vLLM server)
  uv run python experiments/02_anthropomorphic_judge/run.py --judge

  # Full conversation mode (all turns instead of first turn only)
  uv run python experiments/02_anthropomorphic_judge/run.py --judge --all_turns
"""

import argparse
import asyncio
import glob
import json
import os

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
EXP01_RESULTS = os.path.join(os.path.dirname(EXPERIMENT_DIR), "01_chit_chat_filter", "results")

# Input: chit-chat conversations extracted from exp 01
INPUT_FILE = os.path.join(RESULTS_DIR, "chit_chat_conversations.jsonl")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "anthropomorphic_labels.jsonl")


def prepare():
    """Merge exp 01 labels with original data to produce chit-chat-only input."""
    # Load chit-chat labels from exp 01 (may be sharded)
    label_files = glob.glob(os.path.join(EXP01_RESULTS, "chit_chat_labels*.jsonl"))
    keep_hashes = set()
    for lf in label_files:
        with open(lf) as f:
            for line in f:
                record = json.loads(line)
                if record.get("keep") is True:
                    keep_hashes.add(record["conversation_hash"])

    print(f"Found {len(keep_hashes)} chit-chat conversations from exp 01")

    # Load original conversations and filter to kept ones
    wildchat_file = os.path.join(EXP01_RESULTS, "wildchat_en.jsonl")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    count = 0
    with open(wildchat_file) as fin, open(INPUT_FILE, "w") as fout:
        for line in fin:
            row = json.loads(line)
            if row["conversation_hash"] in keep_hashes:
                fout.write(line)
                count += 1

    print(f"Wrote {count} chit-chat conversations to {INPUT_FILE}")


def judge(args):
    """Run AnthropomorphicJudge on the chit-chat conversations."""
    from pipeline.anthropomorphic import AnthropomorphicJudge
    from pipeline.judge import JudgeConfig

    shard_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    output = OUTPUT_FILE.replace(".jsonl", f"_part_{shard_id}.jsonl")

    config = JudgeConfig(
        model_name=args.model_name,
        input_path=INPUT_FILE,
        output_path=output,
    )
    judge = AnthropomorphicJudge(config, all_turns=args.all_turns)
    asyncio.run(judge.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare input from exp 01 chit-chat labels")
    parser.add_argument("--judge", action="store_true", help="Run anthropomorphic judge")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--all_turns", action="store_true", help="Send full conversation instead of first turn only")
    args = parser.parse_args()

    if not args.prepare and not args.judge:
        parser.error("Specify --prepare and/or --judge")

    if args.prepare:
        prepare()
    if args.judge:
        judge(args)
