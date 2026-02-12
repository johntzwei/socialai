"""Initialize annotations CSV and compute inter-annotator agreement.

Usage:
  uv run python experiments/01_chit_chat_filter/score.py --init   # create annotations.csv
  uv run python experiments/01_chit_chat_filter/score.py           # compute agreement
"""

import argparse
import json
import os

import pandas as pd

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(EXPERIMENT_DIR, "results")
LABELS_FILE = os.path.join(RESULTS_DIR, "chit_chat_labels_part_0.jsonl")
DATA_FILE = os.path.join(RESULTS_DIR, "wildchat_en.jsonl")
CSV_FILE = os.path.join(RESULTS_DIR, "annotations.csv")


def init_csv():
    """Create annotations.csv from judge results + source conversations."""
    # Load conversations for first messages
    convos = {}
    with open(DATA_FILE) as f:
        for line in f:
            row = json.loads(line)
            msgs = row.get("conversation", [])
            first_msg = msgs[0]["content"].strip() if msgs else ""
            if len(first_msg) > 300:
                first_msg = first_msg[:300] + "..."
            convos[row["conversation_hash"]] = first_msg

    # Load judge labels
    records = []
    with open(LABELS_FILE) as f:
        for line in f:
            row = json.loads(line)
            h = row["conversation_hash"]
            records.append({
                "conversation_hash": h,
                "first_message": convos.get(h, ""),
                "judge_label": "chit_chat" if row.get("keep") else "not_chit_chat",
                "judge_reasoning": row.get("reasoning", ""),
                "human_label": "",
                "claude_label": "",
                "notes": "",
            })

    df = pd.DataFrame(records)
    df.to_csv(CSV_FILE, index=False)
    print(f"Wrote {len(df)} rows to {CSV_FILE}")


def cohens_kappa(col_a, col_b):
    """Compute Cohen's kappa between two label columns."""
    labels = sorted(set(col_a) | set(col_b))
    n = len(col_a)
    agree = sum(a == b for a, b in zip(col_a, col_b))
    p_o = agree / n

    # Expected agreement by chance
    p_e = sum(
        (sum(a == l for a in col_a) / n) * (sum(b == l for b in col_b) / n)
        for l in labels
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def score():
    """Compute pairwise agreement and Cohen's kappa."""
    df = pd.read_csv(CSV_FILE)

    pairs = [
        ("judge_label", "human_label"),
        ("judge_label", "claude_label"),
        ("human_label", "claude_label"),
    ]

    for col_a, col_b in pairs:
        mask = (df[col_a] != "") & (df[col_b] != "") & df[col_a].notna() & df[col_b].notna()
        subset = df[mask]
        if len(subset) == 0:
            print(f"{col_a} vs {col_b}: no overlapping annotations yet")
            continue

        agree = (subset[col_a] == subset[col_b]).sum()
        total = len(subset)
        kappa = cohens_kappa(subset[col_a].tolist(), subset[col_b].tolist())
        print(f"{col_a} vs {col_b}: {agree}/{total} agree ({agree/total:.1%}), kappa={kappa:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true", help="Create annotations.csv from judge results")
    args = parser.parse_args()

    if args.init:
        init_csv()
    else:
        if not os.path.exists(CSV_FILE):
            print(f"No annotations file found. Run with --init first.")
        else:
            score()
