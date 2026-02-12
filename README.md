# socialai

Studying anthropomorphic behavior in LLMs. We filter and analyze large-scale conversation datasets (WildChat) using LLM-as-a-judge pipelines to identify when AI assistants present themselves as having human-like qualities.

## Setup

```bash
bash install_uv.sh   # if uv not installed
uv sync
```

## Usage

```bash
# Run an experiment
uv run python experiments/01_chit_chat_filter/run.py --preprocess --judge

# Submit to SLURM
sbatch slurm/run_gpu.sbatch experiments/01_chit_chat_filter/run.py --judge
sbatch slurm/run_preempt.sbatch experiments/01_chit_chat_filter/run.py --judge

# Array jobs (sharded across GPUs)
sbatch --array=0-4 slurm/run_gpu.sbatch experiments/01_chit_chat_filter/run.py --judge

# Run tests
uv run pytest
```

## Project Structure

```text
src/allegro/       # Shared library code (reusable modules)
src/pipeline/      # LLM-as-a-judge pipeline (base class, judges)
experiments/       # Numbered experiment scripts
slurm/             # SLURM job templates
tests/             # Tests
```

## Pipeline

The judge pipeline (`src/pipeline/`) uses a vLLM server with async inference, SLURM array sharding, and automatic resumption. Two judges are implemented:

1. **Chit-chat filter** (`pipeline.chit_chat`) — classifies whether a conversation is casual chit-chat vs. task-oriented
2. **Anthropomorphic judge** (`pipeline.anthropomorphic`) — detects whether an AI assistant claims emotions, experiences, identity, or consciousness
