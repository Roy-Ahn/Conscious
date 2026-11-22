# Conscious

An AI-assisted AI research laboratory, built on
[Agent Laboratory](https://github.com/SamuelSchmidgall/AgentLaboratory)
([paper](https://arxiv.org/abs/2501.04227), MIT license). LLM agents (PhD student, postdoc,
professor, ML engineer, SW engineer and reviewers) take a research idea through three phases:

1. **Literature review**: search and summarize arXiv papers.
2. **Experimentation**: plan, prepare data, write and run experiment code, interpret results.
3. **Report writing**: draft and review a LaTeX paper.

In co-pilot mode you stay in the loop and give feedback between phases.

The upstream code is pinned as a git submodule in `third_party/AgentLaboratory` and is not modified.

## Why this reference

| Candidate | Fit |
| --- | --- |
| [SamuelSchmidgall/AgentLaboratory](https://github.com/SamuelSchmidgall/AgentLaboratory) | **Chosen.** Human-in-the-loop research assistant for ML research; runs on CPU with a hosted LLM; MIT license. |
| [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) / [v2](https://github.com/SakanaAI/AI-Scientist-v2) | Fully autonomous paper generation; templates need NVIDIA GPUs; custom license. |
| [SakanaAI/AI-Scientist-v3](https://github.com/SakanaAI/AI-Scientist-v3) | Orchestrates external coding agents; newer and less settled. |

## Setup

Requires Python 3.12 (Debian/Ubuntu: `sudo apt-get install python3.12-venv`).

```bash
git clone --recurse-submodules <this repo> && cd <repo>
scripts/setup_lab.sh               # add --with-latex for PDF papers, --gpu for CUDA PyTorch
```

The script creates `.venv`, installs a CPU-only PyTorch build plus upstream's requirements and
`requirements-lab.txt`, copies `.env.example` to `.env`, and runs `scripts/smoke_test.py`.
`requirements-lab.txt` adds packages that upstream imports but doesn't list: TensorFlow,
tf-keras, Flask, Flask-SQLAlchemy and sentence-transformers.

Then put at least one key in `.env`:

```bash
OPENAI_API_KEY=sk-...      # or DEEPSEEK_API_KEY=...
```

## Running research

```bash
scripts/run_lab.sh                          # configs/quickstart.yaml: a small self-consistency study on GSM8K
scripts/run_lab.sh configs/my-idea.yaml
```

Each run gets its own folder under `runs/`, because Agent Laboratory deletes its research directory at
the start of every run. Results end up in `runs/<run>/MATH_research_dir/` (code in `src/`, paper in
`tex/`), and checkpoints go in `runs/<run>/state_saves/`. To resume, rerun with `RUN_DIR=runs/<run>` and
`load-existing: True`.

To write your own experiment, copy `configs/quickstart.yaml`, change `research-topic`, `llm-backend` and
`task-notes`. Detailed task notes, including the compute available to the agents, have the biggest
effect on the quality of the results. Set `copilot-mode: True` to steer the agents interactively.

## Safety

The ML-engineer agent writes Python and runs it with `exec` in a local subprocess, with no sandbox.
Run the lab in a disposable VM or container, keep secrets you don't want exposed out of the
environment, and set a spending limit on your LLM API key.
