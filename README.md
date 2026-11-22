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
`requirements-lab.txt`, installs the [Cursor CLI](https://cursor.com/docs/cli), copies
`.env.example` to `.env`, and runs `scripts/smoke_test.py`. `requirements-lab.txt` adds packages
that upstream imports but doesn't list: TensorFlow, tf-keras, Flask, Flask-SQLAlchemy and
sentence-transformers.

Then authenticate the Cursor CLI. Create an API key in the
[Cursor dashboard](https://cursor.com/dashboard) under Integrations, and put it in `.env`:

```bash
CURSOR_API_KEY=...
CURSOR_MODEL=claude-opus-5-5   # any slug from `agent --list-models`; empty = account default
```

Check it with `.venv/bin/python scripts/smoke_test.py --live`, which sends one tiny request.

## LLM backend: Cursor models

Every LLM call goes to a Cursor model, so no OpenAI or DeepSeek key and no local GPU are needed.
That covers both the agents' calls and the calls made by experiment code the agents write.
`lab/launch.py` installs `lab/cursor_backend.py` before running upstream's `ai_lab_repo.py`. The
backend replaces upstream's `inference.query_model` and `utils.query_*` helpers. Each call runs
`agent -p --mode ask` (headless and read-only) in an empty temporary workspace. Prompts larger than
the command-line limit are passed through a file.

- `llm-backend: "cursor"` in a config uses `CURSOR_MODEL`; `"cursor:<slug>"` selects a specific model.
- Generated experiment code calls the model with `query_llm(prompt, system)`.
- The CLI has no temperature setting, so upstream's `temp` / `temperature` arguments are ignored.
- Every call is one Cursor agent request and counts toward your Cursor plan's usage. A full run makes
  hundreds of calls. `CURSOR_MAX_CONCURRENCY` (default 4) limits parallel calls per process.
- `LAB_BACKEND=upstream` in `.env` switches back to upstream's OpenAI / DeepSeek calls.
  Use an upstream model name in `llm-backend` in that case.

## Running research

```bash
scripts/run_lab.sh                          # configs/quickstart.yaml: self-verification prompting on GSM8K
scripts/run_lab.sh configs/my-idea.yaml
```

Each run gets its own folder under `runs/`, because Agent Laboratory deletes its research directory at
the start of every run. Results end up in `runs/<run>/MATH_research_dir/` (code in `src/`, paper in
`tex/`), and checkpoints go in `runs/<run>/state_saves/`. To resume, rerun with `RUN_DIR=runs/<run>` and
`load-existing: True`.

To write your own experiment, copy `configs/quickstart.yaml`, change `research-topic`, `llm-backend` and
`task-notes`. Detailed task notes, including the compute available to the agents, have the biggest
effect on the quality of the results. Set `copilot-mode: True` to steer the agents interactively.
With limited GPU access, keep experiments to model queries and light CPU work, and say so in the
task notes. The quickstart config shows how.

Tests for the Cursor backend use a fake CLI and need no account:
`.venv/bin/python -m unittest discover -s tests`.

## Safety

The ML-engineer agent writes Python and runs it with `exec` in a local subprocess, with no sandbox.
Run the lab in a disposable VM or container, keep secrets you don't want exposed out of the
environment, and set a usage or spending limit on your Cursor or LLM API account.
