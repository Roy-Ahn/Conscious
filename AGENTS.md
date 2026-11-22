# Agent instructions

This repo is an AI-assisted AI research lab: [Agent Laboratory](https://github.com/SamuelSchmidgall/AgentLaboratory)
(`third_party/AgentLaboratory`, unmodified) with every LLM call routed to a Cursor model (`lab/cursor_backend.py`).
See `README.md` for the design.

## Environment

The Cloud Agent environment runs `scripts/setup_lab.sh --with-latex` as its install step, so `.venv`,
the Cursor CLI (`agent`) and pdflatex are normally ready. If `.venv` is missing, or the branch changed
`requirements-lab.txt` or the submodule, rerun it (idempotent, takes a few minutes).

`CURSOR_API_KEY` comes from the Cursor Cloud Agent secrets. Check everything with:

```bash
.venv/bin/python scripts/smoke_test.py --live
```

If authentication fails, ask the user to add `CURSOR_API_KEY` under Cursor Dashboard > Cloud Agents > Secrets
(key from Dashboard > Integrations > API Keys). Do not ask for the key in chat and never commit it.

## Running research for the user

1. Write the user's idea into a new config by copying `configs/quickstart.yaml`. Set `research-topic`
   and `task-notes`. Keep `copilot-mode: False`, because nobody is at a terminal to answer prompts. The VM is
   CPU-only (no GPU): tell the agents so in `task-notes` and keep experiments to `query_llm()` calls and
   light CPU work.
2. Run it in tmux, because runs take from tens of minutes to hours:

   ```bash
   tmux new-session -d -s lab "scripts/run_lab.sh configs/<name>.yaml"
   ```

   Follow progress in `runs/<newest>/run.log` (phases are logged as `Beginning phase: ...`). Wait with
   process-bound waits, not tight polling.
3. When the run ends (or fails), run `scripts/export_run.sh`. Commit `results/<run>/` and the config on your
   branch, push, and open or update a PR. The user reads results on GitHub, often from a phone.
4. In the reply, lead with the outcome: the research question, the key numbers from the
   results-interpretation phase, and links to the report: `results/<run>/research_dir_*/report.txt` and
   `readme.md`, plus `tex/temp.pdf` when LaTeX compiled. Experiment code and output are in `src/` and figures
   are `Figure_*.png`. Report failures honestly, with the failing phase and the last lines of `run.log`.

## Development

- Tests (fake CLI, no account needed): `.venv/bin/python -m unittest discover -s tests`
- Do not edit `third_party/AgentLaboratory`; adapt behavior in `lab/` instead.
- The ML-engineer agent `exec`s generated code without a sandbox. Keep runs inside the Cloud Agent VM.
