#!/usr/bin/env bash
# Launches an Agent Laboratory research run in an isolated directory under runs/.
#
# Usage: scripts/run_lab.sh [config.yaml]   (default: configs/quickstart.yaml)
# Resume a previous run by setting RUN_DIR=runs/<name> and `load-existing: True` in the config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT/third_party/AgentLaboratory"
VENV="${VENV:-$ROOT/.venv}"
CONFIG="$(realpath "${1:-$ROOT/configs/quickstart.yaml}")"

if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "Set OPENAI_API_KEY or DEEPSEEK_API_KEY in .env (or the environment) first." >&2
  exit 1
fi

# Agent Laboratory wipes ./MATH_research_dir at the start of every run, so each run gets its own cwd.
RUN_DIR="${RUN_DIR:-$ROOT/runs/$(date +%Y%m%d-%H%M%S)-$(basename "$CONFIG" .yaml)}"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"
echo "Run directory: $RUN_DIR"

cd "$RUN_DIR"
PYTHONPATH="$LAB${PYTHONPATH:+:$PYTHONPATH}" exec "$VENV/bin/python" "$LAB/ai_lab_repo.py" --yaml-location "$RUN_DIR/config.yaml"
