#!/usr/bin/env bash
# Launches an Agent Laboratory research run in an isolated directory under runs/.
#
# Usage: scripts/run_lab.sh [config.yaml]   (default: configs/quickstart.yaml)
# LLM calls go to a Cursor model by default (CURSOR_API_KEY / CURSOR_MODEL in .env);
# LAB_BACKEND=upstream uses upstream's OpenAI / DeepSeek APIs instead.
# Resume a previous run by setting RUN_DIR=runs/<name> and `load-existing: True` in the config.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT/third_party/AgentLaboratory"
VENV="${VENV:-$ROOT/.venv}"
CONFIG="$(realpath "${1:-$ROOT/configs/quickstart.yaml}")"

# Load .env without overriding variables already set in the environment.
if [ -f "$ROOT/.env" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [ -z "${!key:-}" ] && [ -n "$value" ] && export "$key=$value"
  done < "$ROOT/.env"
fi

export PATH="$HOME/.local/bin:$PATH"
LAB_BACKEND="${LAB_BACKEND:-cursor}"
export LAB_BACKEND
if [ "$LAB_BACKEND" = "upstream" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "LAB_BACKEND=upstream needs OPENAI_API_KEY or DEEPSEEK_API_KEY in .env (or the environment)." >&2
  exit 1
fi

# Agent Laboratory wipes ./MATH_research_dir at the start of every run, so each run gets its own cwd.
RUN_DIR="${RUN_DIR:-$ROOT/runs/$(date +%Y%m%d-%H%M%S)-$(basename "$CONFIG" .yaml)}"
mkdir -p "$RUN_DIR"
cp "$CONFIG" "$RUN_DIR/config.yaml"
echo "Run directory: $RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

cd "$RUN_DIR"
PYTHONPATH="$LAB${PYTHONPATH:+:$PYTHONPATH}" exec "$VENV/bin/python" "$ROOT/lab/launch.py" --yaml-location "$RUN_DIR/config.yaml"
