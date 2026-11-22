#!/usr/bin/env bash
# Copies a finished run's shareable outputs (config, experiment code, LaTeX/PDF report, figures,
# timings) from the git-ignored runs/ folder into results/<run-name>/ so they can be committed.
#
# Usage: scripts/export_run.sh [runs/<run-name>]   (default: the newest run)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${1:-$(ls -td "$ROOT"/runs/*/ 2>/dev/null | head -1)}"
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "No run directory found." >&2
  exit 1
fi
RUN_DIR="$(realpath "$RUN_DIR")"
DEST="$ROOT/results/$(basename "$RUN_DIR")"
mkdir -p "$DEST"

cp "$RUN_DIR/config.yaml" "$DEST/" 2>/dev/null || true
[ -d "$RUN_DIR/MATH_research_dir" ] && cp -r "$RUN_DIR/MATH_research_dir/." "$DEST/"
find "$RUN_DIR" -maxdepth 1 \( -name '*.png' -o -name '*.pdf' -o -name 'agent_times_*.txt' -o -name 'run.log' \) \
  -exec cp {} "$DEST/" \;
find "$DEST" \( -name '*.aux' -o -name '*.out' -o -name '__pycache__' \) -prune -exec rm -rf {} +

echo "Exported to ${DEST#"$ROOT"/}:"
find "$DEST" -type f | sed "s|^$ROOT/||" | sort
