#!/usr/bin/env bash
# Sets up the AI-assisted research laboratory (Agent Laboratory) environment.
#
# Usage: scripts/setup_lab.sh [--with-latex] [--gpu]
#   --with-latex  install pdflatex + the LaTeX packages the paper-writing agent uses (needs sudo/apt)
#   --gpu         install the default (CUDA) PyTorch wheel instead of the CPU-only build
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT/third_party/AgentLaboratory"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3.12}"

WITH_LATEX=0
GPU=0
for arg in "$@"; do
  case "$arg" in
    --with-latex) WITH_LATEX=1 ;;
    --gpu) GPU=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

cd "$ROOT"
git submodule update --init --recursive

if ! command -v "$PYTHON" >/dev/null; then
  echo "$PYTHON not found; Agent Laboratory recommends Python 3.12." >&2
  exit 1
fi

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV" || {
    echo "venv creation failed; on Debian/Ubuntu run: sudo apt-get install -y python3.12-venv" >&2
    exit 1
  }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

TORCH_VERSION="$(grep -E '^torch==' "$LAB/requirements.txt" | cut -d= -f3)"
if [ "$GPU" -eq 1 ]; then
  pip install "torch==$TORCH_VERSION"
else
  pip install "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
fi
pip install -r "$LAB/requirements.txt" -r "$ROOT/requirements-lab.txt"

if [ "$WITH_LATEX" -eq 1 ]; then
  SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra \
    texlive-fonts-recommended texlive-fonts-extra texlive-science texlive-plain-generic
fi

[ -f "$ROOT/.env" ] || cp "$ROOT/.env.example" "$ROOT/.env"

python "$ROOT/scripts/smoke_test.py"
echo
echo "Lab ready. Add an API key to .env, then run: scripts/run_lab.sh configs/quickstart.yaml"
