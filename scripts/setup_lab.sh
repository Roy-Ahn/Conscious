#!/usr/bin/env bash
# Sets up the AI-assisted research laboratory (Agent Laboratory) environment, including the
# Cursor CLI that the default LLM backend uses. Idempotent and non-interactive, so it doubles as
# the Cursor Cloud Agent environment's install step.
#
# Usage: scripts/setup_lab.sh [--with-latex] [--gpu]
#   --with-latex  install pdflatex + the LaTeX packages the paper-writing agent uses (needs sudo/apt)
#   --gpu         install the default (CUDA) PyTorch wheel instead of the CPU-only build
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAB="$ROOT/third_party/AgentLaboratory"
VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3.12}"
LATEX_PACKAGES=(texlive-latex-base texlive-latex-recommended texlive-latex-extra
  texlive-fonts-recommended texlive-fonts-extra texlive-science texlive-plain-generic)

WITH_LATEX=0
GPU=0
for arg in "$@"; do
  case "$arg" in
    --with-latex) WITH_LATEX=1 ;;
    --gpu) GPU=1 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if sudo -n true 2>/dev/null; then SUDO="sudo"; else SUDO="no-sudo"; fi
fi
apt_install() {
  if [ "$SUDO" = "no-sudo" ]; then
    echo "Need root to install: $*" >&2
    return 1
  fi
  $SUDO apt-get update -qq
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends "$@"
}

cd "$ROOT"
git submodule update --init --recursive

if ! command -v "$PYTHON" >/dev/null; then
  echo "$PYTHON not found; Agent Laboratory recommends Python 3.12." >&2
  exit 1
fi
if [ ! -x "$VENV/bin/pip" ]; then
  rm -rf "$VENV"
  if ! "$PYTHON" -m venv "$VENV" 2>/dev/null; then
    rm -rf "$VENV"
    apt_install "${PYTHON}-venv"
    "$PYTHON" -m venv "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install -q --upgrade pip wheel

TORCH_VERSION="$(grep -E '^torch==' "$LAB/requirements.txt" | cut -d= -f3)"
if [ "$GPU" -eq 1 ]; then
  pip install -q "torch==$TORCH_VERSION"
else
  pip install -q "torch==$TORCH_VERSION" --index-url https://download.pytorch.org/whl/cpu
fi
pip install -q -r "$LAB/requirements.txt" -r "$ROOT/requirements-lab.txt"

if [ "$WITH_LATEX" -eq 1 ]; then
  if command -v pdflatex >/dev/null && kpsewhich soul.sty fontawesome5.sty sidecap.sty >/dev/null 2>&1; then
    echo "LaTeX packages already installed."
  else
    apt_install "${LATEX_PACKAGES[@]}"
  fi
fi

export PATH="$HOME/.local/bin:$PATH"
if ! command -v agent >/dev/null && ! command -v cursor-agent >/dev/null; then
  curl -fsSL https://cursor.com/install | bash
fi
# Non-interactive login shells (Cloud Agent install/start, cron) do not read ~/.bashrc PATH edits.
if [ "$SUDO" != "no-sudo" ] && [ -x "$HOME/.local/bin/agent" ]; then
  $SUDO ln -sf "$HOME/.local/bin/agent" /usr/local/bin/agent
  $SUDO ln -sf "$HOME/.local/bin/cursor-agent" /usr/local/bin/cursor-agent
fi

[ -f "$ROOT/.env" ] || cp "$ROOT/.env.example" "$ROOT/.env"

python "$ROOT/scripts/smoke_test.py"
echo
echo "Lab ready. Add CURSOR_API_KEY to .env or as a Cursor secret (or run \`agent login\`), then run: scripts/run_lab.sh"
