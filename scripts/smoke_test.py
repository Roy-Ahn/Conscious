"""Checks that the Agent Laboratory environment is usable.

By default no LLM request is made. `--live` additionally sends one tiny request to the configured backend.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "third_party" / "AgentLaboratory"
sys.path[:0] = [str(ROOT / "lab"), str(LAB)]
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ["PATH"] = f"{Path.home() / '.local' / 'bin'}:{os.environ['PATH']}"

env_file = ROOT / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.strip().startswith("#") and value.strip():
            os.environ.setdefault(key.strip(), value.strip())

BACKEND = os.getenv("LAB_BACKEND", "cursor")
LIVE = "--live" in sys.argv
failures = []


def check(name, fn, required=True):
    try:
        detail = fn()
        print(f"[ok]   {name}" + (f": {detail}" if detail else ""))
    except Exception as e:  # noqa: BLE001
        tag = "FAIL" if required else "warn"
        print(f"[{tag}] {name}: {e}")
        if required:
            failures.append(name)


def import_lab():
    # Upstream writes scratch files to the cwd on import, so keep them out of the repo.
    cwd = os.getcwd()
    os.chdir(tempfile.mkdtemp(prefix="agentlab-smoke-"))
    try:
        import ai_lab_repo  # noqa: F401
        import torch
    finally:
        os.chdir(cwd)
    return f"torch {torch.__version__}"


def run_code():
    from tools import execute_code

    out = execute_code("import numpy as np\nprint(int(np.arange(5).sum()))", timeout=60)
    if out.strip() != "10":
        raise RuntimeError(f"unexpected output {out!r}")
    return "generated-code executor works"


def arxiv_search():
    from tools import ArxivSearch

    res = ArxivSearch().find_papers_by_str("LLM agents as research assistants", N=2)
    if not res:
        raise RuntimeError("no results (network or arXiv rate limit?)")
    return res.splitlines()[0]


def cursor_routing():
    import cursor_backend
    import inference
    from tools import execute_code

    cursor_backend.install()
    if inference.query_model is not cursor_backend.query_model:
        raise RuntimeError("inference.query_model was not replaced")
    out = execute_code("print(query_llm.__module__, query_gpt4omini.__module__)", timeout=60).strip()
    if out != "cursor_backend cursor_backend":
        raise RuntimeError(f"experiment code does not see the Cursor helpers: {out!r}")
    return "agent and experiment-code LLM calls go to the Cursor CLI"


def cursor_cli():
    import cursor_backend

    return cursor_backend.cli_path()


def cursor_auth():
    import cursor_backend

    return f"model: {cursor_backend.check()}"


def upstream_keys():
    keys = [k for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY") if os.getenv(k)]
    if not keys:
        raise RuntimeError("no OPENAI_API_KEY / DEEPSEEK_API_KEY set; research runs will refuse to start")
    return ", ".join(keys)


def live_request():
    if BACKEND == "cursor":
        import cursor_backend

        reply = cursor_backend.complete("Reply with exactly the word OK.", "You are a test endpoint.")
    else:
        import inference

        reply = inference.query_model("gpt-4o-mini", "Reply with exactly the word OK.", "You are a test endpoint.")
    if "OK" not in reply:
        raise RuntimeError(f"unexpected reply {reply[:200]!r}")
    return reply[:80]


def pdflatex():
    import re
    import subprocess

    if not shutil.which("pdflatex"):
        raise RuntimeError("not installed; keep `compile-latex: False` or run setup_lab.sh --with-latex")
    # The paper-writing agent injects these packages into every document (utils.compile_latex).
    packages = re.findall(r"usepackage(?:\[[^\]]*\])?\{([^}]+)\}", (LAB / "utils.py").read_text())
    missing = [p for p in packages if subprocess.run(["kpsewhich", f"{p}.sty"], capture_output=True).returncode]
    if missing:
        raise RuntimeError(f"missing LaTeX packages: {', '.join(missing)}")
    return f"{len(packages)} paper packages available"


check("Agent Laboratory imports", import_lab)
check("Experiment code execution", run_code)
check("arXiv literature search", arxiv_search, required=False)
if BACKEND == "cursor":
    check("Cursor CLI installed", cursor_cli)
    check("Cursor backend routing", cursor_routing)
    check("Cursor authentication", cursor_auth, required=LIVE)
else:
    check("Upstream API key", upstream_keys, required=LIVE)
if LIVE:
    check(f"Live {BACKEND} request", live_request)
check("pdflatex", pdflatex, required=False)

if failures:
    print(f"\nSmoke test failed: {', '.join(failures)}")
    sys.exit(1)
print("\nSmoke test passed.")
