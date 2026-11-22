"""Checks that the Agent Laboratory environment is usable without spending any LLM tokens."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "third_party" / "AgentLaboratory"
sys.path.insert(0, str(LAB))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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


def api_keys():
    keys = [k for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY") if os.getenv(k)]
    env_file = ROOT / ".env"
    if not keys and env_file.exists():
        for line in env_file.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY") and v.strip():
                keys.append(k.strip())
    if not keys:
        raise RuntimeError("no OPENAI_API_KEY / DEEPSEEK_API_KEY set; research runs will refuse to start")
    return ", ".join(keys)


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
check("LLM API key", api_keys, required=False)
check("pdflatex", pdflatex, required=False)

if failures:
    print(f"\nSmoke test failed: {', '.join(failures)}")
    sys.exit(1)
print("\nSmoke test passed.")
