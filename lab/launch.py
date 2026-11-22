"""Runs Agent Laboratory's ai_lab_repo.py with the configured LLM backend.

LAB_BACKEND=cursor (default) sends every LLM call to a Cursor model (see cursor_backend.py).
LAB_BACKEND=upstream keeps upstream's OpenAI / DeepSeek API calls.
Either way, generated experiment code can call `query_llm(prompt, system)`.
"""
import os
import runpy
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent.parent / "third_party" / "AgentLaboratory"
sys.path[:0] = [str(Path(__file__).resolve().parent), str(LAB)]


def main():
    backend = os.getenv("LAB_BACKEND", "cursor")
    if backend == "cursor":
        import cursor_backend

        try:
            print(f"LLM backend: Cursor ({cursor_backend.check()})")
        except cursor_backend.CursorBackendError as e:
            sys.exit(str(e))
        cursor_backend.install()
        # ai_lab_repo.py refuses to start without one of these; nothing reads it once the backend is installed.
        if not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = "unused-cursor-backend"
    elif backend == "upstream":
        import utils

        def query_llm(prompt, system=None, api_key=None, attempt=0, temperature=0.0):
            return utils.query_gpt4omini(prompt, system, api_key or os.environ["OPENAI_API_KEY"],
                                         temperature=temperature)

        utils.query_llm = query_llm
    else:
        sys.exit(f"Unknown LAB_BACKEND={backend!r}; use 'cursor' or 'upstream'.")

    script = LAB / "ai_lab_repo.py"
    sys.argv = [str(script), *sys.argv[1:]]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
