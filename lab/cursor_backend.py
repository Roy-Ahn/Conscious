"""Routes Agent Laboratory's LLM calls through a Cursor model via the Cursor CLI (`agent -p`).

Upstream funnels every agent call through `inference.query_model` and every call made by
generated experiment code through the `utils.query_*` helpers. `install()` replaces those
functions, so no upstream file is modified. Environment variables:

  CURSOR_API_KEY          Cursor API key (or log in once with `agent login`)
  CURSOR_MODEL            model slug from `agent --list-models`; empty = account default
  CURSOR_CLI              path to the CLI (default: `agent` / `cursor-agent` on PATH or ~/.local/bin)
  CURSOR_MAX_CONCURRENCY  max simultaneous CLI calls per process (default 4)
  CURSOR_TIMEOUT          seconds per call (default 600)
"""
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

# Linux caps a single argv string at 128 KiB (MAX_ARG_STRLEN); larger prompts go through a file.
ARG_LIMIT_BYTES = 100_000

PREAMBLE = (
    "You are acting as a raw language-model completion endpoint for an automated research "
    "pipeline (Agent Laboratory). Do not use tools, do not explore files, and do not mention "
    "being a coding assistant. Follow the SYSTEM instructions and answer the USER message exactly "
    "as they require, including any required command or code-block formats. Your reply is passed "
    "verbatim to the pipeline.\n\n"
)
FILE_PREAMBLE = (
    "You are acting as a raw language-model completion endpoint for an automated research "
    "pipeline (Agent Laboratory). The request is too large to send inline: read the file "
    "`prompt.md` in the current workspace (the only tool call you may make), then answer it. "
    "Follow its SYSTEM instructions and answer its USER message exactly as they require, including "
    "any required command or code-block formats. Your final message is passed verbatim to the "
    "pipeline, so it must contain only the answer."
)

_semaphore = threading.BoundedSemaphore(int(os.getenv("CURSOR_MAX_CONCURRENCY", "4")))


class CursorBackendError(RuntimeError):
    pass


def cli_path():
    explicit = os.getenv("CURSOR_CLI")
    if explicit:
        return explicit
    for name in ("agent", "cursor-agent"):
        found = shutil.which(name) or shutil.which(str(Path.home() / ".local" / "bin" / name))
        if found:
            return found
    raise CursorBackendError(
        "Cursor CLI not found. Install it with `curl https://cursor.com/install -fsS | bash` "
        "or set CURSOR_CLI."
    )


def _format(system, prompt):
    parts = []
    if system:
        parts.append(f"<SYSTEM>\n{system}\n</SYSTEM>")
    parts.append(f"<USER>\n{prompt}\n</USER>")
    return "\n\n".join(parts)


def _final_text(stdout):
    """Returns the last assistant message (the answer after any tool call) from stream-json output."""
    last_message, result = None, None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            text = "".join(c.get("text", "") for c in event.get("message", {}).get("content", []))
            if text:
                last_message = text
        elif event.get("type") == "result" and not event.get("is_error"):
            result = event.get("result")
    return last_message if last_message is not None else result


def complete(prompt, system=None, model=None, tries=3, timeout=None):
    """Sends one system+user request to the Cursor model and returns its text reply."""
    model = model or os.getenv("CURSOR_MODEL") or None
    timeout = timeout or float(os.getenv("CURSOR_TIMEOUT", "600"))
    body = _format(system, prompt)
    last_error = "no output"
    with _semaphore, tempfile.TemporaryDirectory(prefix="cursor-llm-") as workspace:
        if len(body.encode()) <= ARG_LIMIT_BYTES:
            message = PREAMBLE + body
        else:
            Path(workspace, "prompt.md").write_text(body)
            message = FILE_PREAMBLE
        cmd = [cli_path(), "-p", "--output-format", "stream-json", "--mode", "ask", "--trust",
               "--workspace", workspace]
        if model:
            cmd += ["--model", model]
        cmd.append(message)
        for attempt in range(tries):
            try:
                proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True, timeout=timeout)
                text = _final_text(proc.stdout) if proc.returncode == 0 else None
                if text:
                    return text.strip()
                last_error = (proc.stderr or proc.stdout).strip()[-2000:] or f"exit code {proc.returncode}"
            except subprocess.TimeoutExpired:
                last_error = f"timed out after {timeout:.0f}s"
            if attempt + 1 < tries:
                time.sleep(2 ** attempt * 2)
    raise CursorBackendError(f"Cursor model call failed: {last_error}")


def query_model(model_str, prompt, system_prompt, openai_api_key=None, gemini_api_key=None,
                anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    """Drop-in replacement for `inference.query_model`.

    `llm-backend: "cursor:<slug>"` in a config selects that model; any other value uses CURSOR_MODEL.
    Temperature is not supported by the CLI and is ignored.
    """
    model = model_str.split(":", 1)[1] if model_str.startswith("cursor:") else None
    return complete(prompt, system_prompt, model=model, tries=min(tries, 3))


def query_llm(prompt, system=None, api_key=None, attempt=0, temperature=0.0):
    """Model helper for generated experiment code; signature-compatible with `utils.query_*`."""
    return complete(prompt, system)


def install():
    import inference
    import utils

    inference.query_model = query_model
    for name in ("query_gpt4omini", "query_gpt4o", "query_deepseekv3", "query_qwen",
                 "query_gemini", "query_gemini2p0"):
        setattr(utils, name, query_llm)
    utils.query_llm = query_llm


def check(model=None):
    """Fails fast if the CLI is missing, not authenticated, or the model slug is unknown."""
    model = model or os.getenv("CURSOR_MODEL") or None
    proc = subprocess.run([cli_path(), "--list-models"], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise CursorBackendError(
            "Cursor CLI is not authenticated. Set CURSOR_API_KEY in .env or run `agent login`.\n"
            + (proc.stderr or proc.stdout).strip()
        )
    if model and model.split("[", 1)[0] not in proc.stdout:
        raise CursorBackendError(f"CURSOR_MODEL={model!r} is not available. Available models:\n{proc.stdout}")
    return model or "account default"
