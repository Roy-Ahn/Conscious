"""Tests lab/cursor_backend.py against a fake Cursor CLI (no network or Cursor account needed).

Run with: .venv/bin/python -m unittest discover -s tests
"""
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "lab"), str(ROOT / "third_party" / "AgentLaboratory")]

import cursor_backend  # noqa: E402

FAKE_CLI = textwrap.dedent(
    """\
    #!/usr/bin/env python3
    import json, os, sys
    from pathlib import Path

    args = sys.argv[1:]
    log = Path(os.environ["FAKE_CLI_LOG"])
    calls = json.loads(log.read_text()) if log.exists() else []
    calls.append({"args": args, "cwd": os.getcwd(),
                  "prompt_file": Path("prompt.md").read_text() if Path("prompt.md").exists() else None})
    log.write_text(json.dumps(calls))

    if "--list-models" in args:
        print("Available models\\nclaude-opus-5-5 - Claude Opus 5.5\\ngpt-5.6 - GPT-5.6")
        sys.exit(0)
    fail_times = int(os.environ.get("FAKE_CLI_FAIL_TIMES", "0"))
    if len(calls) <= fail_times:
        print("Error: rate limited", file=sys.stderr)
        sys.exit(1)

    def emit(event):
        print(json.dumps(event))

    emit({"type": "system", "subtype": "init", "model": "fake"})
    if Path("prompt.md").exists():
        emit({"type": "assistant", "message": {"content": [{"type": "text", "text": "I'll read prompt.md"}]}})
        emit({"type": "tool_call", "subtype": "completed"})
        answer = "FILE:" + str(len(Path("prompt.md").read_text()))
    else:
        answer = "ECHO:" + args[-1][-40:]
    emit({"type": "assistant", "message": {"content": [{"type": "text", "text": answer}]}})
    emit({"type": "result", "subtype": "success", "is_error": False, "result": answer})
    """
)


class CursorBackendTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cli = Path(self.tmp.name, "agent")
        cli.write_text(FAKE_CLI)
        cli.chmod(cli.stat().st_mode | stat.S_IEXEC)
        self.log = Path(self.tmp.name, "calls.json")
        self.env = {"CURSOR_CLI": str(cli), "FAKE_CLI_LOG": str(self.log), "CURSOR_MODEL": "claude-opus-5-5"}
        self.saved = {k: os.environ.get(k) for k in [*self.env, "FAKE_CLI_FAIL_TIMES"]}
        os.environ.update(self.env)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def calls(self):
        return json.loads(self.log.read_text())

    def test_inline_prompt_uses_read_only_print_mode(self):
        reply = cursor_backend.complete("What is 2+2?", "Be terse.")
        self.assertEqual(reply, "ECHO:" + "Be terse.\n</SYSTEM>\n\n<USER>\nWhat is 2+2?\n</USER>"[-40:])
        (call,) = self.calls()
        args = call["args"]
        for flag in ("-p", "--trust"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--mode") + 1], "ask")
        self.assertEqual(args[args.index("--output-format") + 1], "stream-json")
        self.assertEqual(args[args.index("--model") + 1], "claude-opus-5-5")
        self.assertEqual(Path(args[args.index("--workspace") + 1]), Path(call["cwd"]))
        self.assertIn("<SYSTEM>\nBe terse.\n</SYSTEM>", args[-1])
        self.assertFalse(Path(call["cwd"]).exists(), "temporary workspace should be removed")

    def test_large_prompt_goes_through_file_and_returns_final_message(self):
        big = "x" * (cursor_backend.ARG_LIMIT_BYTES + 10)
        reply = cursor_backend.complete(big, "sys")
        (call,) = self.calls()
        self.assertEqual(call["args"][-1], cursor_backend.FILE_PREAMBLE)
        self.assertIn(big, call["prompt_file"])
        self.assertEqual(reply, f"FILE:{len(call['prompt_file'])}")

    @mock.patch("cursor_backend.time.sleep")
    def test_retries_then_raises(self, _sleep):
        os.environ["FAKE_CLI_FAIL_TIMES"] = "1"
        self.assertTrue(cursor_backend.complete("hi").startswith("ECHO:"))
        self.assertEqual(len(self.calls()), 2)

        self.log.unlink()
        os.environ["FAKE_CLI_FAIL_TIMES"] = "99"
        with self.assertRaisesRegex(cursor_backend.CursorBackendError, "rate limited"):
            cursor_backend.complete("hi", tries=2)

    def test_query_model_model_selection(self):
        cursor_backend.query_model("cursor:gpt-5.6", "p", "s")
        cursor_backend.query_model("gpt-4o-mini", "p", "s", temp=0.7)
        first, second = (c["args"] for c in self.calls())
        self.assertEqual(first[first.index("--model") + 1], "gpt-5.6")
        self.assertEqual(second[second.index("--model") + 1], "claude-opus-5-5")

    def test_check_validates_model(self):
        self.assertEqual(cursor_backend.check(), "claude-opus-5-5")
        with self.assertRaisesRegex(cursor_backend.CursorBackendError, "not available"):
            cursor_backend.check("no-such-model")

    def test_install_replaces_upstream_calls(self):
        import inference
        import utils

        cursor_backend.install()
        self.assertIs(inference.query_model, cursor_backend.query_model)
        self.assertIs(utils.query_gpt4omini, cursor_backend.query_llm)
        self.assertTrue(utils.query_llm("q", "s").startswith("ECHO:"))


if __name__ == "__main__":
    unittest.main()
