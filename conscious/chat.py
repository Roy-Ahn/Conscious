"""Talk to a mind from the terminal.

    python -m conscious.chat [file.mind]

End a line with "." to tell it something. Leave the "." off to ask it to
finish the sentence. After an answer, "+" praises it and "-" rebukes it.
"""

from __future__ import annotations

import json
import os
import sys

from .mind import Mind

HELP = """\
  ann parent bob .     tell it a fact
  ann parent           ask it to complete
  +  /  -              praise / rebuke the last answer
  :ask                 let it ask you something
  :state               what hurts, what it wants, how big it is
  :why                 how it reached its last deliberate answer
  :sleep               let it consolidate
  :save                save to the file given at start-up
  :quit"""


def describe(reply) -> str:
    if reply.text == "":
        return "(heard)"
    how = "by habit" if reply.system == 1 else "after thinking"
    note = f"{how}, {reply.steps} steps, read {reply.window} words back"
    if reply.slept:
        note += ", then slept"
    return f"{reply.text}    [{note}; pain {reply.pain:.2f}]"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if path and os.path.exists(path):
        mind = Mind.load(path)
        print(f"Woke {path}: {mind.size()['traces']} memories.")
    else:
        mind = Mind()
        print("A new mind. It knows nothing yet.")
    print(HELP)
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line == ":quit":
            break
        if line in ("+", "-"):
            if mind.last is None:
                print("(nothing to judge yet)")
            else:
                mind.feedback(+1 if line == "+" else -1)
                print(f"({'praised' if line == '+' else 'rebuked'}; {mind.report()['feeling']})")
        elif line == ":ask":
            print(mind.ask() or "(nothing to ask)")
        elif line == ":state":
            print(json.dumps(mind.report(), indent=2))
        elif line == ":why":
            print(json.dumps(mind.moments[-1], indent=2) if mind.moments else "(no deliberation yet)")
        elif line == ":sleep":
            print(json.dumps(mind.sleep(), indent=2))
        elif line == ":save":
            if path:
                mind.save(path)
                print(f"Saved to {path}.")
            else:
                print("No file was given at start-up.")
        elif line.startswith(":"):
            print(HELP)
        else:
            print(describe(mind.interact(line)))
    if path:
        mind.save(path)
        print(f"Saved to {path}.")


if __name__ == "__main__":
    main()
