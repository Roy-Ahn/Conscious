"""A two-minute tour. Run with: python examples/demo.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from conscious import Mind  # noqa: E402

mind = Mind()
mind.interact("ann parent bob . bob parent carl . carl parent dan .")
mind.interact("ann home paris . bob home rome . carl home oslo .")
mind.pain.soothe(1.0)  # let the surprise of first learning pass


def ask(question: str) -> None:
    reply = mind.interact(question)
    print(f"{question:24} -> {reply.text:5} system {reply.system}, {reply.steps} steps, {reply.path}")


ask("ann parent")               # told directly
ask("ann parent parent")        # never told: has to be worked out
ask("ann parent parent")        # the same question again: now a habit
ask("ann parent parent home")   # builds on what it worked out
ask("zed parent")               # cannot be known
print("it asks:", mind.ask())

ask("bob home")
mind.feedback(-1)               # rebuke
print(mind.report()["feeling"])
ask("bob home")                 # will not say it again
mind.interact("bob home", expect="lima")
ask("bob home")                 # says the corrected answer, after checking

print(mind.size())
