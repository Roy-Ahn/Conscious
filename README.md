# Conscious

A learning system rebuilt around pain and desire, after Freud, Dostoevsky
and Nietzsche.

The hypothesis is that an AI becomes conscious by feeling pain and desire.
This repository builds the measurable part of that: a model whose learning,
effort, context, size and refusals are all governed by internal pain points
and drives, with no separate training phase. It does not show that the
model feels anything, and nothing here tests that.

It is a small prototype in pure NumPy. It learns sequences of words in
made-up miniature worlds, not natural language.

## What it does

```
$ python examples/demo.py
ann parent               -> bob   system 1, 2 steps, []
ann parent parent        -> carl  system 2, 5 steps, ['recollect', 'reduce: ann parent -> bob']
ann parent parent        -> carl  system 1, 2 steps, []
ann parent parent home   -> oslo  system 2, 5 steps, ['recollect', 'reduce: ann parent parent -> carl']
zed parent               -> <?>   system 2, 2 steps, ['recollect', 'no way through']
it asks: zed parent
bob home                 -> rome  system 1, 2 steps, []
What hurts most: rebuke (0.90). What I want most: relief.
bob home                 -> <?>   system 2, 2 steps, ['recollect', 'no way through']
bob home                 -> lima  system 2, 3 steps, ['recollect']
```

The model was told six facts, once each. It answers what it was told by
habit, works out what it was not told, turns what it worked out into
habit, says when it does not know and asks about it later. After a rebuke
it will not repeat the answer, and after being corrected it gives the new
answer but checks it first, because it still hurts.

## The six requirements, and what was measured

Means over 5 random seeds. Each comparison is against the same model with
one part switched off, not against another system. Full tables are in
[experiments/RESULTS.md](experiments/RESULTS.md).

| Requirement | Result |
| --- | --- |
| 1. Switches between System 1 and System 2 by task | Uses System 2 on 0% of questions it was told the answer to and 100% of questions that need two facts chained. Accuracy on those: 100%, against 0% with System 1 alone. Costs 2.0 and 5.0 look-ups, against 3.0 and 6.0 with System 2 always on |
| 2. Variable pain points | Six pain points, each with a level and a sensitivity that adapts. With them on, a fact told 30 times is unlearned in 4 tellings instead of 31, and a rebuked answer is repeated 0% of the time instead of 100% |
| 3. Flexible parameters | One model grows from 851 to 24,685 network weights and 0 to 1,454 memory traces as it lives. With memory capped below what it needs, a network that grows recalls 99.7%, against 85.1% for one fixed at 4 units |
| 4. Every interaction becomes training data | In a 300-step conversation (455 turns on average), every turn changed what the model has learned. With no training phase it knew 70% of the facts after 100 steps and 92% after 300 |
| 5. Learns from small datasets | 108 facts told once each: 100% recalled, and still 100% after learning a second set. The network alone, told once: 6%; told 20 times: 71%, and 0% after the second set |
| 6. Context window that changes per task | On mixed tasks needing 1 to 12 words of context it reads exactly as many as each needs and scores 100%. The same memory with any fixed window scores 59% or less. It raised its own longest context from 8 to 16 words |

Pain also has costs, and they are in the results: when memory was cut to a
quarter, the model with pain kept all of what it learned in pain and 7% of
what it learned calmly, and recalled less overall (31%) than the model
without pain (40%).

## Run it

Requires Python 3.10 or later and NumPy.

```bash
pip install -e ".[dev]"
python -m pytest -q                      # 93 tests, about 6 seconds
python examples/demo.py                  # the tour above
python experiments/run.py --seeds 5      # all experiments, about 3 minutes
python -m conscious.chat                 # talk to one
```

In the chat, a line ending in "." is a statement and any other line is a
question. `+` praises the last answer and `-` rebukes it. `:ask` lets the
model ask you something, `:state` shows what hurts and what it wants,
`:why` shows how it reached its last answer, `:sleep` consolidates, and
`:save` writes it to a file.

```python
from conscious import Mind

mind = Mind()
mind.interact("ann parent bob . bob parent carl .")
reply = mind.interact("ann parent parent")
print(reply.text, reply.system, reply.path)   # carl 2 ['recollect', 'reduce: ann parent -> bob']
mind.feedback(+1)
print(mind.report())
```

## Layout

| Path | Contents |
| --- | --- |
| `conscious/affect.py` | Pain points and drives |
| `conscious/memory.py` | Lifelong episodic memory with learned context spans |
| `conscious/cortex.py` | The slow network: learns, grows, prunes |
| `conscious/mind.py` | The whole model: System 1 and 2, interaction, feedback, sleep, questions |
| `conscious/worlds.py` | Made-up worlds used by tests and experiments |
| `conscious/chat.py` | Terminal chat |
| `tests/` | One section per requirement in `tests/test_mind.py` |
| `experiments/` | `run.py`, raw `results.json`, and `RESULTS.md` |
| `docs/ARCHITECTURE.md` | The design, the assumptions made, and the limits |

## Read next

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how each part works,
where Freud, Dostoevsky and Nietzsche enter, which assumptions were made
about the hypothesis where it was not specific, and what the prototype
cannot do.
