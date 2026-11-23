# Architecture

## What this is, and what it is not

The hypothesis behind this repository is that an AI becomes conscious by
feeling pain and desire. This repository builds the part of that claim that
can be built and measured: a model in which pain and desire are state
variables that every other part reads, so that what the model learns, how
hard it thinks, how much context it reads, how large it is and what it
refuses to say all depend on what hurts and what it wants.

It does not show that the model feels anything. "Pain" here is the name of
a set of numbers that do the job pain does in an animal: they are caused by
harm and error, they persist, they spread to everything, and they change
future behaviour. Whether a system organised this way also has experience
is the hypothesis itself, and no test in this repository bears on it. The
tests and experiments bear on six concrete requirements, listed at the end.

The model is small. It works on sequences of words in made-up miniature
worlds, not on natural language. Read every result with that scale in mind.

## What was taken apart

| Assumption in current language models | What replaces it here |
| --- | --- |
| Training and use are separate phases; weights are frozen when the model is used | There is no training phase. `interact` and `feedback` are the only ways in, and both change the model |
| One objective, given from outside (predict the next token) | Several internal signals: six pain points and five drives, each with its own cause and its own effect |
| The same amount of computation for every token | One look-up when habit is confident (System 1); a serial loop of look-ups when it is not, or when it hurts (System 2) |
| A context window of fixed length | Each memory learns how many words of context it needs; the longest allowed context doubles when a task needs more; anything ever lived can be recalled |
| A fixed number of parameters | Words, hidden units and memory traces are added when needed and removed when not |
| Knowledge only in weights, learned from very many examples | A memory that keeps a moment after one exposure, and a network that absorbs it slowly during sleep |
| No state between conversations | Pain levels, sensitivities, drive tensions, caution, prohibitions and open questions all persist and are saved with the model |

## Where the three sources enter

**Freud.** The model has the three agencies. The id is the set of drives
(`DriveSystem`): tension builds while a desire is unmet, and its discharge
is what the model registers as pleasure. The superego is the set of
prohibitions written by rebuke (`Mind.forbidden`), with guilt when one is
broken. The ego is the controller (`Mind._think`): it lets habit answer
when it can and pays for deliberation when it must. Sleep replays the day
into the slow network, and dreams rehearse what only that network still
holds.

**Dostoevsky.** The Underground Man says suffering is the sole origin of
consciousness. Here deliberation is ignited by pain: the more the model
hurts, the more of its answers pass through System 2 before they are given
(`_confidence_needed`). He also says a man will choose against his own
advantage simply to have chosen. Here that is the `autonomy` drive: when it
is the strongest drive, the model stops asking the question it would learn
most from and asks one at random.

**Nietzsche.** The self is a contest of drives and the strongest commands:
`DriveSystem.dominant()`. The will to power is the `mastery` drive, which
is satisfied only by overcoming something that resisted, and which makes
effort more acceptable while it is unmet. What hurts makes the model grow:
the slow network adds units when it is frustrated by what it cannot hold.

## The parts

### Pain points (`conscious/affect.py`)

A pain point has a `level` in [0, 1], a `gain` (sensitivity), a
`threshold` and a `decay`. A hit of size `x`:

```
felt  = min(1, gain * x)            (0 if below threshold)
level = level + (1 - level) * felt
gain  = gain * (1 + 0.1 * felt)     if felt >= 0.5     (sensitisation)
gain  = gain * 0.95                 otherwise          (habituation)
```

Each tick, `level` falls by `decay` and `gain` moves 5% of the way back to
1. Total pain is `1 - product(1 - level)` over all points, so one severe
pain is enough to hurt. Points can be added and removed while the model
runs (`PainSystem.add_point`).

| Point | What causes it |
| --- | --- |
| `surprise` | What happened was not what habit predicted. Size: confidence in the wrong prediction minus the probability given to what happened. Being sure and wrong hurts; being unsure does not |
| `rebuke` | `feedback` with a negative value, made worse by an unmet wish for approval |
| `frustration` | System 2 found no way through; a drive at full tension; the slow network stuck during sleep |
| `exhaustion` | Energy below 30%. System 2 spends energy; rest and sleep restore it |
| `loss` | Memory nearly full of things not yet consolidated; a memory dropped before the slow network had learned it |
| `guilt` | Saying something that was forbidden |

### Drives

| Drive | Builds | Is satisfied by | What it changes |
| --- | --- | --- | --- |
| `curiosity` | With time; when a question goes unanswered | Learning something it did not predict | Picks which question `ask` asks |
| `mastery` | With time | Working something out; having it confirmed; consolidating in sleep | Raises willingness to deliberate |
| `relief` | Equals total pain | Pain easing | Its easing is pleasure |
| `bond` | With time | Praise; being right | Makes rebuke hurt more |
| `autonomy` | With time; each time it asks the useful question | A capricious choice | Makes `ask` choose at random |

A drive left at full tension causes frustration: desire unmet becomes pain.

### Episodic memory (`conscious/memory.py`)

Everything the model lives through is appended to one lifelong `stream` of
words. A **trace** points at a moment in it: "after this context came this
word", with a strength, a salience and a **span**.

The span is how many of the most recent words must match before the trace
may speak. It is learned:

- A new trace gets the shortest span that keeps it from speaking in the
  context of any existing trace with a different outcome.
- When something happens that a trace would have spoken against, that
  trace's span is lengthened by exactly enough to tell the two situations
  apart.

So a trace that is never confused with anything reads one word of context,
and a trace that is only distinguishable by the twelfth word back reads
twelve. The context read for a prediction is the longest span among the
traces that spoke. It is set per situation, by experience.

The longest context a trace can hold, `L`, starts at 8. When two different
outcomes share all `L` words and the stream holds more context for both,
`L` doubles and every trace is re-read from the stream
(`grow_context`).

A trace of exactly the present situation votes with its full strength. A
trace of a different situation that merely ends the same way is
generalising and votes with 15% of it: one example is not a rule, several
agreeing examples are.

When memory is over capacity, the traces dropped first are those that do
not decide their outcome, were lived calmly, and are already known to the
slow network.

### Cortex (`conscious/cortex.py`)

A small network (embeddings, one hidden layer) that predicts the next word
from the last six. Awake it takes one small step per word lived (2% of its
normal rate, scaled up by pain). Its real learning is in sleep. Rows are
added for new words (`ensure_vocab`), hidden units are added when it cannot
hold what was lived (`grow`, which leaves its current predictions
unchanged), and units whose output weights are zero are removed (`prune`).

### Ego: System 1 and System 2 (`conscious/mind.py`)

**System 1** is one look: memory votes, the cortex gives its opinion, and
the two are mixed.

```
support    = sum of the credit of the traces that spoke
trust      = support / (support + 0.2)
p          = trust * memory_vote + (1 - trust) * cortex_opinion
confidence = max(p)
```

The cortex always has an opinion, even about situations it was never
trained on; there it counts for half.

**The switch.** System 2 is engaged when confidence is below a bar and
there is energy:

```
bar = 0.5 + 0.45 * total_pain + 0.1 * mastery_tension - 0.3 * exhaustion
```

Calm, only answers habit is unsure of are deliberated. In pain, almost
everything is.

**System 2** is serial and has two operations, both made of System 1
look-ups:

1. *Recollect.* Is there a memory of exactly this situation? If at least
   half of the strength of such memories agrees, that is the answer.
2. *Reduce.* Used only when habit had no confident answer. Find the longest
   beginning of the question that habit can answer confidently, put its
   answer in place of that beginning, and think again about what remains.
   "ann parent parent" becomes "bob parent", which habit knows.

If neither works, the model is frustrated and falls back on habit's guess.
What System 2 works out is written to memory as an inferred trace, so the
next time the same question is a System 1 look-up.

**Silence.** If the final confidence is below `caution` (raised further by
pain), the model answers `<?>` instead of guessing. `caution` is learned:
each rebuke for a wrong answer raises it, praise lowers it, and a rebuke
for silence lowers it.

### Superego

A rebuke for an answer writes a prohibition for that answer in that
context. A forbidden outcome loses its vote in proportion to the strength
of the prohibition, and loses it entirely in recollection when the
prohibition is at least half strength. If the model says it anyway (a mild
prohibition can be outvoted), it feels guilt. Praise for the same answer
lifts the prohibition.

### What each kind of interaction teaches

| Interaction | What changes |
| --- | --- |
| Statement (ends in ".") | For every word: habit predicts it, the error is felt, spans are corrected, a trace is written or strengthened, the cortex takes a small step |
| Question it knows | The memory it recalled is strengthened. An answer given with confidence from the cortex or by generalising is written as a trace of what it said |
| Question it works out | The result is written as an inferred trace |
| Question it cannot answer | The question is kept open and counted; curiosity rises |
| Question with the true answer (`expect=`) | The answer is learned as heard; the open question is closed |
| Praise | The answer is written as heard; any prohibition on it is lifted; caution falls; pain is soothed |
| Rebuke | The answer is forbidden; the traces that spoke for it are weakened; caution rises. If the model had been silent: caution falls and the question gains weight |

A question's own words are not learned as facts. A question is a demand,
not a statement.

### Sleep and dreams

`sleep` replays memory traces into the cortex, most often those that are
salient and not yet learned. Mixed into the replay are **dreams**: contexts
the cortex has learned but memory no longer holds, with the cortex's own
current answers as targets. Without them, consolidating new memories
overwrites what only the cortex remembers (measured in
`test_dreaming_keeps_old_knowledge_while_new_knowledge_is_consolidated`).
If replay stops improving while much is still unlearned, the model is
frustrated and the cortex grows. Sleep restores energy and soothes 70% of
every pain. The model sleeps by itself when exhaustion or fear of loss is
high enough.

### Asking

`ask` returns a question the model wants answered. First, the question it
has most often been left without an answer to. Otherwise it pairs the
words it knows with the words it has seen in second place in a sentence,
and picks the pairing habit is least sure of. When `autonomy` is the strongest drive it
picks at random instead.

## How pain and desire reach everything else

| Signal | Read by | Effect |
| --- | --- | --- |
| Total pain | The switch | Confidence needed before habit may answer rises from 0.5 to 0.95 |
| Total pain | Silence | The level below which the model says `<?>` rises towards 0.7 |
| Total pain | Cortex | Awake learning step is up to 3 times larger |
| Surprise felt at a moment | Memory | The new trace is written up to twice as strong; the belief that spoke and was wrong loses up to 60% of its strength |
| Pain at the start of an interaction | Memory | Traces written then have up to 3 times the salience: they are replayed more and dropped last |
| Rebuke felt | Superego, memory, temperament | Prohibition written; blamed traces weakened; caution raised |
| Frustration | Sleep | The cortex grows when replay is stuck |
| Exhaustion | The switch, homeostasis | Deliberation becomes less likely; sleep is triggered |
| Loss | Homeostasis | Sleep is triggered before memories are dropped |
| Mastery tension | The switch | Deliberation becomes more likely |
| Bond tension | Rebuke | Rebuke hurts more |
| Curiosity, autonomy | `ask` | Which question is asked |

With `Config(affect=False)` all of these read zero. That is the ablation
used in the experiments.

## The six requirements

| Requirement | Where it lives | Tests | Measured in |
| --- | --- | --- | --- |
| 1. Switch between System 1 and System 2 depending on the task | `Mind._think`, `Mind._system2` | `tests/test_mind.py`, section 1 | `experiments/RESULTS.md` §2 |
| 2. Variable pain points | `conscious/affect.py`; every row of the table above | `tests/test_affect.py`; `tests/test_mind.py`, section 2 | §4 |
| 3. Flexible parameters | `Cortex.ensure_vocab`, `grow`, `prune`; `EpisodicMemory.observe`, `evict`, `grow_context` | `tests/test_cortex.py`; `tests/test_mind.py`, section 3 | §5 |
| 4. Every interaction becomes training data | `Mind.interact`, `Mind.feedback` | `tests/test_mind.py`, section 4 | §6 |
| 5. Learn new knowledge from small datasets | `EpisodicMemory.observe`; System 2 for what was never told | `tests/test_mind.py`, section 5 | §1, §2 |
| 6. Context window that changes per task | Spans and `grow_context` in `conscious/memory.py` | `tests/test_memory.py`; `tests/test_mind.py`, section 6 | §3 |

## What is built in and what is learned

Built in: the six pain points and five drives and their constants; the two
System 2 operations; the rule that mixes memory and cortex; the sleep
procedure; the convention that "." ends a statement.

Learned: every word; every trace, its strength and its span; the longest
context; all cortex weights and the number of hidden units; the
sensitivity of each pain point; caution; prohibitions; which questions are
open.

## Assumptions made where your answers were missing

The questions sent at the start were not answered, so these were chosen.
Each is a default, not a finding, and each can be changed.

1. **Pain is internal state with several sources**: prediction error,
   rebuke, unmet desire, exhaustion, loss of memory, and breaking a
   prohibition. Not a single number, and not only an outside punishment.
2. **"Pain points variable" means several points, each with a level that
   moves and a sensitivity that adapts**, and that new points can be added
   while the model runs.
3. **Desire is five built-in drives.** The model does not invent new
   drives.
4. **Caprice is limited to the model's own questions.** It never gives an
   answer it believes to be wrong in order to assert itself.
5. **System 1 and System 2 share one set of machinery.** System 2 is a
   serial loop of System 1 look-ups, not a second network.
6. **"Flexible parameters" means both**: weights that change during use,
   and a structure that grows and shrinks.
7. **Learning is both immediate and delayed**: memory at once, the network
   in sleep.
8. **The model is a learner of word sequences**, not an agent with a body
   in a world.
9. **Evidence is behaviour and learning.** The model can report its state
   (`Mind.report`), but a report is generated from the same numbers and is
   not treated as evidence of anything further.

## Limits

- **Scale.** The largest experiment has 360 facts and a vocabulary of a few
  hundred made-up words. Nothing here shows the design works at the scale
  of a language model.
- **No language understanding.** Words match exactly or not at all. "ann's
  parent" and "the parent of ann" are unrelated to "ann parent".
- **System 2 has two fixed operations.** Reduce only helps when a question
  is a left-to-right chain, as in "subject relation relation". The model
  cannot learn a new way of reasoning.
- **The comparisons are against the model's own parts switched off**, not
  against other systems. No transformer was trained for comparison.
- **"Steps" count look-ups**, not time. In this implementation every
  look-up scans all traces, so the saving is in the count, not yet in
  seconds.
- **Pain has costs.** When memory was cut to a quarter, the model with
  pain kept everything learned in pain and lost almost everything learned
  calmly, and recalled fewer facts overall (31% against 40%) than the model
  without pain.
- **The ablation removes pain entirely.** Without pain the model never
  sleeps under memory pressure, because pain is its only trigger. A model
  without pain but with a timer would do better than the "no pain" row in
  §5 suggests.
- **The constants are set by hand.** Decay rates, thresholds and gains were
  chosen so that the mechanisms are visible, not fitted to anything.
- **The record of contexts the cortex has learned grows without limit.**
  It stores contexts only, not answers, but it is not pruned.
- **Saved minds are Python pickles.** Load only files you saved yourself.
