"""Mind: the whole model.

Three agencies, after Freud:

  id        the drives (`DriveSystem`): what the model wants
  superego  prohibitions written by rebuke: what it must not say again
  ego       the controller in this file: answers by habit when it can,
            deliberates when habit fails or when it hurts

Two ways of answering:

  System 1  one look at memory and cortex
  System 2  a serial loop that checks an answer against exact recollection
            and composes answers out of several System 1 look-ups

There is no training phase. `interact` and `feedback` are the only ways in,
and both always change the model.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field

import numpy as np

from .affect import DriveSystem, PainSystem
from .cortex import Cortex
from .memory import HEARD, INFERRED, SAID, EpisodicMemory
from .vocab import Vocab


@dataclass
class Config:
    seed: int = 0

    # which parts are switched on (the ablations used in the experiments)
    affect: bool = True
    use_memory: bool = True
    use_cortex: bool = True
    allow_system2: bool = True
    force_system2: bool = False
    fixed_window: int | None = None

    # memory
    max_context: int = 8
    context_limit: int = 256
    capacity: int | None = None

    # cortex
    dim: int = 16
    hidden: int = 24
    cortex_window: int = 6
    lr: float = 0.01
    max_hidden: int = 512
    grow_cortex: bool = True

    # ego: when to deliberate, when to stay silent
    confidence_calm: float = 0.5
    confidence_hurt: float = 0.9
    abstain_calm: float = 0.3
    abstain_hurt: float = 0.6
    prior_weight: float = 0.2
    novel_trust: float = 0.5
    reduce_confidence: float = 0.6
    max_depth: int = 4
    max_new_tokens: int = 6

    # energy
    energy_system1: float = 0.001
    energy_system2: float = 0.01
    energy_recover: float = 0.02
    exhaustion_at: float = 0.3

    # how pain changes learning
    revise_rate: float = 0.8
    pain_salience: float = 2.0
    pain_plasticity: float = 2.0
    said_strength: float = 0.25
    inferred_strength: float = 0.5

    # sleep
    auto_sleep: bool = True
    sleep_passes: int = 40
    sleep_rounds: int = 4
    batch: int = 32
    grow_tolerance: float = 0.15
    grow_factor: float = 0.5


@dataclass
class Thought:
    probs: np.ndarray
    top: int
    confidence: float
    window: int
    source: str
    steps: int = 1
    system: int = 1
    path: list[str] = field(default_factory=list)


@dataclass
class Reply:
    text: str
    tokens: list[int]
    abstained: bool = False
    system: int = 1
    steps: int = 0
    window: int = 0
    confidence: float = 0.0
    sources: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    correct: bool | None = None
    pain: float = 0.0
    pleasure: float = 0.0
    slept: bool = False

    def __str__(self) -> str:
        return self.text


class Mind:
    def __init__(self, config: Config | None = None, **overrides):
        self.cfg = config or Config(**overrides)
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.vocab = Vocab()
        self.memory = EpisodicMemory(cfg.max_context, cfg.context_limit,
                                     cfg.capacity, bos=self.vocab.bos)
        self.cortex = Cortex(len(self.vocab), cfg.dim, cfg.hidden,
                             cfg.cortex_window, cfg.lr, cfg.seed)
        self.pain = PainSystem(enabled=cfg.affect)
        self.drives = DriveSystem(enabled=cfg.affect)
        self.energy = 1.0
        self.tick = 0
        self.familiar: set[tuple[int, ...]] = set()
        self.forbidden: dict[tuple[int, ...], dict[int, float]] = {}
        self.journal: list[dict] = []
        self.moments: list[dict] = []
        self.asked: set[tuple[int, ...]] = set()
        self.open_questions: list[tuple[int, ...]] = []
        self.last: dict | None = None
        self.sleeps = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ ego

    def _vigilance(self) -> float:
        return self.pain.total()

    def _confidence_needed(self) -> float:
        """How sure habit must be before the ego lets it answer.

        Pain raises the bar, exhaustion lowers it, and an unmet wish to
        master something makes the effort of thinking more acceptable.
        """
        cfg = self.cfg
        bar = cfg.confidence_calm + (cfg.confidence_hurt - cfg.confidence_calm) * self._vigilance()
        bar += 0.1 * self.drives.tension("mastery")
        bar -= 0.3 * self.pain.level("exhaustion")
        return float(np.clip(bar, 0.2, 0.97))

    def _abstain_below(self) -> float:
        cfg = self.cfg
        return cfg.abstain_calm + (cfg.abstain_hurt - cfg.abstain_calm) * self._vigilance()

    def _spend(self, amount: float) -> None:
        self.energy = max(0.0, self.energy - amount)

    def _seen(self, context: list[int]) -> tuple[int, ...]:
        """What the cortex sees of a context; used to tell if it is new to it."""
        return tuple(context[-self.cfg.cortex_window:])

    def _bans(self, context: list[int]) -> dict[int, float]:
        return self.forbidden.get(tuple(context), {})

    # ------------------------------------------------------------- system 1

    def _system1(self, context: list[int]) -> Thought:
        cfg = self.cfg
        self._spend(cfg.energy_system1)
        size = len(self.vocab)
        rev, _ = self.memory.key(context)
        context_len = min(len(context), self.memory.L)
        support, window = 0.0, 0
        if cfg.use_memory:
            csl = self.memory.match(rev)
            votes, support, window, _ = self.memory.vote(csl, size, context_len, cfg.fixed_window)
        if cfg.use_cortex:
            prior = self.cortex.predict(rev)
        else:
            prior = np.full(size, 1.0 / size)
        # The cortex always has an opinion, even about situations it has
        # never been in. There it is believed only in part.
        if self._seen(context) not in self.familiar:
            prior = prior * cfg.novel_trust
        if support > 0:
            trust = support / (support + cfg.prior_weight)
            probs = trust * votes / votes.sum() + (1.0 - trust) * prior
            source = "memory"
        else:
            probs = prior.copy()
            window = min(context_len, cfg.cortex_window)
            source = "cortex"
        for token, weight in self._bans(context).items():
            probs[token] *= 1.0 - weight
        top = int(np.argmax(probs))
        return Thought(probs, top, float(probs[top]), window, source)

    # ------------------------------------------------------------- system 2

    def _think(self, context: list[int], depth: int = 0, closing: bool = False) -> Thought:
        cfg = self.cfg
        thought = self._system1(context)
        if not cfg.allow_system2 or self.energy < cfg.energy_system2:
            return thought
        if closing and thought.top == self.vocab.end:
            # Nothing is at stake in ending a sentence; habit decides.
            return thought
        if cfg.force_system2 or thought.confidence < self._confidence_needed():
            return self._system2(context, thought, depth)
        return thought

    def _system2(self, context: list[int], habit: Thought, depth: int) -> Thought:
        cfg = self.cfg
        self._spend(cfg.energy_system2)
        steps = habit.steps + 1
        path = ["recollect"]
        specials = (self.vocab.bos, self.vocab.unknown)

        # Recollect: is there a memory of exactly this situation?
        rev, _ = self.memory.key(context)
        exact = self.memory.exact(self.memory.match(rev)) if cfg.use_memory else np.zeros(0, bool)
        if exact.any():
            votes = np.zeros(len(self.vocab))
            np.add.at(votes, self.memory.nxt[: self.memory.n][exact],
                      self.memory.strength[: self.memory.n][exact])
            for token, weight in self._bans(context).items():
                votes[token] *= 1.0 - weight
            top = int(np.argmax(votes))
            share = float(votes[top] / max(votes.sum(), 1e-12))
            if share >= 0.5:
                return Thought(votes / max(votes.sum(), 1e-12), top, share,
                               min(len(context), self.memory.L), "recollected",
                               steps, 2, path)

        # Reduce: resolve the longest prefix habit is sure of, put its
        # answer in place of the prefix, and think again about the rest.
        if habit.confidence < cfg.confidence_calm and depth < cfg.max_depth:
            for cut in range(len(context) - 1, 2, -1):
                if self.energy < cfg.energy_system2:
                    break
                self._spend(cfg.energy_system2)
                part = self._system1(context[:cut])
                steps += 1
                if part.confidence < cfg.reduce_confidence or part.top in specials \
                        or part.top == self.vocab.end:
                    continue
                reduced = [self.vocab.bos, part.top] + context[cut:]
                inner = self._think(reduced, depth + 1)
                steps += inner.steps
                banned = self._bans(context).get(inner.top, 0.0) >= 0.5
                if inner.confidence >= cfg.confidence_calm and inner.top not in specials \
                        and not banned:
                    step = (f"{self.vocab.decode(context[1:cut])} -> "
                            f"{self.vocab.words[part.top]}")
                    return Thought(inner.probs, inner.top,
                                   min(part.confidence, inner.confidence),
                                   len(context) - 1, "derived", steps, 2,
                                   path + ["reduce: " + step] + inner.path)
            if depth == 0:
                self.pain.inflict("frustration", 0.2)
                path.append("no way through")

        habit.steps = steps
        habit.system = 2
        habit.path = path
        return habit

    # ------------------------------------------------------------- speaking

    def _generate(self, context: list[int]):
        tokens: list[int] = []
        thoughts: list[Thought] = []
        abstained = False
        for _ in range(self.cfg.max_new_tokens):
            thought = self._think(context + tokens, closing=bool(tokens))
            if thought.confidence < self._abstain_below():
                if not tokens:
                    tokens.append(self.vocab.unknown)
                    thoughts.append(thought)
                    abstained = True
                break
            weight = self._bans(context + tokens).get(thought.top, 0.0)
            if weight > 0:
                self.pain.inflict("guilt", weight)
            tokens.append(thought.top)
            thoughts.append(thought)
            if thought.top == self.vocab.end:
                break
        return tokens, thoughts, abstained

    # ------------------------------------------------------------- learning

    def _experience(self, context: list[int], outcome: int, pos: int,
                    source: int, strength: float) -> float:
        """Live through one moment: predict it, feel the error, keep it."""
        cfg = self.cfg
        thought = self._system1(context)
        p_outcome = float(thought.probs[outcome])
        wrong = thought.top != outcome
        felt = self.pain.inflict("surprise", max(0.0, thought.confidence - p_outcome)) if wrong else 0.0
        self.drives.satisfy("curiosity", 0.3 * (1.0 - min(1.0, p_outcome)))

        rev, truncated = self.memory.key(context)
        span = 1
        if cfg.fixed_window is None:
            lesson = self.memory.learn_spans(rev, truncated, outcome, self.memory.match(rev))
            if lesson.needs_longer_context and self.memory.grow_context():
                rev, truncated = self.memory.key(context)
                lesson = self.memory.learn_spans(rev, truncated, outcome, self.memory.match(rev))
            span = lesson.new_span
            if felt > 0:
                # Only the belief that spoke and was wrong is shaken. Other
                # outcomes this context has had are left alone.
                held = lesson.contradicted
                blamed = held[self.memory.nxt[held] == thought.top]
                self.memory.destabilise(blamed, cfg.revise_rate * felt)

        hurting = self.pain.total()
        self.memory.observe(rev, truncated, outcome, pos,
                            strength * (1.0 + felt),
                            1.0 + cfg.pain_salience * hurting,
                            source, self.tick, span)
        self.familiar.add(self._seen(context))
        if cfg.use_cortex:
            self.cortex.learn(rev[None, :], np.array([outcome]),
                              1.0 + cfg.pain_plasticity * hurting)
        if cfg.capacity is not None and self.memory.count > cfg.capacity:
            room = max(1, cfg.capacity // 20)
            _, lost = self.memory.evict(self.memory.count - cfg.capacity + room)
            self.pain.inflict("loss", lost)
        return felt

    def _absorb(self, sequence: list[int], source: int = HEARD,
                strength: float = 1.0, start: int = 1) -> float:
        base = self.memory.append_stream(sequence)
        felt = 0.0
        for t in range(start, len(sequence)):
            felt = max(felt, self._experience(sequence[:t], sequence[t], base + t,
                                              source, strength))
        return felt

    def _encode(self, text: str) -> list[int]:
        tokens = self.vocab.encode(text)
        self.cortex.ensure_vocab(len(self.vocab))
        return tokens

    # ---------------------------------------------------------- interaction

    def interact(self, text: str, expect: str | None = None) -> Reply:
        """Hear `text`; answer it if it is unfinished; learn from it always.

        A text ending in "." is a statement: every word of it is learned.
        Anything else is a question: the model completes it. A question is
        a demand, not a fact, so its own words are not learned as facts;
        what is learned is the answer (the true one if `expect` is given,
        otherwise the one the model worked out), and if there was no
        answer, the question itself is kept open.
        """
        tokens = self._encode(text)
        context = [self.vocab.bos] + tokens
        pain_before = self.pain.total()
        if tokens and tokens[-1] == self.vocab.end and expect is None:
            self._absorb(context)
            reply = Reply("", [])
            self.last = None
            kind = "statement"
        else:
            produced, thoughts, abstained = self._generate(context)
            answer = [t for t in produced if t != self.vocab.end]
            reply = Reply(
                text=self.vocab.decode(answer),
                tokens=answer,
                abstained=abstained,
                system=max(t.system for t in thoughts),
                steps=sum(t.steps for t in thoughts),
                window=thoughts[0].window,
                confidence=thoughts[0].confidence,
                sources=[t.source for t in thoughts],
                path=[step for t in thoughts for step in t.path],
            )
            if reply.system == 2:
                self.moments.append({"tick": self.tick, "about": text,
                                     "path": reply.path, "answer": reply.text})
            self.last = {"context": context, "produced": produced,
                         "thoughts": thoughts, "abstained": abstained}
            if expect is not None:
                truth = self._encode(expect)
                if truth and truth[-1] == self.vocab.end:
                    truth = truth[:-1]
                reply.correct = answer == truth
                self._absorb(context + truth + [self.vocab.end], start=len(context))
                self._settle(context)
                if reply.correct:
                    self.drives.satisfy("bond", 0.3)
                    if reply.system == 2:
                        self.drives.satisfy("mastery", 0.5)
            elif abstained:
                self.memory.append_stream(context)
                self._wonder(context)
            else:
                self._keep_own_words(context, produced, thoughts)
            kind = "query"
        reply.slept = self._end_of_moment()
        reply.pain = self.pain.total()
        reply.pleasure = self.drives.pleasure
        self.journal.append({
            "tick": self.tick, "kind": kind, "heard": text, "said": reply.text,
            "expect": expect, "system": reply.system, "steps": reply.steps,
            "window": reply.window, "pain_before": round(pain_before, 4),
            "pain_after": round(reply.pain, 4),
        })
        return reply

    def _wonder(self, context: list[int]) -> None:
        """A question it could not answer stays open and feeds curiosity."""
        question = tuple(context[1:])
        if question not in self.open_questions:
            self.open_questions.append(question)
        if self.cfg.affect:
            curiosity = self.drives.drives["curiosity"]
            curiosity.tension = min(1.0, curiosity.tension + 0.2)

    def _settle(self, context: list[int]) -> None:
        question = tuple(context[1:])
        if question in self.open_questions:
            self.open_questions.remove(question)

    def _keep_own_words(self, context, produced, thoughts) -> None:
        """What the model works out for itself becomes memory too."""
        cfg = self.cfg
        base = self.memory.append_stream(context + produced)
        for i, (token, thought) in enumerate(zip(produced, thoughts)):
            here = context + produced[:i]
            rev, truncated = self.memory.key(here)
            if thought.source == "derived":
                lesson = self.memory.learn_spans(rev, truncated, token, self.memory.match(rev))
                self.memory.observe(rev, truncated, token, base + len(here),
                                    cfg.inferred_strength, 1.0, INFERRED,
                                    self.tick, lesson.new_span)
                self.familiar.add(self._seen(here))
                self.drives.satisfy("mastery", 0.3)
            elif (rev.tobytes(), token) in self.memory.index:
                self.memory.observe(rev, truncated, token, base + len(here),
                                    cfg.said_strength, 1.0, SAID, self.tick)

    def feedback(self, value: float) -> None:
        """Praise (value > 0) or rebuke (value < 0) for the last answer."""
        if self.last is None or value == 0:
            return
        cfg = self.cfg
        context, produced = self.last["context"], self.last["produced"]
        answer = [t for t in produced if t not in (self.vocab.end, self.vocab.unknown)]
        if value < 0:
            felt = self.pain.inflict("rebuke", abs(value) * (1.0 + self.drives.tension("bond")))
            if felt > 0:
                for i, token in enumerate(produced):
                    if token not in answer:
                        continue
                    here = context + produced[:i]
                    bans = self.forbidden.setdefault(tuple(here), {})
                    bans[token] = max(bans.get(token, 0.0), min(0.95, abs(value)))
                    rev, _ = self.memory.key(here)
                    csl = self.memory.match(rev)
                    _, _, _, voters = self.memory.vote(csl, len(self.vocab),
                                                       len(here), cfg.fixed_window)
                    blamed = np.flatnonzero(voters & (self.memory.nxt[: self.memory.n] == token))
                    self.memory.destabilise(blamed, cfg.revise_rate * felt)
        else:
            self.drives.satisfy("bond", value)
            self.pain.soothe(0.3 * min(1.0, value))
            if answer:
                base = self.memory.append_stream(context + produced)
                for i, token in enumerate(produced):
                    here = context + produced[:i]
                    rev, truncated = self.memory.key(here)
                    self.forbidden.get(tuple(here), {}).pop(token, None)
                    lesson = self.memory.learn_spans(rev, truncated, token, self.memory.match(rev))
                    self.memory.observe(rev, truncated, token, base + len(here),
                                        min(1.0, value), 1.0, HEARD, self.tick,
                                        lesson.new_span)
                    self.familiar.add(self._seen(here))
                if any(t.source == "derived" for t in self.last["thoughts"]):
                    self.drives.satisfy("mastery", 0.5)
        self.journal.append({"tick": self.tick, "kind": "feedback", "value": value,
                             "pain_after": round(self.pain.total(), 4)})
        self._end_of_moment()

    # ----------------------------------------------------------- homeostasis

    def _end_of_moment(self) -> bool:
        cfg = self.cfg
        self.tick += 1
        self.energy = min(1.0, self.energy + cfg.energy_recover)
        if self.energy < cfg.exhaustion_at:
            self.pain.inflict("exhaustion", (cfg.exhaustion_at - self.energy) / cfg.exhaustion_at)
        if cfg.capacity is not None and self.memory.count >= 0.9 * cfg.capacity:
            slots = np.flatnonzero(self.memory.alive[: self.memory.n])
            unconsolidated = float((self.memory.mastery[slots] < 0.5).mean())
            self.pain.inflict("loss", 0.5 * unconsolidated)
        slept = False
        if cfg.auto_sleep and (self.pain.level("exhaustion") > 0.5
                               or self.pain.level("loss") > 0.25):
            self.sleep()
            slept = True
        self.pain.tick()
        self.drives.tick(self.pain)
        self.history.append({"tick": self.tick, **self.size()})
        return slept

    def sleep(self) -> dict:
        """Replay what was lived into the cortex; grow it if it cannot hold it."""
        cfg = self.cfg
        memory, cortex = self.memory, self.cortex
        memory.compact()
        report = {"traces": memory.n, "grown": 0, "pruned": 0, "evicted": 0,
                  "residual_before": None, "residual_after": None}
        if memory.n and cfg.use_cortex:
            ctx = memory.ctx[: memory.n, : cfg.cortex_window]
            targets = memory.nxt[: memory.n]
            share = memory.shares()
            weight = memory.salience[: memory.n] * share

            def residual():
                memory.mastery[: memory.n] = cortex.score(ctx, targets) / np.maximum(share, 1e-9)
                short = np.clip(1.0 - memory.mastery[: memory.n], 0.0, 1.0)
                return float((weight * short).sum() / weight.sum()), short

            left, short = residual()
            report["residual_before"] = left
            steps = max(1, cfg.sleep_passes * memory.n // (cfg.batch * cfg.sleep_rounds))
            for _ in range(cfg.sleep_rounds * 3):
                if left <= cfg.grow_tolerance / 2:
                    break
                priority = memory.salience[: memory.n] * (short + 0.1)
                priority /= priority.sum()
                for _ in range(steps):
                    batch = self.rng.choice(memory.n, size=min(cfg.batch, memory.n), p=priority)
                    cortex.learn(ctx[batch], targets[batch])
                before, (left, short) = left, residual()
                stuck = left > cfg.grow_tolerance and left > 0.7 * before
                if stuck:
                    self.pain.inflict("frustration", left)
                    wants_room = self.pain.level("frustration") > 0.1 if cfg.affect else True
                    if cfg.grow_cortex and wants_room and cortex.hidden < cfg.max_hidden:
                        units = max(4, int(cortex.hidden * cfg.grow_factor))
                        units = min(units, cfg.max_hidden - cortex.hidden)
                        cortex.grow(units)
                        report["grown"] += units
            report["pruned"] = cortex.prune()
            self.drives.satisfy("mastery", max(0.0, report["residual_before"] - left))
            report["residual_after"] = left
        if cfg.capacity is not None and memory.count > 0.8 * cfg.capacity:
            evicted, lost = memory.evict(memory.count - int(0.6 * cfg.capacity))
            report["evicted"] = evicted
            self.pain.inflict("loss", lost)
            memory.compact()
        self.energy = 1.0
        self.pain.soothe(0.7)
        self.sleeps += 1
        self.journal.append({"tick": self.tick, "kind": "sleep", **report})
        return report

    # --------------------------------------------------------------- desire

    def ask(self, sample: int = 48) -> str | None:
        """Form a question out of what it wants to know.

        Candidates are new pairings of words it has seen open a sentence or
        end one with words it has seen in second place. Curiosity picks the
        pairing it is least sure about. When the wish to choose for itself
        is the strongest drive, it picks at random instead.
        """
        memory, vocab = self.memory, self.vocab
        capricious = (not self.cfg.affect) or self.drives.dominant() == "autonomy"
        while self.open_questions and not capricious:
            question = self.open_questions.pop(0)
            if self._system1([vocab.bos, *question]).confidence < self.cfg.confidence_calm:
                self.asked.add(question)
                return vocab.decode(question)
        alive = np.flatnonzero(memory.alive[: memory.n])
        second = {int(memory.nxt[s]) for s in alive
                  if memory.ctx[s, 1] == vocab.bos and memory.nxt[s] != vocab.end}
        specials = {vocab.bos, vocab.unknown, vocab.end}
        first = [t for t in range(len(vocab)) if t not in second and t not in specials]
        pairs = [(a, b) for a in first for b in second if (a, b) not in self.asked]
        if not pairs:
            return None
        picks = self.rng.choice(len(pairs), size=min(sample, len(pairs)), replace=False)
        pairs = [pairs[i] for i in picks]
        if capricious:
            choice = pairs[0]
            self.drives.satisfy("autonomy", 1.0)
        else:
            doubt = [1.0 - self._system1([vocab.bos, a, b]).confidence for a, b in pairs]
            choice = pairs[int(np.argmax(doubt))]
            if self.cfg.affect:
                self.drives.drives["autonomy"].tension = min(
                    1.0, self.drives.drives["autonomy"].tension + 0.05)
        self.asked.add(choice)
        return vocab.decode(choice)

    # ----------------------------------------------------------- inspection

    def size(self) -> dict:
        return {
            "vocab": len(self.vocab),
            "hidden_units": self.cortex.hidden,
            "cortex_weights": self.cortex.n_params,
            "traces": self.memory.count,
            "context_limit": self.memory.L,
            "stream": len(self.memory.stream),
        }

    def report(self) -> dict:
        worst, level = self.pain.worst()
        want = self.drives.dominant()
        if not self.cfg.affect:
            feeling = "Affect is switched off."
        elif level < 0.05:
            feeling = f"Nothing hurts. What I want most: {want}."
        else:
            feeling = f"What hurts most: {worst} ({level:.2f}). What I want most: {want}."
        return {
            "feeling": feeling,
            "pain": self.pain.snapshot(),
            "pain_total": round(self.pain.total(), 4),
            "drives": self.drives.snapshot(),
            "pleasure": round(self.drives.pleasure, 4),
            "energy": round(self.energy, 4),
            "confidence_needed": round(self._confidence_needed(), 4),
            "size": self.size(),
            "sleeps": self.sleeps,
            "interactions": len(self.journal),
        }

    def fingerprint(self) -> str:
        """A hash of everything learnable; changes whenever the model does."""
        digest = hashlib.sha256()
        for name in sorted(self.cortex.params):
            digest.update(self.cortex.params[name].tobytes())
        memory = self.memory
        digest.update(memory.ctx[: memory.n].tobytes())
        digest.update(memory.nxt[: memory.n].tobytes())
        digest.update(memory.strength[: memory.n].tobytes())
        digest.update(memory.span[: memory.n].tobytes())
        digest.update(memory.alive[: memory.n].tobytes())
        digest.update(str(len(memory.stream)).encode())
        digest.update(repr(sorted((k, sorted(v.items())) for k, v in self.forbidden.items())).encode())
        return digest.hexdigest()

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: str) -> "Mind":
        with open(path, "rb") as handle:
            mind = pickle.load(handle)
        if not isinstance(mind, Mind):
            raise TypeError(f"{path} does not hold a Mind")
        return mind
