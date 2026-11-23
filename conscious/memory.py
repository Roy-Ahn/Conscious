"""Episodic memory: one-shot traces over a lifelong stream of experience.

Everything the model lives through is appended to `stream`. A trace points
at one moment in that stream: "after this context came this token". A trace
is written the first time something happens, so a single exposure is enough
to recall it.

Each trace carries a `span`: how many of the most recent context tokens
must match before the trace is allowed to speak. Spans are learned. A trace
starts by claiming as little context as possible and lengthens its span
only when it turns out to be confused with a different outcome. The context
the model reads for a prediction is therefore set by what it has learned
about that situation, not by a constant.

`L`, the longest context a trace can hold, is not a constant either: when
two different outcomes cannot be told apart within `L` tokens and the
stream holds more context for both, `L` doubles and traces are re-read
from the stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEARD, SAID, INFERRED = 0, 1, 2
PAD = -1


@dataclass
class SpanLesson:
    new_span: int
    contradicted: np.ndarray
    needs_longer_context: bool


class EpisodicMemory:
    def __init__(self, max_context: int = 8, context_limit: int = 256,
                 capacity: int | None = None, bos: int = 0):
        self.L = max_context
        self.context_limit = context_limit
        self.capacity = capacity
        self.bos = bos
        self.stream: list[int] = []
        self.n = 0
        self.index: dict[tuple[bytes, int], int] = {}
        self._allocate(256)

    def _allocate(self, size: int) -> None:
        def grown(name, shape, dtype, fill):
            new = np.full(shape, fill, dtype=dtype)
            old = getattr(self, name, None)
            if old is not None:
                new[: self.n] = old[: self.n]
            setattr(self, name, new)

        grown("ctx", (size, self.L), np.int32, PAD)
        grown("nxt", (size,), np.int32, 0)
        grown("pos", (size,), np.int64, 0)
        grown("truncated", (size,), bool, False)
        grown("strength", (size,), np.float64, 0.0)
        grown("salience", (size,), np.float64, 1.0)
        grown("span", (size,), np.int32, 1)
        grown("born", (size,), np.int32, 0)
        grown("used", (size,), np.int32, 0)
        grown("source", (size,), np.int8, HEARD)
        grown("mastery", (size,), np.float64, 0.0)
        grown("alive", (size,), bool, False)
        self.size = size

    @property
    def count(self) -> int:
        return int(self.alive[: self.n].sum())

    def append_stream(self, tokens: list[int]) -> int:
        base = len(self.stream)
        self.stream.extend(int(t) for t in tokens)
        return base

    def key(self, context: list[int]) -> tuple[np.ndarray, bool]:
        """Most recent token first, padded past the start of the episode."""
        rev = np.full(self.L, PAD, dtype=np.int32)
        tail = context[-self.L:]
        rev[: len(tail)] = tail[::-1]
        return rev, len(context) > self.L

    def match(self, rev: np.ndarray) -> np.ndarray:
        """For every trace, how many of the most recent tokens agree."""
        same = self.ctx[: self.n] == rev[None, :]
        return np.cumprod(same, axis=1).sum(axis=1)

    def vote(self, csl: np.ndarray, vocab_size: int, context_len: int,
             fixed_window: int | None = None, generalise: float = 0.15):
        """Let every trace whose span is satisfied vote for its outcome.

        A trace of exactly the present situation counts in full. A trace of
        a different situation that merely ends the same way is generalising
        and counts for `generalise` of its strength: one example is not a
        rule, several agreeing examples are.

        Returns (votes, support, window, voters). `window` is the number of
        context tokens the vote depended on.
        """
        alive = self.alive[: self.n]
        if fixed_window is None:
            need = self.span[: self.n]
        else:
            need = np.full(self.n, min(fixed_window, self.L), dtype=np.int32)
            generalise = 1.0
        voters = alive & (csl >= need)
        votes = np.zeros(vocab_size)
        if not voters.any():
            return votes, 0.0, 0, voters
        depth = csl[voters]
        credit = self.strength[: self.n][voters] * np.where(depth >= self.L, 1.0, generalise)
        np.add.at(votes, self.nxt[: self.n][voters], credit * (1.0 + depth))
        window = int(min(need[voters].max(), context_len))
        return votes, float(credit.sum()), window, voters

    def exact(self, csl: np.ndarray) -> np.ndarray:
        """Traces whose whole stored context is the present context."""
        return self.alive[: self.n] & (csl >= self.L)

    def learn_spans(self, rev: np.ndarray, truncated: bool, outcome: int,
                    csl: np.ndarray) -> SpanLesson:
        """Adjust spans so that no trace speaks for a different outcome."""
        alive = self.alive[: self.n]
        other = alive & (self.nxt[: self.n] != outcome)
        identical = other & (csl >= self.L)
        separable = other & (csl < self.L)
        confused = separable & (csl >= self.span[: self.n])
        self.span[: self.n][confused] = csl[confused] + 1
        new_span = int(csl[separable].max()) + 1 if separable.any() else 1
        needs_longer = bool(
            truncated and self.L < self.context_limit
            and (identical & self.truncated[: self.n]).any()
        )
        return SpanLesson(new_span, np.flatnonzero(identical), needs_longer)

    def observe(self, rev: np.ndarray, truncated: bool, outcome: int, pos: int,
                strength: float, salience: float, source: int, tick: int,
                span: int = 1) -> tuple[int, bool]:
        """Write a trace, or strengthen it if this exact moment is known."""
        key = (rev.tobytes(), int(outcome))
        slot = self.index.get(key)
        if slot is not None:
            self.strength[slot] += strength
            self.salience[slot] = max(self.salience[slot], salience)
            self.source[slot] = min(self.source[slot], source)
            self.used[slot] = tick
            self.pos[slot] = pos
            return slot, False
        if self.n == self.size:
            self._allocate(self.size * 2)
        slot = self.n
        self.n += 1
        self.ctx[slot] = rev
        self.nxt[slot] = outcome
        self.pos[slot] = pos
        self.truncated[slot] = truncated
        self.strength[slot] = strength
        self.salience[slot] = salience
        self.span[slot] = span
        self.born[slot] = tick
        self.used[slot] = tick
        self.source[slot] = source
        self.mastery[slot] = 0.0
        self.alive[slot] = True
        self.index[key] = slot
        return slot, True

    def kill(self, slots) -> None:
        for slot in np.atleast_1d(slots):
            slot = int(slot)
            if self.alive[slot]:
                self.alive[slot] = False
                self.index.pop((self.ctx[slot].tobytes(), int(self.nxt[slot])), None)

    def destabilise(self, slots, fraction: float, floor: float = 0.05) -> None:
        """Weaken traces that were shown to be wrong; forget them if spent."""
        slots = np.atleast_1d(slots)
        if slots.size == 0 or fraction <= 0.0:
            return
        self.strength[slots] *= 1.0 - min(1.0, fraction)
        self.kill(slots[self.strength[slots] < floor])

    def shares(self) -> np.ndarray:
        """For each trace, its share of the strength of its exact context.

        A context that has had several outcomes cannot be predicted with
        certainty; the share is the best any predictor can do.
        """
        totals: dict[bytes, float] = {}
        keys = []
        for slot in range(self.n):
            key = self.ctx[slot].tobytes()
            keys.append(key)
            if self.alive[slot]:
                totals[key] = totals.get(key, 0.0) + self.strength[slot]
        share = np.zeros(self.n)
        for slot in range(self.n):
            if self.alive[slot]:
                share[slot] = self.strength[slot] / totals[keys[slot]]
        return share

    def evict(self, count: int) -> tuple[int, float]:
        """Drop the traces least worth keeping.

        Returns how many were dropped and how much of what was dropped had
        not been consolidated (0 = nothing was lost, 1 = all of it was).
        """
        slots = np.flatnonzero(self.alive[: self.n])
        if count <= 0 or slots.size == 0:
            return 0, 0.0
        count = min(count, slots.size)
        worth = (self.strength[slots] * self.salience[slots]
                 * (1.05 - np.clip(self.mastery[slots], 0.0, 1.0)))
        victims = slots[np.argsort(worth, kind="stable")[:count]]
        lost = float((1.0 - np.clip(self.mastery[victims], 0.0, 1.0)).mean())
        self.kill(victims)
        return int(count), lost

    def compact(self) -> None:
        keep = np.flatnonzero(self.alive[: self.n])
        if keep.size == self.n:
            return
        for name in ("ctx", "nxt", "pos", "truncated", "strength", "salience",
                     "span", "born", "used", "source", "mastery", "alive"):
            array = getattr(self, name)
            array[: keep.size] = array[keep]
        self.n = int(keep.size)
        self.alive[self.n:] = False
        self._reindex()

    def _reindex(self) -> None:
        self.index = {
            (self.ctx[slot].tobytes(), int(self.nxt[slot])): slot
            for slot in range(self.n) if self.alive[slot]
        }

    def grow_context(self) -> bool:
        """Double the longest context a trace can hold."""
        if self.L >= self.context_limit:
            return False
        self.compact()
        self.L = min(self.L * 2, self.context_limit)
        self.ctx = np.full((self.size, self.L), PAD, dtype=np.int32)
        for slot in range(self.n):
            end = int(self.pos[slot])
            start = end
            while start > 0 and end - start < self.L:
                start -= 1
                if self.stream[start] == self.bos:
                    break
            context = self.stream[start:end]
            self.ctx[slot, : len(context)] = context[::-1]
            self.truncated[slot] = bool(context) and context[0] != self.bos
        self._reindex()
        return True
