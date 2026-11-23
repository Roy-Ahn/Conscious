"""Small synthetic worlds used by the tests and experiments.

Every world is a set of facts of the form "subject relation object". The
names are made up, so nothing can be known in advance.
"""

from __future__ import annotations

import numpy as np

_ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"]
_VOWELS = ["a", "e", "i", "o", "u"]


def names(count: int, rng: np.random.Generator, taken: set[str] | None = None) -> list[str]:
    taken = set() if taken is None else taken
    out = []
    while len(out) < count:
        name = "".join(rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(3))
        if name not in taken:
            taken.add(name)
            out.append(name)
    return out


class World:
    """Chains of people linked by `parent`, each living in a `home`."""

    def __init__(self, chains: int = 10, length: int = 5, places: int = 6, seed: int = 0):
        rng = np.random.default_rng(seed)
        taken: set[str] = {"parent", "home"}
        self.places = names(places, rng, taken)
        self.people: list[str] = []
        self.parent: dict[str, str] = {}
        self.home: dict[str, str] = {}
        for _ in range(chains):
            chain = names(length, rng, taken)
            self.people += chain
            for child, parent in zip(chain, chain[1:]):
                self.parent[child] = parent
            for person in chain:
                self.home[person] = str(rng.choice(self.places))

    def facts(self) -> list[tuple[str, str]]:
        """(question, answer) for every single fact."""
        out = [(f"{c} parent", p) for c, p in self.parent.items()]
        out += [(f"{c} home", h) for c, h in self.home.items()]
        return out

    def statements(self) -> list[str]:
        return [f"{q} {a} ." for q, a in self.facts()]

    def hops(self, depth: int) -> list[tuple[str, str]]:
        """Questions that need `depth` facts chained: parent ... parent [home]."""
        out = []
        for person in self.people:
            current, ok = person, True
            for _ in range(depth - 1):
                if current not in self.parent:
                    ok = False
                    break
                current = self.parent[current]
            if not ok:
                continue
            prefix = " ".join([person] + ["parent"] * (depth - 1))
            if current in self.parent:
                out.append((f"{prefix} parent", self.parent[current]))
            out.append((f"{prefix} home", self.home[current]))
        return out


def depth_rules(depths: list[int], per_depth: int, seed: int = 0):
    """Rules whose answer depends on exactly the last `d` words.

    For each depth d, rules come in pairs that share their last d-1 words
    and differ in the word before, so d-1 words are never enough. Words
    before the last d are noise.

    Returns (rules, noise) where rules is a list of (depth, words, answer).
    """
    rng = np.random.default_rng(seed)
    taken: set[str] = set()
    noise = names(30, rng, taken)
    rules = []
    for depth in depths:
        for _ in range(per_depth // 2):
            shared = names(depth - 1, rng, taken)
            for _ in range(2):
                head = names(1, rng, taken)
                answer = names(1, rng, taken)[0]
                rules.append((depth, head + shared, answer))
    return rules, noise


def with_noise(words: list[str], noise: list[str], rng: np.random.Generator,
               low: int = 1, high: int = 3) -> str:
    prefix = [str(w) for w in rng.choice(noise, size=int(rng.integers(low, high + 1)))]
    return " ".join(prefix + words)
