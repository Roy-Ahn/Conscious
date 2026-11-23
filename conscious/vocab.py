"""A vocabulary that grows whenever a new word is heard."""

from __future__ import annotations

import re

BOS = "<s>"
UNKNOWN = "<?>"
END = "."

_TOKEN = re.compile(r"<\?>|<s>|[^\s.?!,;:]+|[.?!,;:]")


def split(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Vocab:
    def __init__(self):
        self.words: list[str] = []
        self.ids: dict[str, int] = {}
        for word in (BOS, UNKNOWN, END):
            self.add(word)

    def __len__(self) -> int:
        return len(self.words)

    def add(self, word: str) -> int:
        if word not in self.ids:
            self.ids[word] = len(self.words)
            self.words.append(word)
        return self.ids[word]

    def encode(self, text: str) -> list[int]:
        return [self.add(word) for word in split(text)]

    def decode(self, ids) -> str:
        return " ".join(self.words[int(i)] for i in ids)

    @property
    def bos(self) -> int:
        return self.ids[BOS]

    @property
    def unknown(self) -> int:
        return self.ids[UNKNOWN]

    @property
    def end(self) -> int:
        return self.ids[END]
