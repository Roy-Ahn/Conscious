"""Affect: pain points and drives.

Pain and desire are the two state variables every other part of the model
reads. Pain says "what just happened must not happen again". Desire says
"something is missing". Neither is a loss function: both persist between
interactions, both have several independent channels, and both adapt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PainPoint:
    """One channel of pain.

    level      how much it hurts now, in [0, 1]
    gain       sensitivity; a hit of size x is felt as gain * x
    threshold  hits felt below this are not felt at all
    decay      fraction of the level that fades each tick
    """

    name: str
    level: float = 0.0
    gain: float = 1.0
    threshold: float = 0.02
    decay: float = 0.15
    gain_min: float = 0.25
    gain_max: float = 3.0
    sensitise_at: float = 0.5
    sensitise_rate: float = 0.1
    habituate_rate: float = 0.05
    recover_rate: float = 0.05
    hits: int = 0

    def inflict(self, magnitude: float) -> float:
        """Apply a hit and return how strongly it was felt, in [0, 1].

        Strong hits sensitise the point (the next one hurts more). Repeated
        mild hits habituate it (the next one hurts less).
        """
        if magnitude <= 0.0:
            return 0.0
        felt = min(1.0, self.gain * magnitude)
        if felt < self.threshold:
            return 0.0
        self.hits += 1
        self.level += (1.0 - self.level) * felt
        if felt >= self.sensitise_at:
            self.gain *= 1.0 + self.sensitise_rate * felt
        else:
            self.gain *= 1.0 - self.habituate_rate
        self.gain = min(self.gain_max, max(self.gain_min, self.gain))
        return felt

    def soothe(self, fraction: float) -> None:
        self.level *= 1.0 - min(1.0, max(0.0, fraction))

    def tick(self) -> None:
        self.level *= 1.0 - self.decay
        self.gain += self.recover_rate * (1.0 - self.gain)


DEFAULT_PAIN_POINTS = {
    # being confidently wrong about what comes next
    "surprise": dict(decay=0.20),
    # being told off by the interlocutor
    "rebuke": dict(decay=0.10),
    # effort that led nowhere, or a desire left unmet
    "frustration": dict(decay=0.15),
    # running out of energy
    "exhaustion": dict(decay=0.30),
    # losing a memory that had not been consolidated
    "loss": dict(decay=0.10),
    # doing again what was forbidden
    "guilt": dict(decay=0.08),
}


class PainSystem:
    """A set of pain points that can be extended at run time."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.points: dict[str, PainPoint] = {}
        for name, kwargs in DEFAULT_PAIN_POINTS.items():
            self.add_point(name, **kwargs)

    def add_point(self, name: str, **kwargs) -> PainPoint:
        point = PainPoint(name=name, **kwargs)
        self.points[name] = point
        return point

    def remove_point(self, name: str) -> None:
        self.points.pop(name, None)

    def inflict(self, name: str, magnitude: float) -> float:
        if not self.enabled:
            return 0.0
        return self.points[name].inflict(magnitude)

    def level(self, name: str) -> float:
        return self.points[name].level if self.enabled else 0.0

    def total(self) -> float:
        """Overall suffering in [0, 1]: one severe pain is enough to hurt."""
        if not self.enabled:
            return 0.0
        spared = 1.0
        for point in self.points.values():
            spared *= 1.0 - point.level
        return 1.0 - spared

    def worst(self) -> tuple[str, float]:
        point = max(self.points.values(), key=lambda p: p.level)
        return point.name, (point.level if self.enabled else 0.0)

    def soothe(self, fraction: float) -> None:
        for point in self.points.values():
            point.soothe(fraction)

    def tick(self) -> None:
        for point in self.points.values():
            point.tick()

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            name: {"level": round(p.level, 4), "gain": round(p.gain, 4), "hits": p.hits}
            for name, p in self.points.items()
        }


@dataclass
class Drive:
    """One desire.

    Tension builds while the desire is unmet and is discharged when it is
    satisfied. The discharge is what the model registers as pleasure.
    """

    name: str
    rate: float
    weight: float = 1.0
    tension: float = 0.0

    def tick(self) -> None:
        self.tension = min(1.0, self.tension + self.rate)

    def satisfy(self, amount: float) -> float:
        pleasure = max(0.0, min(self.tension, amount))
        self.tension -= pleasure
        return pleasure


DEFAULT_DRIVES = {
    # wants to know what it does not know
    "curiosity": dict(rate=0.04),
    # wants to overcome what resisted it (will to power)
    "mastery": dict(rate=0.03),
    # wants pain to stop (pleasure principle); tension tracks total pain
    "relief": dict(rate=0.0),
    # wants the interlocutor's approval
    "bond": dict(rate=0.02),
    # wants to choose for itself, even against its advantage (caprice)
    "autonomy": dict(rate=0.02),
}


class DriveSystem:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.drives: dict[str, Drive] = {
            name: Drive(name=name, **kwargs) for name, kwargs in DEFAULT_DRIVES.items()
        }
        self.pleasure = 0.0
        self.pleasure_decay = 0.3

    def add_drive(self, name: str, rate: float, weight: float = 1.0) -> Drive:
        drive = Drive(name=name, rate=rate, weight=weight)
        self.drives[name] = drive
        return drive

    def tension(self, name: str) -> float:
        return self.drives[name].tension if self.enabled else 0.0

    def satisfy(self, name: str, amount: float) -> float:
        if not self.enabled:
            return 0.0
        pleasure = self.drives[name].satisfy(amount)
        self.pleasure = min(1.0, self.pleasure + pleasure)
        return pleasure

    def dominant(self) -> str:
        """The drive that currently commands the others."""
        return max(self.drives.values(), key=lambda d: d.weight * d.tension).name

    def tick(self, pain: PainSystem) -> None:
        if not self.enabled:
            return
        self.pleasure *= 1.0 - self.pleasure_decay
        relief = self.drives["relief"]
        hurting = pain.total()
        if hurting < relief.tension:
            self.pleasure = min(1.0, self.pleasure + (relief.tension - hurting))
        relief.tension = hurting
        for drive in self.drives.values():
            if drive.name != "relief":
                drive.tick()
                if drive.tension >= 0.999:
                    pain.inflict("frustration", 0.05)

    def snapshot(self) -> dict[str, float]:
        return {name: round(d.tension, 4) for name, d in self.drives.items()}
