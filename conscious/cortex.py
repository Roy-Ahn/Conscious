"""Cortex: the slow, parametric learner.

A small network that predicts the next token from the most recent ones. It
learns a little from every experience while awake and a lot from replay
while asleep. Its size is not fixed: rows are added when new words are
heard, hidden units are added when it cannot hold what it has lived, and
units that carry nothing are removed.
"""

from __future__ import annotations

import numpy as np


class Cortex:
    def __init__(self, vocab_size: int, dim: int = 16, hidden: int = 24,
                 window: int = 6, lr: float = 0.01, seed: int = 0):
        self.dim = dim
        self.window = window
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        inputs = window * dim
        self.params = {
            "E": self.rng.normal(0.0, 0.3, (vocab_size, dim)),
            "W1": self.rng.normal(0.0, 1.0 / np.sqrt(inputs), (hidden, inputs)),
            "b1": np.zeros(hidden),
            "W2": np.zeros((vocab_size, hidden)),
            "b2": np.zeros(vocab_size),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0

    @property
    def vocab_size(self) -> int:
        return self.params["E"].shape[0]

    @property
    def hidden(self) -> int:
        return self.params["W1"].shape[0]

    @property
    def n_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def _extend(self, name: str, block: np.ndarray, axis: int) -> None:
        self.params[name] = np.concatenate([self.params[name], block], axis=axis)
        zeros = np.zeros_like(block)
        self.m[name] = np.concatenate([self.m[name], zeros], axis=axis)
        self.v[name] = np.concatenate([self.v[name], zeros], axis=axis)

    def ensure_vocab(self, vocab_size: int) -> int:
        """Add embedding and output rows for words not seen before."""
        extra = vocab_size - self.vocab_size
        if extra <= 0:
            return 0
        self._extend("E", self.rng.normal(0.0, 0.3, (extra, self.dim)), 0)
        self._extend("W2", np.zeros((extra, self.hidden)), 0)
        self._extend("b2", np.zeros(extra), 0)
        return extra

    def grow(self, units: int) -> None:
        """Add hidden units without changing what the network computes."""
        inputs = self.window * self.dim
        self._extend("W1", self.rng.normal(0.0, 1.0 / np.sqrt(inputs), (units, inputs)), 0)
        self._extend("b1", np.zeros(units), 0)
        self._extend("W2", np.zeros((self.vocab_size, units)), 1)

    def prune(self, tolerance: float = 1e-4, keep_at_least: int = 4) -> int:
        """Remove hidden units whose output weights carry nothing."""
        norms = np.abs(self.params["W2"]).max(axis=0)
        keep = norms > tolerance
        if keep.sum() < keep_at_least:
            keep[np.argsort(-norms)[:keep_at_least]] = True
        removed = int((~keep).sum())
        if removed == 0:
            return 0
        for store in (self.params, self.m, self.v):
            store["W1"] = store["W1"][keep]
            store["b1"] = store["b1"][keep]
            store["W2"] = store["W2"][:, keep]
        return removed

    def _forward(self, ctx: np.ndarray):
        ctx = np.atleast_2d(ctx)[:, : self.window]
        if ctx.shape[1] < self.window:
            pad = np.full((ctx.shape[0], self.window - ctx.shape[1]), -1, dtype=ctx.dtype)
            ctx = np.concatenate([ctx, pad], axis=1)
        present = ctx >= 0
        emb = self.params["E"][np.where(present, ctx, 0)] * present[..., None]
        x = emb.reshape(ctx.shape[0], -1)
        h = np.tanh(x @ self.params["W1"].T + self.params["b1"])
        logits = h @ self.params["W2"].T + self.params["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        return ctx, present, x, h, p

    def predict(self, ctx: np.ndarray) -> np.ndarray:
        return self._forward(ctx)[4][0]

    def score(self, ctx: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Probability the cortex alone gives to each target."""
        out = np.empty(len(targets))
        for start in range(0, len(targets), 1024):
            sl = slice(start, start + 1024)
            p = self._forward(ctx[sl])[4]
            out[sl] = p[np.arange(p.shape[0]), targets[sl]]
        return out

    def gradients(self, ctx: np.ndarray, targets: np.ndarray):
        ctx, present, x, h, p = self._forward(ctx)
        batch = ctx.shape[0]
        rows = np.arange(batch)
        p_target = p[rows, targets]
        loss = float(-np.log(np.maximum(p_target, 1e-12)).mean())
        d_logits = p.copy()
        d_logits[rows, targets] -= 1.0
        d_logits /= batch
        d_h = d_logits @ self.params["W2"]
        d_z = d_h * (1.0 - h * h)
        d_x = (d_z @ self.params["W1"]).reshape(batch, self.window, self.dim)
        d_x *= present[..., None]
        d_E = np.zeros_like(self.params["E"])
        np.add.at(d_E, np.where(present, ctx, 0), d_x)
        grads = {
            "E": d_E,
            "W1": d_z.T @ x,
            "b1": d_z.sum(axis=0),
            "W2": d_logits.T @ h,
            "b2": d_logits.sum(axis=0),
        }
        return loss, p_target, grads

    def learn(self, ctx: np.ndarray, targets: np.ndarray, lr_scale: float = 1.0):
        """One Adam step. `lr_scale` is where pain raises plasticity."""
        loss, p_target, grads = self.gradients(ctx, np.atleast_1d(targets))
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        step = self.lr * lr_scale * np.sqrt(1 - b2 ** self.t) / (1 - b1 ** self.t)
        for name, grad in grads.items():
            self.m[name] = b1 * self.m[name] + (1 - b1) * grad
            self.v[name] = b2 * self.v[name] + (1 - b2) * grad * grad
            self.params[name] -= step * self.m[name] / (np.sqrt(self.v[name]) + eps)
        return loss, p_target
