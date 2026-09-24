"""Evaluation metrics for typed decisions (docs/TRAINING.md §17.2 subset used by E1).

ECE uses 15 equal-width bins with the first bin including 0, matching Laya's common.ece_score.
"""

from __future__ import annotations

import numpy as np


def ece15(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


class Accumulator:
    """Collects per-question probabilities and gold indices; reports per-source metrics."""

    def __init__(self):
        self.rows: dict[str, list[tuple[np.ndarray, int]]] = {}

    def add(self, source: str, probs: np.ndarray, gold: int) -> None:
        self.rows.setdefault(source, []).append((probs, gold))

    def summary(self) -> dict[str, dict]:
        out = {}
        for src, rows in self.rows.items():
            conf = np.array([p.max() for p, _ in rows])
            correct = np.array([float(p.argmax() == g) for p, g in rows])
            nll = np.array([-np.log(max(p[g], 1e-12)) for p, g in rows])
            brier = np.array([((p - np.eye(len(p))[g]) ** 2).sum() for p, g in rows])
            chance = np.array([1.0 / len(p) for p, _ in rows])
            lo, hi = bootstrap_ci(correct)
            out[src] = {
                "n": len(rows),
                "acc": float(correct.mean()),
                "acc_ci95": [lo, hi],
                "chance": float(chance.mean()),
                "nll": float(nll.mean()),
                "brier": float(brier.mean()),
                "ece15": ece15(conf, correct),
                "mean_conf": float(conf.mean()),
            }
        return out


def bootstrap_ci(correct: np.ndarray, iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(correct)
    means = correct[rng.integers(0, n, size=(iters, n))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
