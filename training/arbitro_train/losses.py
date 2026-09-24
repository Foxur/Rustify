"""Analytic strictly-proper-score loss (docs/TRAINING.md §9.1, ADR-023).

Per question with reported distribution q (softmax over its option logits) and soft target t:

    loss = CE(t, q) - w_sph * spherical(t, q) + w_rps * RPS(t, q) [score questions only]

This is the negative of Laya's ``proper_reward`` (log score + spherical - RPS) minimised
analytically instead of through the RLCD sampling estimator. There is no hard log floor:
``log_softmax`` is stable, and a floor would zero the gradient on confidently wrong items
(docs/TRAINING.md §9.1, risk TR-5).
"""

from __future__ import annotations

import torch

QTYPE_SCORE = 1


def gather_logits(marker_logits: torch.Tensor, marker_index: torch.Tensor, marker_mask: torch.Tensor) -> torch.Tensor:
    """[M] marker logits -> [Q, K] padded logits; padded slots get -1e4 (as Laya does)."""
    z = marker_logits.index_select(0, marker_index.reshape(-1)).view(marker_index.shape)
    return z.masked_fill(~marker_mask, -1e4)


def proper_score_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    qtype: torch.Tensor,
    weight: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """logits/targets/mask: [Q, K]; qtype/weight: [Q]. Returns (mean loss over weighted questions, stats as 0-d tensors)."""
    logits = logits.float()
    logq = torch.log_softmax(logits, dim=-1)
    q = logq.exp() * mask
    t = targets.float() * mask
    ce = -(t * logq.masked_fill(~mask, 0.0)).sum(-1)
    sph = (t * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    loss = ce - w_sph * sph
    is_score = (qtype == QTYPE_SCORE).float()
    k = mask.sum(-1).clamp(min=2).float()
    rps = (((torch.cumsum(q, -1) - torch.cumsum(t, -1)) ** 2) * mask).sum(-1) / (k - 1)
    loss = loss + w_rps * rps * is_score
    w = weight.float()
    denom = w.sum().clamp_min(1.0)
    total = (loss * w).sum() / denom
    with torch.no_grad():
        acc = ((logits.argmax(-1) == t.argmax(-1)).float() * w).sum() / denom
        ce_mean = (ce * w).sum() / denom
    # Tensors, not floats: converting here would force a GPU sync every step.
    return total, {"ce": ce_mean, "acc": acc}
