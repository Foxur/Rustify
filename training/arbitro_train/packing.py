"""Packed micro-batches with static shapes (docs/TRAINING.md §8.1–§8.2).

A micro-batch is a flat stream of exactly ``token_budget`` tokens. Real sequences are packed
first; the unused rest becomes one dummy tail segment that attends only to itself and is
excluded from the loss. ``cu_seqlens`` is padded to a fixed length with zero-length segments,
and markers/questions are padded to fixed maxima, so ``torch.compile`` sees one shape.

The M0 spikes use ``synthetic_batch``: random token ids, one question per sequence (layout L0)
and random soft targets. Throughput does not depend on token values, so this measures the real
trainer cost without any data licence question. The Rust data core (arbitro-data) replaces
this generator in M3b and must emit the same fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class PackSpec:
    token_budget: int = 12288          # T (C3)
    min_len: int = 256                 # sequence length range, inclusive
    max_len: int = 512
    max_segments: int = 0              # 0 = derived: token_budget // min_len + 2
    min_options: int = 2
    max_options: int = 20
    max_options_padded: int = 0        # K; 0 = max_options

    def segments(self) -> int:
        return self.max_segments or self.token_budget // max(1, self.min_len) + 2

    def k(self) -> int:
        return self.max_options_padded or self.max_options


def synthetic_batch(spec: PackSpec, vocab_size: int, mask_token_id: int, seed: int,
                    special_ids: tuple[int, ...] = ()) -> dict:
    """Build one packed micro-batch on the CPU (numpy); ``to_device`` moves it."""
    rng = np.random.default_rng(seed)
    t = spec.token_budget
    n_seg_max = spec.segments()
    k_max = spec.k()
    lengths: list[int] = []
    used = 0
    while True:
        n = int(rng.integers(spec.min_len, spec.max_len + 1))
        if used + n > t or len(lengths) >= n_seg_max - 1:  # keep one slot for the tail
            break
        lengths.append(n)
        used += n
    tail = t - used

    low = max(special_ids) + 1 if special_ids else 0
    input_ids = rng.integers(low, vocab_size, size=t, dtype=np.int64)
    position_ids = np.zeros(t, dtype=np.int64)
    token_qtype = np.zeros(t, dtype=np.int64)
    marker_rows: list[int] = []
    marker_index = np.full((n_seg_max, k_max), 0, dtype=np.int64)
    marker_mask = np.zeros((n_seg_max, k_max), dtype=bool)
    targets = np.zeros((n_seg_max, k_max), dtype=np.float32)
    qtype = np.zeros(n_seg_max, dtype=np.int64)
    loss_weight = np.zeros(n_seg_max, dtype=np.float32)

    start = 0
    for s, n in enumerate(lengths):
        position_ids[start:start + n] = np.arange(n)
        qt = int(rng.integers(0, 3))
        k = 2 if qt == 2 else int(rng.integers(spec.min_options, min(spec.max_options, n - 2) + 1))
        k = min(k, k_max)
        pos = np.sort(rng.choice(np.arange(1, n - 1), size=k, replace=False)) + start
        input_ids[pos] = mask_token_id
        token_qtype[start:start + n] = qt
        for j, p in enumerate(pos):
            marker_index[s, j] = len(marker_rows)
            marker_rows.append(int(p))
        marker_mask[s, :k] = True
        logits = rng.normal(size=k) * 2.0
        p = np.exp(logits - logits.max())
        targets[s, :k] = p / p.sum()
        qtype[s] = qt
        loss_weight[s] = 1.0
        start += n
    if tail:
        position_ids[start:] = np.arange(tail)

    # cu_seqlens: real segments, the tail, then zero-length padding segments.
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if tail:
        cu.append(t)
    while len(cu) < n_seg_max + 1:
        cu.append(t)

    # Pad marker rows to a fixed count (points at row 0; masked out of the loss).
    m_max = n_seg_max * k_max
    real_markers = len(marker_rows)
    marker_rows_arr = np.zeros(m_max, dtype=np.int64)
    marker_rows_arr[:real_markers] = marker_rows

    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_qtype": token_qtype,
        "cu_seqlens": np.asarray(cu, dtype=np.int32),
        "max_seqlen": int(max(spec.max_len, tail)),
        "marker_rows": marker_rows_arr,
        "marker_index": marker_index,
        "marker_mask": marker_mask,
        "targets": targets,
        "qtype": qtype,
        "loss_weight": loss_weight,
        "real_tokens": int(used),
        "real_sequences": len(lengths),
        "real_markers": real_markers,
    }


def to_device(batch: dict, device: torch.device | str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v).to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def batch_from_sequences(seqs: list[list[int]], markers: list[list[int]], qtypes: list[int],
                         token_budget: int | None = None, max_segments: int | None = None,
                         k_max: int | None = None) -> dict:
    """Pack explicit token sequences (e.g. from Laya's build_sequence) into the same format.

    Used by the self-test and later by small real-data experiments. With ``token_budget``
    the stream is padded by a dummy tail segment and zero-length segments like above."""
    lengths = [len(s) for s in seqs]
    used = sum(lengths)
    t = token_budget or used
    if used > t:
        raise ValueError(f"{used} tokens exceed the budget {t}")
    n_seg = max_segments or (len(seqs) + (1 if t > used else 0))
    k_max = k_max or max(len(m) for m in markers)
    input_ids = np.zeros(t, dtype=np.int64)
    position_ids = np.zeros(t, dtype=np.int64)
    token_qtype = np.zeros(t, dtype=np.int64)
    marker_rows: list[int] = []
    marker_index = np.zeros((n_seg, k_max), dtype=np.int64)
    marker_mask = np.zeros((n_seg, k_max), dtype=bool)
    qtype = np.zeros(n_seg, dtype=np.int64)
    loss_weight = np.zeros(n_seg, dtype=np.float32)
    start = 0
    for s, (ids, mk, qt) in enumerate(zip(seqs, markers, qtypes)):
        n = len(ids)
        input_ids[start:start + n] = ids
        position_ids[start:start + n] = np.arange(n)
        token_qtype[start:start + n] = qt
        for j, p in enumerate(mk):
            marker_index[s, j] = len(marker_rows)
            marker_rows.append(start + p)
        marker_mask[s, :len(mk)] = True
        qtype[s] = qt
        loss_weight[s] = 1.0
        start += n
    if t > used:
        position_ids[used:] = np.arange(t - used)
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    if t > used:
        cu.append(t)
    while len(cu) < n_seg + 1:
        cu.append(t)
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "token_qtype": token_qtype,
        "cu_seqlens": np.asarray(cu, dtype=np.int32),
        "max_seqlen": int(max(lengths + [t - used])),
        "marker_rows": np.asarray(marker_rows, dtype=np.int64),
        "marker_index": marker_index,
        "marker_mask": marker_mask,
        "targets": marker_mask.astype(np.float32) / np.maximum(marker_mask.sum(-1, keepdims=True), 1),
        "qtype": qtype,
        "loss_weight": loss_weight,
        "real_tokens": used,
        "real_sequences": len(seqs),
        "real_markers": len(marker_rows),
    }
