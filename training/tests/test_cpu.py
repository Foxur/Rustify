"""CPU tests for the packed trainer (run in CI and before every GPU session): pytest -q training/tests"""

from __future__ import annotations

import numpy as np
import torch

from arbitro_train import selftest
from arbitro_train.bench.train_throughput import PaddedModel, _padded_view, flops_per_token
from arbitro_train.config import PRESETS, tiny_config
from arbitro_train.losses import gather_logits, proper_score_loss
from arbitro_train.model import DecisionModel, init_like_hf, load_encoder_state_dict
from arbitro_train.packing import PackSpec, synthetic_batch, to_device


def test_hf_parity_cpu():
    assert selftest.check_hf_parity("cpu")


def test_param_counts_match_laya_encoders():
    # docs/DECISIONS.md MEM1/MEM2: encoder parts of laya-en / laya-multilingual
    assert PRESETS["modernbert-large"].num_parameters() == 394_781_696
    assert PRESETS["mmbert-base"].num_parameters() == 306_939_648
    assert sum(p.numel() for p in DecisionModel(tiny_config()).encoder.parameters()) == tiny_config().num_parameters()


def test_synthetic_batch_static_shapes():
    spec = PackSpec(token_budget=1024, min_len=64, max_len=128, max_options=8)
    shapes = set()
    for seed in range(5):
        b = synthetic_batch(spec, 500, 1, seed, special_ids=(0, 1))
        cu = b["cu_seqlens"]
        assert cu[0] == 0 and cu[-1] == spec.token_budget and np.all(np.diff(cu) >= 0)
        assert len(cu) == spec.segments() + 1
        n = b["real_sequences"]
        assert b["real_tokens"] == cu[n]
        assert b["max_seqlen"] == spec.max_len  # tail is always shorter than max_len
        rows = b["marker_rows"][: b["real_markers"]]
        assert np.all(b["input_ids"][rows] == 1)
        seg = np.searchsorted(cu[1:], rows, side="right")
        for s in range(n):
            k = b["marker_mask"][s].sum()
            assert np.all(seg[b["marker_index"][s, :k]] == s)
        assert np.allclose(b["targets"][b["loss_weight"] > 0].sum(-1), 1.0, atol=1e-5)
        shapes.add(tuple((k, v.shape) for k, v in b.items() if isinstance(v, np.ndarray)))
    assert len(shapes) == 1, "every micro-batch must have identical shapes (torch.compile)"


def test_training_step_overfits_tiny():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = DecisionModel(cfg, head_layers=2, dropout=0.0)
    init_like_hf(model.encoder)
    b = to_device(synthetic_batch(PackSpec(token_budget=512, min_len=40, max_len=80, max_options=6), cfg.vocab_size, 1, 0, (0, 1)), "cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(40):
        z = model(b, attn_impl="sdpa")
        loss, _ = proper_score_loss(gather_logits(z, b["marker_index"], b["marker_mask"]), b["targets"], b["marker_mask"], b["qtype"], b["loss_weight"])
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
    assert losses[-1] < 0.8 * losses[0], losses[::10]


def test_padded_mode_matches_packed():
    """The padded-HF benchmark arm must compute the same function as the packed arm."""
    torch.manual_seed(0)
    cfg = tiny_config(local_attention=16)
    packed = DecisionModel(cfg, head_layers=2, dropout=0.0).eval()
    init_like_hf(packed.encoder)
    padded = PaddedModel(cfg, ckpt=False).eval()
    load_encoder_state_dict(packed.encoder, padded.encoder.state_dict())
    padded.head.load_state_dict(packed.head.state_dict())
    hb = synthetic_batch(PackSpec(token_budget=400, min_len=30, max_len=60, max_options=5), cfg.vocab_size, 1, 3, (0, 1))
    hbp = _padded_view(hb, 60, cfg.pad_token_id)
    with torch.no_grad():
        zp = packed(to_device(hb, "cpu"), attn_impl="sdpa")[: hb["real_markers"]]
        zh = padded(to_device(hbp, "cpu"), attn_impl="sdpa")[: hb["real_markers"]]
    assert torch.allclose(zp, zh, atol=1e-4), float((zp - zh).abs().max())


def test_proper_score_loss_is_minimised_at_target():
    t = torch.tensor([[0.7, 0.2, 0.1, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    w = torch.ones(1)
    good, _ = proper_score_loss(torch.log(t.clamp_min(1e-9)), t, mask, torch.tensor([1]), w)
    for other in ([0.5, 0.3, 0.2, 0.0], [0.9, 0.05, 0.05, 0.0], [0.34, 0.33, 0.33, 0.0]):
        o = torch.tensor([other])
        bad, _ = proper_score_loss(torch.log(o), t, mask, torch.tensor([1]), w)
        assert float(good) < float(bad)


def test_flops_estimate_matches_design():
    # docs/DECISIONS.md MEM4: ~0.78 GFLOP per token forward for ModernBERT-large + head at L = 512
    fwd = flops_per_token(PRESETS["modernbert-large"], 512.0) / 3
    assert 0.70e9 < fwd < 0.85e9, fwd


def test_varlen_callsite_contract(monkeypatch):
    """Exercise the varlen call site on CPU with a reference stand-in that enforces the
    torch.nn.attention.varlen.varlen_attn contract (shapes, int32 cu_seqlens, window tuple)."""
    import arbitro_train.model as M

    calls = []

    def fake_varlen(q, k, v, cu_q, cu_k, max_q, max_k, *, scale=None, window_size=(-1, -1), **kw):
        assert q.dim() == 3 and q.shape == k.shape == v.shape
        assert cu_q.dtype == torch.int32 and cu_k.dtype == torch.int32
        assert isinstance(max_q, int) and max_q >= int((cu_q[1:] - cu_q[:-1]).max())
        assert isinstance(window_size, tuple) and len(window_size) == 2
        calls.append(window_size)
        w = None if window_size[0] < 0 else window_size[0]
        return M.packed_attention(q, k, v, cu_q, max_q, w, impl="sdpa")

    monkeypatch.setattr(M, "_varlen_attn", fake_varlen)
    torch.manual_seed(0)
    cfg = tiny_config(local_attention=16)
    model = DecisionModel(cfg, head_layers=2, dropout=0.0).eval()
    init_like_hf(model.encoder)
    b = to_device(synthetic_batch(PackSpec(token_budget=300, min_len=30, max_len=60, max_options=5), cfg.vocab_size, 1, 1, (0, 1)), "cpu")
    with torch.no_grad():
        z_ref = model(b, attn_impl="sdpa")
        z_var = model(b, attn_impl="varlen")
    assert torch.allclose(z_ref, z_var, atol=1e-6)
    enc_windows = calls[: cfg.num_hidden_layers]
    assert enc_windows == [(-1, -1) if t == "full" else (8, 8) for t in cfg.layer_types]
    assert calls[cfg.num_hidden_layers:] == [(-1, -1)] * 2  # head layers: full attention
