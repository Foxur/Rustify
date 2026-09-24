"""Packed (unpadded) ModernBERT encoder and a laya-v1-style decision head for training.

Why our own module (docs/TRAINING.md §7.3): transformers 5.x runs ModernBERT padded, so
every pad token costs compute. Here a micro-batch is a flat token stream of T tokens,
segments are delimited by ``cu_seqlens`` and positions restart at 0 in every segment.

Attention backends:
  * ``varlen``: ``torch.nn.attention.varlen.varlen_attn`` (FlashAttention-2 on sm_80+), with
    window (w, w) on sliding layers and (-1, -1) on global layers. Needs CUDA and
    bf16/fp16 inputs (i.e. autocast). This is the training path.
  * ``sdpa``: dense ``scaled_dot_product_attention`` with a block-diagonal (+ window) boolean
    mask. O(T^2) memory; used on CPU, in fp32 and as the reference in the self-test.

Numerics follow the Hugging Face reference: LayerNorm without bias (eps 1e-5), fused Wqkv
laid out [q | k | v], half-split RoPE computed in fp32, scale 1/sqrt(head_dim), exact-erf
GELU in the GeGLU MLP, layer 0 without attention norm, a final LayerNorm.
Sliding window: key j is visible from query i iff |i - j| <= local_attention // 2 inside the
same segment (docs/ANALYSIS.md §4.4).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EncoderConfig

AttnImpl = Literal["auto", "varlen", "sdpa"]

try:  # torch >= 2.10
    from torch.nn.attention.varlen import varlen_attn as _varlen_attn
except Exception:  # pragma: no cover - older torch
    _varlen_attn = None


def varlen_available() -> bool:
    return _varlen_attn is not None and torch.cuda.is_available()


def _choose_impl(impl: AttnImpl, q: torch.Tensor) -> str:
    if impl != "auto":
        return impl
    if _varlen_attn is not None and q.is_cuda and q.dtype in (torch.bfloat16, torch.float16):
        return "varlen"
    return "sdpa"


def segment_ids_from_cu(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """[T] segment index per token from cumulative lengths (zero-length segments allowed)."""
    idx = torch.arange(total, device=cu_seqlens.device)
    return torch.bucketize(idx, cu_seqlens[1:].to(idx.dtype), right=True)


def dense_mask(cu_seqlens: torch.Tensor, total: int, window: int | None) -> torch.Tensor:
    """Boolean [T, T] mask, True = may attend: same segment and (optionally) |i - j| <= window."""
    seg = segment_ids_from_cu(cu_seqlens, total)
    mask = seg[:, None] == seg[None, :]
    if window is not None and window >= 0:
        i = torch.arange(total, device=cu_seqlens.device)
        mask &= (i[:, None] - i[None, :]).abs() <= window
    return mask


def packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    window: int | None,
    impl: AttnImpl = "auto",
    mask_cache: dict | None = None,
) -> torch.Tensor:
    """q, k, v: [T, H, Dh] -> [T, H, Dh]. ``window`` is one-sided (None = global)."""
    scale = q.shape[-1] ** -0.5
    chosen = _choose_impl(impl, q)
    if chosen == "varlen":
        if _varlen_attn is None:
            raise RuntimeError("torch.nn.attention.varlen.varlen_attn is not available in this torch build")
        ws = (window, window) if window is not None else (-1, -1)
        cu = cu_seqlens.to(torch.int32)
        return _varlen_attn(q, k, v, cu, cu, max_seqlen, max_seqlen, scale=scale, window_size=ws)
    total = q.shape[0]
    key = ("mask", window, total)
    mask = mask_cache.get(key) if mask_cache is not None else None
    if mask is None:
        mask = dense_mask(cu_seqlens, total, window)
        if mask_cache is not None:
            mask_cache[key] = mask
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=mask[None, None],
        scale=scale,
    )
    return out.squeeze(0).transpose(0, 1)


def rope_cos_sin(position_ids: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 [T, head_dim] cos/sin tables, built exactly like transformers (fp32 inv_freq)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=position_ids.device) / head_dim))
    freqs = position_ids.to(torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, H, Dh]; computed in fp32 and cast back (as the HF reference does)."""
    xf = x.float()
    out = xf * cos[:, None, :] + _rotate_half(xf) * sin[:, None, :]
    return out.to(x.dtype)


class ModernBertAttention(nn.Module):
    def __init__(self, cfg: EncoderConfig, layer_idx: int):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.window = cfg.window_one_side if cfg.layer_types[layer_idx] == "sliding" else None
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=False)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen, impl, mask_cache):
        t = x.shape[0]
        qkv = self.Wqkv(x).view(t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = packed_attention(q, k, v, cu_seqlens, max_seqlen, self.window, impl, mask_cache)
        return self.Wo(out.reshape(t, -1))


class ModernBertMLP(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        inp, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(F.gelu(inp) * gate)  # exact erf GELU


class ModernBertLayer(nn.Module):
    def __init__(self, cfg: EncoderConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = (
            nn.Identity() if layer_idx == 0 else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        )
        self.attn = ModernBertAttention(cfg, layer_idx)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)
        self.mlp = ModernBertMLP(cfg)
        self.is_global = cfg.layer_types[layer_idx] == "full"

    def forward(self, x, rope, cu_seqlens, max_seqlen, impl, mask_cache):
        cos, sin = rope["global" if self.is_global else "local"]
        x = x + self.attn(self.attn_norm(x), cos, sin, cu_seqlens, max_seqlen, impl, mask_cache)
        return x + self.mlp(self.mlp_norm(x))


class ModernBertEmbeddings(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(self, input_ids):
        return self.norm(self.tok_embeddings(input_ids))


class PackedModernBert(nn.Module):
    """ModernBERT over a packed token stream. Parameter names match HF ``ModernBertModel``."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.config = cfg
        self.embeddings = ModernBertEmbeddings(cfg)
        self.layers = nn.ModuleList(ModernBertLayer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        attn_impl: AttnImpl = "auto",
    ) -> torch.Tensor:
        cfg = self.config
        x = self.embeddings(input_ids)
        rope = {"global": rope_cos_sin(position_ids, cfg.head_dim, cfg.global_rope_theta)}
        rope["local"] = (
            rope["global"]
            if cfg.local_rope_theta == cfg.global_rope_theta
            else rope_cos_sin(position_ids, cfg.head_dim, cfg.local_rope_theta)
        )
        mask_cache: dict = {}
        for layer in self.layers:
            x = layer(x, rope, cu_seqlens, max_seqlen, attn_impl, mask_cache)
        return self.final_norm(x)


def init_like_hf(model: PackedModernBert, initializer_range: float = 0.02, cutoff: float = 2.0) -> None:
    """Truncated-normal init as in transformers' ModernBertPreTrainedModel (for random-weight spikes)."""
    n = model.config.num_hidden_layers
    std_out = initializer_range / math.sqrt(2.0 * n)
    for name, p in model.named_parameters():
        if p.dim() == 1:
            nn.init.ones_(p)
            continue
        std = std_out if name.endswith(("attn.Wo.weight", "mlp.Wo.weight")) else initializer_range
        nn.init.trunc_normal_(p, mean=0.0, std=std, a=-cutoff * std, b=cutoff * std)


# --------------------------------------------------------------------------------------------
# Decision head (laya-v1 style: type embedding, small transformer, per-marker scorer)
# --------------------------------------------------------------------------------------------


class HeadLayer(nn.Module):
    """Pre-norm transformer layer with biases and a ReLU FFN, like torch's TransformerEncoderLayer
    (norm_first=True) that Laya's head uses; attention runs packed (full, per segment)."""

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = max(1, d // 64)
        self.head_dim = d // self.n_heads
        self.norm1 = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.linear1 = nn.Linear(d, 4 * d)
        self.linear2 = nn.Linear(4 * d, d)
        self.dropout = dropout

    def forward(self, x, cu_seqlens, max_seqlen, impl, mask_cache):
        t = x.shape[0]
        qkv = self.in_proj(self.norm1(x)).view(t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        a = packed_attention(q, k, v, cu_seqlens, max_seqlen, None, impl, mask_cache)
        x = x + F.dropout(self.out_proj(a.reshape(t, -1)), self.dropout, self.training)
        h = F.dropout(F.relu(self.linear1(self.norm2(x))), self.dropout, self.training)
        return x + F.dropout(self.linear2(h), self.dropout, self.training)


class DecisionHead(nn.Module):
    """Adds type_emb[qtype] to every token, runs ``head_layers`` packed layers, scores markers.

    ``readout="hybrid"`` adds the mean of each option's text tokens to its marker state (the E1
    setup, docs/TRAINING.md §4.5); ``"marker"`` is Laya's read-out. Returns fp32 logits for the
    marker rows; the last scorer projection runs in fp32 (docs/TRAINING.md §7.3)."""

    def __init__(self, d: int, head_layers: int = 2, dropout: float = 0.1, readout: str = "marker"):
        super().__init__()
        if readout not in ("marker", "hybrid"):
            raise ValueError(f"unknown readout {readout!r}")
        self.readout = readout
        self.type_emb = nn.Embedding(3, d)
        self.layers = nn.ModuleList(HeadLayer(d, dropout) for _ in range(head_layers))
        self.scorer_norm = nn.LayerNorm(d)
        self.scorer_fc = nn.Linear(d, d)
        self.scorer_out = nn.Linear(d, 1)

    def forward(self, h, token_qtype, cu_seqlens, max_seqlen, marker_rows, attn_impl: AttnImpl = "auto",
                span_rows=None, span_owner=None):
        x = h + self.type_emb(token_qtype).to(h.dtype)
        mask_cache: dict = {}
        for layer in self.layers:
            x = layer(x, cu_seqlens, max_seqlen, attn_impl, mask_cache)
        m = x.index_select(0, marker_rows)
        if self.readout == "hybrid":
            if span_rows is None:
                raise ValueError("hybrid read-out needs span_rows/span_owner in the batch")
            n = marker_rows.shape[0]
            rows = x.index_select(0, span_rows).float()
            sums = rows.new_zeros(n + 1, rows.shape[1]).index_add_(0, span_owner, rows)
            cnt = rows.new_zeros(n + 1).index_add_(0, span_owner, torch.ones_like(span_owner, dtype=rows.dtype))
            m = m + (sums[:n] / cnt[:n].clamp_min(1.0)[:, None]).to(m.dtype)
        m = F.gelu(self.scorer_fc(self.scorer_norm(m)))
        with torch.autocast(device_type=m.device.type, enabled=False):
            return F.linear(m.float(), self.scorer_out.weight.float(), self.scorer_out.bias.float()).squeeze(-1)


class DecisionModel(nn.Module):
    def __init__(self, cfg: EncoderConfig, head_layers: int = 2, dropout: float = 0.1, readout: str = "marker"):
        super().__init__()
        self.encoder = PackedModernBert(cfg)
        self.head = DecisionHead(cfg.hidden_size, head_layers, dropout, readout)

    def forward(self, batch: dict, attn_impl: AttnImpl = "auto") -> torch.Tensor:
        h = self.encoder(batch["input_ids"], batch["position_ids"], batch["cu_seqlens"], batch["max_seqlen"], attn_impl)
        return self.head(h, batch["token_qtype"], batch["cu_seqlens"], batch["max_seqlen"], batch["marker_rows"], attn_impl,
                         batch.get("span_rows"), batch.get("span_owner"))


# --------------------------------------------------------------------------------------------
# Weight loading
# --------------------------------------------------------------------------------------------

_DROP_PREFIXES = ("head.", "decoder.", "classifier.")


def load_encoder_state_dict(model: PackedModernBert, state_dict: dict[str, torch.Tensor]) -> dict:
    """Load HF ModernBERT weights. Accepts keys with prefix ``model.`` (Hub MLM checkpoints),
    ``encoder.`` (Laya checkpoints) or none (``ModernBertModel.state_dict()``); MLM/classifier
    head keys are dropped. Raises on missing or unexpected encoder keys."""
    prefixes = ("model.", "encoder.")
    mapped = {}
    for k, v in state_dict.items():
        for p in prefixes:
            if k.startswith(p):
                k = k[len(p):]
                break
        else:
            if k.startswith(_DROP_PREFIXES):
                continue
        if k.startswith(_DROP_PREFIXES) or k.endswith("inv_freq"):
            continue
        mapped[k] = v
    own = model.state_dict()
    missing = sorted(set(own) - set(mapped))
    unexpected = sorted(set(mapped) - set(own))
    if missing or unexpected:
        raise KeyError(f"encoder key mismatch: missing={missing[:8]}... ({len(missing)}), unexpected={unexpected[:8]}... ({len(unexpected)})")
    for k, v in mapped.items():
        if own[k].shape != v.shape:
            raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(own[k].shape)}")
    model.load_state_dict({k: v.to(own[k].dtype) for k, v in mapped.items()}, strict=True)
    return {"loaded": len(mapped)}


def load_pretrained_encoder(repo_or_dir: str, cfg: EncoderConfig | None = None, revision: str | None = None) -> PackedModernBert:
    """Download (or read) a HF ModernBERT checkpoint and load it into ``PackedModernBert``."""
    import json
    import os

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    path = repo_or_dir
    if not os.path.isdir(path):
        path = snapshot_download(repo_or_dir, revision=revision, allow_patterns=["config.json", "*.safetensors"])
    with open(os.path.join(path, "config.json")) as f:
        hf_cfg = json.load(f)
    cfg = cfg or EncoderConfig.from_hf_dict(hf_cfg, name=repo_or_dir)
    model = PackedModernBert(cfg)
    sd: dict[str, torch.Tensor] = {}
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(path, fn)))
    load_encoder_state_dict(model, sd)
    return model
