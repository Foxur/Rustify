"""Encoder configurations for the packed ModernBERT module.

The presets mirror the Hugging Face configs of the backbones named in
docs/TRAINING.md §5.1. Throughput spikes build models from these presets with
random weights, so they need no download; real weights load by key name
(see ``model.load_encoder_state_dict``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    # "full" or "sliding" per layer.
    layer_types: tuple[str, ...]
    global_rope_theta: float
    local_rope_theta: float
    # Total window; each side sees local_attention // 2 tokens (|i - j| <= 64 for 128).
    local_attention: int = 128
    norm_eps: float = 1e-5
    pad_token_id: int = 0
    mask_token_id: int = 0
    max_position_embeddings: int = 8192
    name: str = field(default="custom", compare=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def window_one_side(self) -> int:
        return self.local_attention // 2

    def num_parameters(self) -> int:
        d, i, v, n = self.hidden_size, self.intermediate_size, self.vocab_size, self.num_hidden_layers
        per_layer = 3 * d * d + d * d + d * 2 * i + i * d + 2 * d  # Wqkv, Wo, Wi, Wo, 2 norms
        return v * d + d + n * per_layer - d + d  # embeddings + emb norm + layers (layer 0 has no attn_norm) + final norm

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_hf_dict(cls, d: dict, name: str = "custom", mask_token_id: int | None = None) -> "EncoderConfig":
        """Build from a Hugging Face ModernBERT ``config.json`` (new or legacy format)."""
        n = d["num_hidden_layers"]
        if d.get("layer_types"):
            types = tuple("full" if t == "full_attention" else "sliding" for t in d["layer_types"])
        else:
            every = d.get("global_attn_every_n_layers", 3)
            types = tuple("full" if i % every == 0 else "sliding" for i in range(n))
        rp = d.get("rope_parameters")
        if isinstance(rp, dict) and "full_attention" in rp:
            g_theta = float(rp["full_attention"]["rope_theta"])
            l_theta = float((rp.get("sliding_attention") or rp["full_attention"])["rope_theta"])
        else:
            g_theta = float(d.get("global_rope_theta", 160000.0))
            l_theta = float(d.get("local_rope_theta") or g_theta)
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            num_hidden_layers=n,
            num_attention_heads=d["num_attention_heads"],
            intermediate_size=d["intermediate_size"],
            layer_types=types,
            global_rope_theta=g_theta,
            local_rope_theta=l_theta,
            local_attention=d.get("local_attention", 128),
            norm_eps=d.get("norm_eps", 1e-5),
            pad_token_id=d.get("pad_token_id", 0) or 0,
            mask_token_id=mask_token_id if mask_token_id is not None else d.get("mask_token_id", 0) or 0,
            max_position_embeddings=d.get("max_position_embeddings", 8192),
            name=name,
        )


def _every_third(n: int) -> tuple[str, ...]:
    return tuple("full" if i % 3 == 0 else "sliding" for i in range(n))


PRESETS: dict[str, EncoderConfig] = {
    # answerdotai/ModernBERT-large (Apache-2.0); laya-en's encoder.
    "modernbert-large": EncoderConfig(
        vocab_size=50368, hidden_size=1024, num_hidden_layers=28, num_attention_heads=16,
        intermediate_size=2624, layer_types=_every_third(28), global_rope_theta=160000.0,
        local_rope_theta=10000.0, pad_token_id=50283, mask_token_id=50284, name="modernbert-large",
    ),
    # answerdotai/ModernBERT-base (Apache-2.0).
    "modernbert-base": EncoderConfig(
        vocab_size=50368, hidden_size=768, num_hidden_layers=22, num_attention_heads=12,
        intermediate_size=1152, layer_types=_every_third(22), global_rope_theta=160000.0,
        local_rope_theta=10000.0, pad_token_id=50283, mask_token_id=50284, name="modernbert-base",
    ),
    # jhu-clsp/mmBERT-base (MIT; Gemma-2-derived tokenizer, see Q8); laya-multilingual's encoder.
    "mmbert-base": EncoderConfig(
        vocab_size=256000, hidden_size=768, num_hidden_layers=22, num_attention_heads=12,
        intermediate_size=1152, layer_types=_every_third(22), global_rope_theta=160000.0,
        local_rope_theta=160000.0, pad_token_id=0, mask_token_id=4, name="mmbert-base",
    ),
}

# Hugging Face repositories for real weights (downloaded on the training machine only).
HF_REPOS = {
    "modernbert-large": "answerdotai/ModernBERT-large",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "mmbert-base": "jhu-clsp/mmBERT-base",
}


def tiny_config(local_attention: int = 16, layers: int = 4, vocab: int = 128) -> EncoderConfig:
    """Small config for CPU tests; window 8 per side so short sequences exercise it."""
    return EncoderConfig(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=layers, num_attention_heads=4,
        intermediate_size=96, layer_types=_every_third(layers), global_rope_theta=160000.0,
        local_rope_theta=10000.0, local_attention=local_attention, pad_token_id=0,
        mask_token_id=1, name="tiny",
    )
