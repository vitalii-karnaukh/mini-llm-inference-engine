"""Model config, read from a Hugging Face `config.json`."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Shape and numeric settings of a Qwen2 / Llama-style decoder.

    Field names match Hugging Face `config.json` keys, so loading is a direct copy.
    """

    hidden_size: int  # d: width of each token's vector
    intermediate_size: int  # MLP inner width
    num_hidden_layers: int  # L
    num_attention_heads: int  # query heads
    num_key_value_heads: int  # K/V heads (GQA when < num_attention_heads)
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = False
    attention_bias: bool = True  # Qwen2 has bias on q/k/v projections; Llama does not
    torch_dtype: str = "bfloat16"
    model_type: str = "qwen2"

    @property
    def head_dim(self) -> int:
        """Size of one attention head: hidden_size / num_attention_heads."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_queries_per_kv(self) -> int:
        """How many query heads share one K/V head (GQA group size)."""
        return self.num_attention_heads // self.num_key_value_heads

    def kv_bytes_per_token(self, bytes_per_value: int = 2) -> int:
        """KV cache bytes for one token: 2 (K and V) x layers x kv_heads x head_dim x bytes."""
        return (
            2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * bytes_per_value
        )

    @classmethod
    def from_dict(cls, raw: dict) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in raw.items() if k in known}
        # Qwen2 configs do not always state attention_bias; Qwen2 always uses q/k/v bias.
        if "attention_bias" not in raw and raw.get("model_type") == "llama":
            values["attention_bias"] = False
        return cls(**values)

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> ModelConfig:
        path = Path(model_dir) / "config.json"
        with path.open() as f:
            return cls.from_dict(json.load(f))
