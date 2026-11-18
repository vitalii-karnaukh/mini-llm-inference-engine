"""Load safetensors checkpoints into our modules.

Our module names mirror Hugging Face names (e.g. `model.layers.0.self_attn.q_proj.weight`),
so most keys load without renaming. Pass `rename` for anything that differs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

LM_HEAD_KEY = "lm_head.weight"
EMBED_KEY = "model.embed_tokens.weight"


def checkpoint_files(model_dir: str | Path) -> list[Path]:
    """All `*.safetensors` files in a model directory, sorted."""
    files = sorted(Path(model_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")
    return files


def load_state_dict(
    model_dir: str | Path,
    dtype: torch.dtype | None = None,
    rename: Callable[[str], str | None] | None = None,
) -> dict[str, torch.Tensor]:
    """Read every tensor from the checkpoint into a dict on CPU.

    Args:
        model_dir: directory with `*.safetensors` files.
        dtype: cast floating-point tensors to this dtype (None keeps the stored dtype).
        rename: maps a checkpoint key to our key; return None to skip the tensor.
    """
    state: dict[str, torch.Tensor] = {}
    for path in checkpoint_files(model_dir):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                new_key = rename(key) if rename else key
                if new_key is None:
                    continue
                tensor = f.get_tensor(key)
                if dtype is not None and tensor.is_floating_point():
                    tensor = tensor.to(dtype)
                state[new_key] = tensor
    return state


def load_weights(
    model: nn.Module,
    model_dir: str | Path,
    dtype: torch.dtype | None = None,
    rename: Callable[[str], str | None] | None = None,
    tie_word_embeddings: bool = False,
) -> None:
    """Load a checkpoint into `model`, strictly.

    With tied embeddings the checkpoint has no `lm_head.weight`; we reuse the
    embedding matrix for it, as Hugging Face does.

    Raises:
        KeyError: on any missing or unexpected key, listing them all.
        ValueError: on any shape mismatch, listing them all.
    """
    state = load_state_dict(model_dir, dtype=dtype, rename=rename)
    expected = model.state_dict()

    if tie_word_embeddings and LM_HEAD_KEY in expected and LM_HEAD_KEY not in state:
        state[LM_HEAD_KEY] = state[EMBED_KEY]

    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise KeyError(f"Checkpoint mismatch. Missing: {missing}. Unexpected: {unexpected}.")

    bad_shapes = [
        f"{k}: model {tuple(expected[k].shape)} vs checkpoint {tuple(state[k].shape)}"
        for k in expected
        if expected[k].shape != state[k].shape
    ]
    if bad_shapes:
        raise ValueError("Shape mismatch:\n" + "\n".join(bad_shapes))

    model.load_state_dict(state, strict=True)
