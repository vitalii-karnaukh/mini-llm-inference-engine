import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from mini_engine.model.loader import load_state_dict, load_weights


class TinyLM(nn.Module):
    """Fake model with HF-style names, enough to exercise the loader."""

    def __init__(self, vocab: int = 10, dim: int = 4):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, dim)
        self.model.norm = nn.LayerNorm(dim, elementwise_affine=True, bias=False)
        self.lm_head = nn.Linear(dim, vocab, bias=False)


def write_checkpoint(path, tensors):
    path.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path / "model.safetensors"))


def test_loads_matching_checkpoint(tmp_path):
    src = TinyLM()
    write_checkpoint(tmp_path, src.state_dict())
    dst = TinyLM()
    load_weights(dst, tmp_path)
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


def test_tied_embeddings_fill_missing_lm_head(tmp_path):
    src = TinyLM()
    tensors = {k: v for k, v in src.state_dict().items() if k != "lm_head.weight"}
    write_checkpoint(tmp_path, tensors)
    dst = TinyLM()
    load_weights(dst, tmp_path, tie_word_embeddings=True)
    assert torch.equal(dst.lm_head.weight, src.model.embed_tokens.weight)


def test_missing_key_raises_without_tying(tmp_path):
    tensors = {k: v for k, v in TinyLM().state_dict().items() if k != "lm_head.weight"}
    write_checkpoint(tmp_path, tensors)
    with pytest.raises(KeyError, match="lm_head.weight"):
        load_weights(TinyLM(), tmp_path)


def test_shape_mismatch_raises(tmp_path):
    write_checkpoint(tmp_path, TinyLM(vocab=12).state_dict())
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_weights(TinyLM(vocab=10), tmp_path)


def test_rename_and_skip(tmp_path):
    write_checkpoint(tmp_path, {"a.weight": torch.ones(2), "drop.me": torch.zeros(1)})
    state = load_state_dict(tmp_path, rename=lambda k: None if k.startswith("drop") else "b." + k)
    assert set(state) == {"b.a.weight"}


def test_dtype_cast_only_floats(tmp_path):
    write_checkpoint(tmp_path, {"f": torch.ones(2), "i": torch.ones(2, dtype=torch.int64)})
    state = load_state_dict(tmp_path, dtype=torch.bfloat16)
    assert state["f"].dtype == torch.bfloat16
    assert state["i"].dtype == torch.int64


def test_no_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_state_dict(tmp_path)
