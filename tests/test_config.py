import json

import pytest

from mini_engine.model.config import ModelConfig

# Values as written in a Qwen2-style config.json (used only to test parsing).
SAMPLE = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "intermediate_size": 4864,
    "num_hidden_layers": 24,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "some_unknown_key": 123,
}


def test_parses_known_fields_and_ignores_unknown():
    cfg = ModelConfig.from_dict(SAMPLE)
    assert cfg.hidden_size == 896
    assert cfg.num_key_value_heads == 2
    assert cfg.tie_word_embeddings is True
    assert not hasattr(cfg, "some_unknown_key")


def test_derived_values():
    cfg = ModelConfig.from_dict(SAMPLE)
    assert cfg.head_dim == 64
    assert cfg.num_queries_per_kv == 7
    # 2 (K,V) x 24 layers x 2 kv heads x 64 dims x 2 bytes
    assert cfg.kv_bytes_per_token(2) == 12_288


def test_qwen2_defaults_to_qkv_bias_llama_does_not():
    assert ModelConfig.from_dict(SAMPLE).attention_bias is True
    llama = {**SAMPLE, "model_type": "llama"}
    assert ModelConfig.from_dict(llama).attention_bias is False


def test_from_model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(SAMPLE))
    assert ModelConfig.from_model_dir(tmp_path).vocab_size == 151936


@pytest.mark.weights
def test_real_config_loads(model_dir):
    cfg = ModelConfig.from_model_dir(model_dir)
    assert cfg.hidden_size % cfg.num_attention_heads == 0
    assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
