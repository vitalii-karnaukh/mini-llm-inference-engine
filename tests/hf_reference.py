"""Hugging Face reference model for parity tests.

Loads the HF model in fp32 on CPU and captures intermediate outputs with forward
hooks, so each of our layers can be compared against the matching HF layer.

Example:
    ref = HFReference(model_dir)
    out = ref.run(input_ids, capture=["model.layers.0.self_attn", "model.norm"])
    out.logits                                  # [batch, seq, vocab]
    out.captured["model.layers.0.self_attn"]    # first output of that module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import torch


@dataclass
class ReferenceOutput:
    logits: torch.Tensor  # [batch, seq, vocab]
    captured: dict[str, torch.Tensor] = field(default_factory=dict)


class HFReference:
    def __init__(self, model_dir: str | Path, dtype: torch.dtype = torch.float32):
        self.model_dir = Path(model_dir)
        self.dtype = dtype

    @cached_property
    def model(self) -> torch.nn.Module:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(self.model_dir, torch_dtype=self.dtype)
        model.eval()
        return model

    @cached_property
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.model_dir)

    def encode(self, text: str) -> torch.Tensor:
        """Token ids for one prompt, shape [1, seq]."""
        return self.tokenizer(text, return_tensors="pt").input_ids

    @torch.no_grad()
    def run(self, input_ids: torch.Tensor, capture: list[str] | None = None) -> ReferenceOutput:
        """Forward pass; `capture` lists module names whose outputs to keep."""
        captured: dict[str, torch.Tensor] = {}
        handles = []
        modules = dict(self.model.named_modules())
        for name in capture or []:
            if name not in modules:
                raise KeyError(f"No module named {name!r} in the HF model")

            def hook(_mod, _inp, out, name=name):
                captured[name] = (out[0] if isinstance(out, tuple) else out).detach().clone()

            handles.append(modules[name].register_forward_hook(hook))
        try:
            logits = self.model(input_ids=input_ids, use_cache=False).logits
        finally:
            for h in handles:
                h.remove()
        return ReferenceOutput(logits=logits, captured=captured)

    @torch.no_grad()
    def greedy_text(self, prompt: str, max_new_tokens: int = 20) -> str:
        """HF greedy generation, for comparing against our generate()."""
        ids = self.encode(prompt)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
