"""Print a model's config and every weight with its shape, plus parameter totals.

Usage:
    python scripts/inspect_weights.py weights/Qwen2.5-0.5B
    python scripts/inspect_weights.py weights/Qwen2.5-0.5B --layer 0   # only layer 0 + globals

Step 1.0 of docs/PLAN.md: read the output and explain where the parameters live.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

from safetensors import safe_open

from mini_engine.model.config import ModelConfig
from mini_engine.model.loader import checkpoint_files

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def group_of(key: str) -> str:
    """Bucket a weight name for the totals table."""
    if "embed_tokens" in key:
        return "embedding"
    if key.startswith("lm_head"):
        return "lm_head"
    if "self_attn" in key:
        return "attention"
    if "mlp" in key:
        return "mlp"
    if "norm" in key:
        return "norms"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--layer", type=int, default=None, help="show only this layer's weights")
    args = parser.parse_args()

    cfg = ModelConfig.from_model_dir(args.model_dir)
    print("== Config ==")
    for name, value in vars(cfg).items():
        print(f"  {name:28s} {value}")
    print(f"  {'head_dim (derived)':28s} {cfg.head_dim}")
    print(f"  {'queries per kv (derived)':28s} {cfg.num_queries_per_kv}")
    print(f"  {'kv bytes/token fp16':28s} {cfg.kv_bytes_per_token(2):,}")

    totals: dict[str, int] = defaultdict(int)
    dtypes: set[str] = set()
    rows: list[tuple[str, tuple[int, ...], str]] = []
    for path in checkpoint_files(args.model_dir):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                shape = tuple(sl.get_shape())
                dtype = str(sl.get_dtype())
                numel = 1
                for dim in shape:
                    numel *= dim
                totals[group_of(key)] += numel
                dtypes.add(dtype)
                rows.append((key, shape, dtype))

    def layer_index(key: str) -> int:
        m = LAYER_RE.search(key)
        return int(m.group(1)) if m else -1

    print(f"\n== Weights ({len(rows)} tensors, dtypes: {sorted(dtypes)}) ==")
    for key, shape, dtype in sorted(rows, key=lambda r: (layer_index(r[0]), r[0])):
        idx = layer_index(key)
        if args.layer is not None and idx not in (-1, args.layer):
            continue
        print(f"  {key:60s} {str(shape):22s} {dtype}")

    total = sum(totals.values())
    print("\n== Parameters by group ==")
    for group, count in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {group:12s} {count:>14,}  ({100 * count / total:5.1f}%)")
    print(f"  {'total':12s} {total:>14,}  (~{total * 2 / 1e9:.2f} GB in fp16/bf16)")
    if cfg.tie_word_embeddings and "lm_head" not in totals:
        print("  note: tie_word_embeddings=True, so lm_head reuses the embedding matrix")


if __name__ == "__main__":
    main()
