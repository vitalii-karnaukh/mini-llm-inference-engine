"""Download a model's weights, config and tokenizer into ./weights/<name>.

Usage:
    python scripts/download_weights.py                       # Qwen/Qwen2.5-0.5B
    python scripts/download_weights.py --model Qwen/Qwen2.5-0.5B --out weights
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download  # installed with transformers

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id")
    parser.add_argument("--out", default="weights", help="parent directory for downloads")
    args = parser.parse_args()

    target = Path(args.out) / args.model.split("/")[-1]
    path = snapshot_download(
        repo_id=args.model,
        local_dir=target,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
    )
    print(f"Downloaded {args.model} to {path}")


if __name__ == "__main__":
    main()
