"""Shared fixtures.

Tests marked `weights` need real model weights. They look in, in order:
  1. $MINI_ENGINE_MODEL_DIR
  2. ./weights/Qwen2.5-0.5B   (where scripts/download_weights.py puts them)
and are skipped if neither has a config.json.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "weights" / "Qwen2.5-0.5B"


@pytest.fixture(scope="session")
def model_dir() -> Path:
    candidates = [os.environ.get("MINI_ENGINE_MODEL_DIR"), DEFAULT_MODEL_DIR]
    for c in candidates:
        if c and (Path(c) / "config.json").exists():
            return Path(c)
    pytest.skip("Model weights not found. Run: python scripts/download_weights.py")
