# mini-llm-inference-engine

A small LLM inference engine built from scratch in PyTorch and Triton, measured
against the roofline and against vLLM. **Work in progress.**

The goal is to understand where serving time and memory actually go: every
component is built by hand, and every design choice is benchmarked, prediction
first, then measurement, with the gap explained.

## Status

| Stage | What | State |
|---|---|---|
| 0 | Repo setup, config, weight loader | done |
| 1 | Qwen2 model from scratch, KV cache, first measurements | next |
| 2 | Continuous batching, paged KV cache, scheduler, load tests vs vLLM | planned |
| 3 | Prefix caching, FlashAttention reference | planned |
| 4 | Tensor parallelism (TP=2) | planned |
| 5 | Triton FlashAttention kernel | planned |

## Quick start

```bash
pip install -e ".[dev]"
python scripts/download_weights.py          # Qwen2.5-0.5B into ./weights
python scripts/inspect_weights.py weights/Qwen2.5-0.5B
pytest -m "not gpu"
```

## Docs

- [Overview](docs/OVERVIEW.md): goals, architecture, components, benchmarks
- [Plan](docs/PLAN.md): build order, step by step
- [Design](docs/DESIGN.md): decisions and tradeoffs
- [Results](docs/RESULTS.md): benchmark write-ups
- [Exercises](docs/EXERCISES.md): questions per step
