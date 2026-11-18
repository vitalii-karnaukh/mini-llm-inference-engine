# Implementation plan

Ordered list of everything we build. Work top to bottom. Each step has an owner:

- **C** = Claude writes it (scaffold, boilerplate, glue, scripts)
- **U** = you write it (CORE, see CLAUDE.md)
- **C→U** = Claude scaffolds with tests, you write the core body

Every step is done only when its "Done when" check passes. Each U step follows
the micro-cycle: predict → read just enough → write → hint if stuck → explain back → log in DESIGN.md.

```
Stage 0  Setup                         ~1 h    (C)
Stage 1  Model + KV cache + numbers    ~10 h   checkpoint v0.1
Stage 2  Serving core                  ~15 h   checkpoint v0.2  ── gate: Nov 30
Stage 3  Prefix cache + FA reference   ~8 h    checkpoint v0.3
Stage 4  Tensor parallelism            ~8 h    checkpoint v0.4
Stage 5  Triton FlashAttention         ~12 h   checkpoint v0.5
Stage 6  Stretch                       only if ahead
```

---

## Stage 0 — Setup (Claude, one PR)

| # | Step | Owner | Done when |
|---|---|---|---|
| 0.1 | `pyproject.toml`, ruff, pytest config, `.gitignore`, GitHub Actions CI (CPU tests) | C | CI green on an empty test |
| 0.2 | `scripts/download_weights.py` (Qwen2.5-0.5B to a local cache) | C | Weights on disk |
| 0.3 | `scripts/inspect_weights.py`: prints config + every weight name and shape | C | Runs, prints ~290 tensors |
| 0.4 | `mini_engine/model/config.py` (dataclass from `config.json`), `loader.py` (safetensors → our module names) | C | Loader test passes on a fake checkpoint |
| 0.5 | `tests/hf_reference.py`: loads the HF model, captures per-layer outputs with hooks | C | Helper used by parity tests |

---

## Stage 1 — Model, KV cache, first numbers (~Nov 18–24)

| # | Step | Owner | Concept you own after | Done when |
|---|---|---|---|---|
| 1.0 | Run `inspect_weights.py`, read every shape, write 5 lines in DESIGN.md on where the 0.5B params live | U | Parameter count, GQA in the weights | You can predict any weight's shape |
| 1.1 | `RMSNorm.forward` | C→U | Normalization | Matches HF RMSNorm |
| 1.2 | RoPE: cos/sin cache builder (C), `apply_rotary_emb` (U) | C→U | Position encoding, dim pairing | Matches HF rotary output |
| 1.3 | `Attention.forward`, prefill only (no cache): q/k/v proj with bias, RoPE, GQA repeat, causal mask, softmax, o_proj | C→U | Q/K/V shapes, GQA, mask | Matches HF layer-0 attention output |
| 1.4 | `MLP.forward` (SwiGLU) | C→U | Gated MLP | Matches HF MLP |
| 1.5 | `DecoderLayer.forward` (residual wiring) | C→U | Pre-norm residual stream | Matches HF layer-0 output |
| 1.6 | `Qwen2Model.forward` + **full logit parity** | C→U | Whole forward pass | Logits match HF on 20 prompts (fp32 tolerance) |
| 1.7 | KV cache, simple version: contiguous per-request cache, write new K/V, read history, **mask offset** for decode | C→U | Prefill vs decode, why cache K/V not Q | Prefill+decode logits == full recompute logits |
| 1.8 | Sampler (greedy, temperature, top-p), stop conditions, `generate()` loop | C | — | Greedy text == HF greedy text |
| 1.9 | Static batching with padding: padded batch + **padding mask** in attention | C→U | Padding cost, mask correctness | Batched outputs == one-by-one outputs |
| 1.10 | Timing harness + laptop measurements: prefill time vs prompt length, decode time per token, KV bytes per token (formula vs `tensor.nbytes`) | C (harness), U (predictions) | Measuring before claiming | Numbers in RESULTS.md with predictions |
| 1.11 | **GPU session 1** (g5.xlarge, ~2 h): setup script (C), ceilings: matmul FLOP/s + copy GB/s (C), then B1, B2, B3 (`bench` mode) | C + U | Roofline with real numbers | B1–B3 in RESULTS.md |
| 1.12 | README v0.1: what it is, B1–B3 charts, limitations | C draft, U edits | — | **Checkpoint v0.1** + explain-back of stage 1 |

---

## Stage 2 — Serving core (~Nov 23–29)

| # | Step | Owner | Concept you own after | Done when |
|---|---|---|---|---|
| 2.1 | `Request`, `RequestStatus`, `SamplingParams`, request queues | C | — | Unit tests pass |
| 2.2 | `BlockPool` + `KVCacheManager`: `allocate`, `free`, `append_slot`, `can_allocate` | C→U | Paging, free list, fragmentation | Block tests incl. out-of-memory, free-then-reuse |
| 2.3 | Paged KV storage `[num_blocks, block_size, n_kv, head_dim]` per layer (C); attention **writes via slot mapping, reads via block table** (U) | C→U | Slot mapping, gather vs paged kernel | Paged logits == stage-1 logits |
| 2.4 | `ModelRunner.prepare_inputs`: flat tokens, positions, slot_mapping, block tables for a mixed prefill+decode batch | C→U | How the scheduler's decision becomes tensors | Mixed batch == running each request alone |
| 2.5 | `Scheduler.schedule`: token budget, running first, FCFS admission, chunked prefill; `preempt` (recompute) | C→U | Continuous batching, TTFT vs TPOT, preemption | Scheduler tests incl. budget overflow and OOM preemption |
| 2.6 | `LLMEngine.step` loop, `add_request`, outputs, stop handling | C | — | 50 concurrent requests finish, text == one-by-one |
| 2.7 | Load generator (Poisson arrivals, length distributions) + metrics. **You approve metric definitions** before use | C, U approves | Latency definitions | Load test runs on laptop |
| 2.8 | Design decision: padding vs packing for mixed batches. Measure padding waste, decide, write it in DESIGN.md | U | Varlen attention tradeoff | Decision + number in DESIGN.md |
| 2.9 | **GPU session 2**: flash-attn decode backend with block table (C glue, U reviews), B5 load test, B1+B5 on vLLM (C scripts), Nsight Systems trace of one decode step | C + U | Why vLLM is faster | Gap to vLLM explained from the trace |
| 2.10 | README v0.2 + RESULTS | C draft, U edits | — | **Checkpoint v0.2** + explain-back |

**Gate, Nov 30:** stage 2 not done → stop adding features, polish v0.2.

---

## Stage 3 — Cheap high-ROI add-ons (~Nov 30 – Dec 3)

| # | Step | Owner | Done when |
|---|---|---|---|
| 3.1 | Prefix caching: chained block hashes, lookup, ref counts, LRU reuse of ref-0 blocks | C→U | Shared-prefix test: blocks reused, outputs unchanged; B6 |
| 3.2 | `flash_ref.py`: tiled attention with online softmax | C→U | Matches naive attention at several T and tile sizes |
| 3.3 | README capacity section: Llama-70B on 8×H100 vs 8×A10G (memory, concurrency, TPOT lower bound) | U writes, C checks math | Every number shows its formula |
| 3.4 | Optional: CUDA graphs for decode, before/after | C implements, U explains why it helps | Before/after in RESULTS.md |
| 3.5 | **Checkpoint v0.3** + explain-back | U | — |

---

## Stage 4 — Tensor parallelism (~Dec 4–8, g5.12xlarge)

| # | Step | Owner | Done when |
|---|---|---|---|
| 4.1 | `parallel_state.py`, torchrun launch script, NCCL init | C | 2 processes all-reduce a test tensor |
| 4.2 | `ColumnParallelLinear`, `RowParallelLinear`: weight sharding + forward + all-reduce placement | C→U | Sharded linear == full linear |
| 4.3 | Wire TP into attention and MLP; loader shards weights per rank (C), head split logic (U) | C→U | TP=2 logits == TP=1 logits |
| 4.4 | B7 inside one box + Nsight Systems NCCL time per step | C + U | B7 in RESULTS.md |
| 4.5 | **Checkpoint v0.4** + explain-back | U | — |

---

## Stage 5 — Triton FlashAttention (~Dec 9–13)

| # | Step | Owner | Done when |
|---|---|---|---|
| 5.1 | Triton forward kernel, causal, fp16 (C: launcher, grid, test harness; U: kernel body) | C→U | Matches `flash_ref.py` |
| 5.2 | Plug in as prefill backend | C | Engine outputs unchanged |
| 5.3 | B4: naive vs yours vs official flash-attn by sequence length; optional Nsight Compute | C + U | B4 in RESULTS.md |
| 5.4 | **Checkpoint v0.5** + explain-back | U | — |

---

## Stage 6 — Stretch (only if ahead)

- TP=2 across two g5.xlarge (B7 cross-node)
- 2-stage pipeline-parallel demo

---

## Running alongside the build (in the project chat)

| Habit | When |
|---|---|
| Predict questions | Start of every U step |
| Explain-back + interview follow-up | End of every U step and every stage |
| "Drill me: sizing" (15 min) | 3× a week |
| Mock system design (45 min) | 1× a week from stage 2, 2× a week in the last 2 weeks |
| Concept-only reading: quantization, speculative decoding, P/D disaggregation, MoE serving | Last 2 weeks |

## Session rule

End every session with passing tests and a pushed commit. The next session starts at the next step in this file.
