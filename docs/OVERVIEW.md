# Project overview

Build a small but correct LLM inference engine from scratch, then measure it
against theory and against vLLM. The repo's value is the measured results, not
the feature count. The step-by-step build order lives in [PLAN.md](PLAN.md).

## Goal and positioning

**What it is:** a Python/PyTorch engine that loads real open weights
(Qwen2.5-0.5B, later Llama-3.2-1B), serves many concurrent requests with
continuous batching and a paged KV cache, runs prefill through a hand-written
Triton FlashAttention kernel, and splits the model across 2 GPUs with tensor
parallelism.

**What it proves:**

- The transformer forward pass is understood down to tensor shapes (logit parity
  with Hugging Face proves it).
- Where inference time and memory go is understood, because it was predicted
  first and then measured.
- Tradeoffs are made and defended: padding vs packing, TP inside vs across
  machines, block size, batch size vs latency.

**What it is not:** a vLLM competitor. Small engines like nano-vllm already
exist. The differentiator is the benchmark write-up: predicted vs measured, with
honest explanations of the gaps.

## Architecture

```
 Poisson load generator ──► Request queue (waiting · running · finished)
                                 │
                                 ▼
                     ┌──────── Scheduler ────────┐        ┌── KVCacheManager ──┐
                     │ token budget per step     │ ◄────► │ BlockPool, 16-tok  │
                     │ mixes prefill and decode  │ alloc/ │ blocks, block      │
                     │ preempts when blocks run  │ free   │ tables, ref counts,│
                     │ out                       │        │ prefix hashes      │
                     └────────────┬──────────────┘        └─────────┬──────────┘
                                  ▼ batch                           │ block tables
                     ┌──────── ModelRunner ──────┐ ◄────────────────┘
                     │ positions, slot_mapping,  │
                     │ block tables → tensors    │
                     └────────────┬──────────────┘
                                  ▼
                 Model (Qwen2-style, TP=2: GPU0 shard | GPU1 shard)
                 prefill attention: own Triton FlashAttention
                 decode attention:  flash-attn with block tables
                                  │
                                  ▼
                 Metrics + bench harness: TTFT, TPOT, tokens/s, KV usage,
                 same workload against vLLM
```

The scheduler is the centre. Each step it pulls requests from the queue, asks
the KV cache manager for blocks, and sends one mixed prefill/decode batch to the
model. Every step's timings feed the metrics harness.

## Components and scope

| # | Component | What gets built | Done when | Priority |
|---|---|---|---|---|
| 1 | Model | Qwen2-style decoder by hand: embeddings, RMSNorm, RoPE, GQA attention, SwiGLU MLP, LM head; loads safetensors | Logits match Hugging Face on 20 prompts | Must |
| 2 | KV cache + generation | Prefill then decode loop; greedy, temperature, top-p; stop tokens | Same text as HF greedy; decode step time measured | Must |
| 3 | Scheduler | Continuous batching, mixed prefill/decode, preemption when KV memory runs out | Poisson load test runs with no deadlocks; TTFT/TPOT per request | Must |
| 4 | Paged KV cache | Fixed-size blocks, free list, per-sequence block table, ref counts | Memory use matches the formula; no fragmentation under mixed lengths | Must |
| 5 | Prefix caching | Hash full prompt blocks; reuse across requests with a shared prefix | TTFT drops on a shared-prefix workload; hit rate reported | Should |
| 6 | FlashAttention | (a) tiled PyTorch reference with online softmax; (b) Triton forward kernel as the prefill backend | (a) matches naive; (b) matches and beats naive at long sequences | (a) Must, (b) Should |
| 7 | Tensor parallelism | TP=2 with torch.distributed + NCCL: column-split Q/K/V and gate/up, row-split o_proj and down, all-reduce after each | Same logits as TP=1; step time inside one box and across two | Should |
| 8 | Metrics + bench harness | Load generator, TTFT/TPOT/throughput percentiles, GPU memory, same harness against vLLM | One command reproduces every README chart | Must |

**Out of scope (concept-level only):** pipeline parallelism beyond an optional
2-stage demo, sequence/context parallelism, quantization, speculative decoding,
prefill/decode disaggregation, paged-attention decode kernels, an HTTP API.

**Design decisions to make and document in DESIGN.md:**

- Batching variable lengths: start with padding, measure the waste, then pack (varlen) if time allows.
- Paged KV without a paged kernel: on CPU, gather blocks into a contiguous tensor before attention. On GPU, use `flash_attn_with_kvcache` with a `block_table` for decode.
- Block size: 16 vs 32 tokens, fragmentation vs overhead.

## Topic coverage

| Interview topic | Covered by | Depth | Question it prepares for |
|---|---|---|---|
| Forward pass, Q/K/V shapes, GQA, RoPE | Model (1) | Built | "Walk me through attention with shapes" |
| KV cache, prefill vs decode | KV cache (2) | Built + measured | "Why is decode memory-bound?" |
| Memory math (weights, KV per token, concurrency) | Paged KV (4), bench (8) | Built + measured | "How many concurrent users fit on this GPU?" |
| Continuous batching, scheduling, preemption | Scheduler (3) | Built | "Design request scheduling for a 70B service" |
| PagedAttention, prefix caching | Paged KV (4), prefix cache (5) | Built | "Why does vLLM use blocks?" |
| FlashAttention, tiling, online softmax | FlashAttention (6) | Built | "Why is FlashAttention faster if FLOPs are the same?" |
| Roofline, TTFT/TPOT formulas | Benchmarks (8) | Measured vs predicted | "Estimate TPOT for 70B on 8 GPUs" |
| Tensor parallelism, communication cost | TP=2 (7) | Built + measured | "Why keep TP inside one node?" |
| Pipeline parallelism | Optional demo | Partial | "When would you use PP over TP?" |
| Quantization, speculative decoding, P/D disaggregation | Not built | Concept | Design-level follow-ups |

## Hardware and budget

Develop on a laptop; rent GPUs only for benchmark sessions. Expected spend about
$50–120 total, on a personal AWS account.

| Stage | Machine | GPUs | Use | Approx. on-demand cost |
|---|---|---|---|---|
| 1–3 development | Laptop (CPU or Apple MPS) | none | Model, cache, scheduler, paged KV, FA reference with a 0.5B model | $0 |
| 1–3 benchmarks, 5 kernel | g5.xlarge | 1× A10G, 24 GB | Short benchmark sessions, vLLM comparison, Triton kernel | ~$1/hr |
| 4 TP inside one box | g5.12xlarge | 4× A10G over PCIe | TP=2 intra-node | ~$5.7/hr |
| 6 TP across boxes (stretch) | 2× g5.xlarge | 1 each, up to 10 Gbps | Show why TP should not cross machines | ~$2/hr |

Prices are approximate us-east-1 on-demand figures; check before renting.

**Measure your own ceilings first:** a large fp16 matmul (achievable FLOP/s) and a
large tensor copy (achievable GB/s) on the A10G. Use those, not datasheet
numbers, in every roofline claim.

**Cost hygiene:** one setup script, weights cached on EBS or S3, stop the instance
after each session.

## Benchmarks

Each experiment is written as: prediction with its formula, measured number, why
they differ.

| ID | Experiment | Prediction to write first | Chart |
|---|---|---|---|
| B1 | Decode step time vs batch size (1→256) | Flat while memory-bound, rises once compute-bound | ms/step and tokens/s vs batch |
| B2 | Prefill time vs prompt length (128→8k) | Roughly linear, then attention's quadratic term shows | ms vs tokens |
| B3 | Achieved decode bandwidth vs measured ceiling | Weight bytes / step time; well under 100% | % of roofline |
| B4 | Naive vs own Triton FA vs official flash-attn | Naive slows or runs out of memory at long T | ms vs seq len |
| B5 | Load test: TTFT/TPOT p50/p95 vs request rate | Flat, then a sharp rise near capacity | latency vs req/s |
| B6 | Prefix caching on/off, shared 1k-token system prompt | TTFT drops by the share of prefill skipped | TTFT on vs off |
| B7 | TP=1 vs TP=2 in one box vs across two boxes | Intra-node modest gain; cross-node slower than TP=1 | ms/token |

**vLLM comparison:** run B1 and B5 against vLLM on the same GPU and model. Losing
is expected; not knowing why is not acceptable. Name the reasons (CUDA graphs,
fused kernels, paged decode attention, Python overhead) and which matters most.

**Profiling:** every gap between predicted and measured is explained from a trace.

| Tool | Answers | Used for |
|---|---|---|
| `torch.profiler` + Perfetto | Which ops take the time, CPU vs GPU | First look at every benchmark |
| Nsight Systems | GPU idle gaps, launch overhead, NCCL time | Gap to vLLM (B1, B5), TP cost (B7) |
| Nsight Compute | Bandwidth, FLOPs, occupancy of one kernel | Triton kernel (B4), optional |

Rule: profile, explain, fix at most one thing, measure again.

## Repo structure

```text
mini-llm-inference-engine/
  README.md
  CLAUDE.md                # collaboration rules for AI assistants
  docs/OVERVIEW.md         # this file
  docs/PLAN.md             # ordered build steps
  docs/DESIGN.md           # design decisions, in my words
  docs/RESULTS.md          # B1–B7 write-ups
  mini_engine/
    engine/                # LLMEngine, Request, SamplingParams
    core/                  # Scheduler, KVCacheManager + BlockPool
    worker/                # ModelRunner
    model/                 # config, loader, layers, qwen2
    attention/             # naive, flash_ref, flash_triton
    distributed/           # parallel_state, ColumnParallelLinear, RowParallelLinear
    metrics.py
  bench/                   # load generator, runners, plots, vllm_compare
  tests/
  scripts/
```

Class names follow vLLM V1 conventions so the code maps onto vLLM when reading it later.

**Quality bar:**

- Tests run in CI on CPU; GPU tests run locally.
- Every benchmark reproducible with one command; versions and GPU recorded.
- Type hints, docstrings on public functions, no notebooks as the main code.
- Small commits with clear messages.
- An honest "Limitations" section in the README.
- No inflated claims: "3.1× slower than vLLM, here's why" beats "production-ready".

## Risks and cut order

The main risk is scope next to a full-time job. If a stage slips, cut from the
top of this list first. Stages 1–2 and their benchmarks are never cut.

1. Stage 6 stretch (cross-machine TP, pipeline demo)
2. Nsight Compute on the kernel
3. Stage 5 Triton kernel (the PyTorch reference stays)
4. CUDA graphs before/after
5. Stage 4 TP=2 (keep the math in the README)

| Risk | Mitigation |
|---|---|
| Logit parity fails and eats days | Compare layer by layer with HF hooks; check RoPE layout and q/k/v bias first |
| Triton kernel debugging stalls | Time-box to 4 days; fall back to the reference |
| GPU costs creep | Stop instances after each session; use spot |
| Building without understanding | After each stage, explain it out loud in 2 minutes without notes |

## Reference material

Write your version first, then read the reference. Stuck for 2 hours on one
problem: read how nano-vllm or vLLM handles that one piece, close it, write your
own. Nothing from these projects is imported as a library.

| Stage | Read | Why |
|---|---|---|
| 1 | [HF modeling_qwen2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py) | The parity reference: RoPE layout, GQA head repeat |
| 1 | [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) | KV and FLOP math, memory-bound decode |
| 1–3 | [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html) | Compute vs memory vs overhead |
| 2 | [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) | Block manager, why paging |
| 2 | [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | Compare scheduler and block manager after yours works |
| 2–3 | [gpt-fast](https://github.com/pytorch-labs/gpt-fast) | CUDA graphs, TP in a few hundred lines |
| 3, 5 | [FlashAttention](https://arxiv.org/abs/2205.14135), [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Tiling, online softmax |
| 5 | [Triton fused attention tutorial](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html) | Compare after writing your kernel |
| 4 | [Megatron-LM](https://arxiv.org/abs/1909.08053), section 3 | Column/row split, all-reduce placement |
