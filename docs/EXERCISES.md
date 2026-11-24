# Exercises

Questions for each step in [PLAN.md](PLAN.md). Each step has the context you
need first, then the questions. Write your answers under each question.

---

## Step 1.0 — Where the parameters live

### Context

**Config** (`weights/Qwen2.5-0.5B/config.json`, or
[online](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json)):

| Field | Value | Meaning |
|---|---|---|
| `hidden_size` (D) | 896 | each token is a vector of 896 numbers |
| `num_attention_heads` (Hq) | 14 | number of Q heads |
| `num_key_value_heads` (Hk) | 2 | number of K/V heads (GQA) |
| head size (d) | 64 | D / Hq = 896 / 14 |
| `intermediate_size` | 4864 | MLP inner width |
| `num_hidden_layers` (L) | 24 | identical layers |
| `vocab_size` | 151936 | possible tokens |

**Shapes:** a linear layer turns `in` numbers into `out` numbers. PyTorch stores
its weight as `[out, in]` (math papers write `[in, out]`; same numbers,
transposed). Weight shapes never depend on batch size or sequence length.

**One decoder layer** (■ = one weight tensor):

```
■ input_layernorm.weight            [896]
■ q_proj.weight [896, 896]   ■ q_proj.bias [896]
■ k_proj.weight [128, 896]   ■ k_proj.bias [128]
■ v_proj.weight [128, 896]   ■ v_proj.bias [128]
■ o_proj.weight [896, 896]
■ post_attention_layernorm.weight   [896]
■ gate_proj.weight [4864, 896]
■ up_proj.weight   [4864, 896]
■ down_proj.weight [896, 4864]

Outside the 24 layers:
■ embed_tokens.weight [151936, 896]
■ norm.weight         [896]
  lm_head shares embed_tokens (tied weights)
```

### Questions

**Q1. Shapes.** What are the shapes of `v_proj.weight` and `o_proj.weight`, and why?
(What goes in, what comes out.)

Answer:

**Q2. Counting.** How many weight tensors does one layer have, and how many does
the whole model have? (Each tensor counts as 1, whatever its size.)

Answer:

**Q3. KV cache.** How many bytes of KV cache does one token need, at 2 bytes per
number? Why does the formula use K/V heads (2) and not query heads (14)?
Formula: 2 (K and V) × L × Hk × d × bytes per number.

Answer:

### Check

Run `python scripts/inspect_weights.py weights/Qwen2.5-0.5B --layer 0`:

- Q1 and Q2: compare with the printed shapes and tensor count.
- Q3: compare with the `kv bytes/token fp16` line at the top.

Then write 3–5 lines in [DESIGN.md](DESIGN.md) under `## 1.0 — Where the parameters live`.
