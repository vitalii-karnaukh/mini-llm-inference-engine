# Exercises

Questions for each step in [PLAN.md](PLAN.md). Each step has the context you
need first, then the questions. Write your answers under each question.

Questions are about concepts that apply to any model. Variables:

| Symbol | Meaning | Example (Qwen2.5-0.5B) |
|---|---|---|
| D | hidden size: numbers per token vector | 896 |
| Hq | query heads | 14 |
| Hkv | key/value heads (GQA when Hkv < Hq) | 2 |
| d | head size = D / Hq | 64 |
| L | number of layers | 24 |
| I | MLP inner width | 4864 |
| V | vocabulary size | 151936 |

---

## Step 1.0 — Where the parameters live

### Context

**Parameters vs activations.**
Parameters (weights) are learned during training, saved in the checkpoint, and
the same for every request. Activations are intermediate results computed for
each input and thrown away after use. One exception: K and V of past tokens are
kept in the KV cache.

**Shapes.** A linear layer turns `in` numbers into `out` numbers. PyTorch stores
its weight as `[out, in]` (math papers write `[in, out]`: same numbers, transposed).
Weight shapes never depend on batch size or sequence length.

**One decoder layer** (parameters only):

```
RMSNorm scale            [D]
Wq                       [Hq·d, D]
Wk                       [Hkv·d, D]
Wv                       [Hkv·d, D]
Wo                       [D, Hq·d]
RMSNorm scale            [D]
W_gate, W_up             [I, D]
W_down                   [D, I]

Outside the layers: embedding [V, D], final RMSNorm [D], LM head [V, D]
```

### Questions

**Q1. Shapes.** Why is Wk `[Hkv·d, D]` while Wo is `[D, Hq·d]`? Why does GQA
shrink Wk and Wv but not Wo?

Answer: Wv matches Wk: D in, Hkv·d out. Wo takes the attention output, which has
one result per query head (Hq·d), and maps it back to D. GQA shares K/V heads,
but each of the Hq queries still produces its own output, so Wo keeps Hq·d.

**Q2. Parameters vs activations.** Which of these are parameters and which are
activations: Wq, K for the current prompt, the RMSNorm scale, `x @ W_upᵀ`,
the attention scores? Which activation does an inference engine keep, and why?

Answer:

**Q3. KV cache.** Write the formula for KV cache bytes per token using the
variables above and b bytes per number. Why does it use Hkv and not Hq?

Answer: 2 (K and V) × Hkv × d × L × b. Example: 2 × 2 × 64 × 24 × 2 = 12,288
bytes ≈ 12.3 kB. Q is needed only for the current token's step; K and V of every
past token are read again by every future token, so only K and V are cached, and
their size follows Hkv.

### Check

Run `python scripts/inspect_weights.py weights/Qwen2.5-0.5B --layer 0` and
compare the shapes and the `kv bytes/token fp16` line with your answers. Then
write 3–5 lines in [DESIGN.md](DESIGN.md) under `## 1.0 — Where the parameters live`.
