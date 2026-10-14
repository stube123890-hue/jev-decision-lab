# jev-decision-lab

Which substrate makes the better **Jev-style decision model** — linear recurrence or attention?

Jev (TypeSafe AI, Sept 2026) is a non-autoregressive "System One" decision model: instead of generating
tokens, it reads a context and directly decides between options via a single cross-attention
decision head (options-as-queries). Jev's internals are undisclosed, so this repo defines the
experiment from scratch: two **complete** Jev-style models, trained end-to-end under identical
conditions, differing **only** in the backbone substrate.

## The experiment (JEV-V2)

| | jev-v8a | jev-transformer |
|---|---|---|
| Backbone | V8-A linear recurrence (~6.37M params) | Attention Transformer (~6.37M params) |
| Head | Jev-style: single cross-attention, options-as-queries | identical |
| Stage 0 | LM pre-train, 1500 steps, char-level | identical |
| Stage 1 | End-to-end decision finetune (backbone UNFROZEN) | identical |
| Stage 2 | Temperature scaling | identical |

Same optimizer, LR schedule, epochs, batch size, seeds, param budget.

**Decision tasks** (in-domain, char-level, multiple choice, 2000/500/500 train/val/test):
- **taskA** — next-sentence selection
- **taskB** — masked-word completion

**Metrics:** accuracy, ECE (raw → calibrated), Brier score, per-decision latency.

## Results (2026-10-13, free Colab T4)

```
config                 task      acc  ece_raw  ece_cal   brier
jev-transformer        taskA   46.2%    0.395    0.063   0.651
jev-transformer        taskB   59.8%    0.321    0.078   0.540
jev-v8a                taskA   50.4%    0.374    0.079   0.637
jev-v8a                taskB   63.4%    0.286    0.063   0.488

substrate effect (taskA):  jev-v8a 50.4% vs jev-transformer 46.2%
substrate effect (taskB):  jev-v8a 63.4% vs jev-transformer 59.8%
latency: {'jev-v8a': '14.8ms', 'jev-transformer': '7.1ms'}
```

**Reading:** the linear substrate wins on decision accuracy — both tasks, with better Brier scores.
End-to-end decision training especially helps the linear side (taskA: 36.0% with a frozen backbone
→ 50.4%). Temperature scaling fixes calibration for both (ECE 0.29–0.40 raw → 0.06–0.08).
Attention is ~2× faster per decision at this tiny scale (7.1 ms vs 14.8 ms).

## Files

- `linear_attention_jev_v2.ipynb` / `make_notebook_jev_v2.py` — V2 experiment + generator
- `linear_attention_jev_v3.ipynb` / `make_notebook_jev_v3.py` — V3 experiment + generator
- `linear_attention_jev_v4a_v8a.ipynb` / `linear_attention_jev_v4b_transformer.ipynb` / `make_notebook_jev_v4_split.py` — V4 split sessions + generator
- `results_jev_v4a_v8a.txt` / `results_jev_v4a_v8a.json`, `results_jev_v4b_transformer.txt` / `results_jev_v4b_transformer.json` — per-session test results

## JEV-V4: long-context retrieval decisions + the crossover (2026-10-14)

New task, new question. **taskC** — long-context retrieval-decision: 2k–8k character passages,
"which of these 4 sentences appears in the passage?", 2000/500/500 train/val/test. Same ~15.09M
budget (v8a 15,090,880 vs transformer 15,088,832 params), same Jev head, same protocol. Plus a
latency sweep at T=256…8192 (ms/decision, OOM as a finding) and a crossover line between the two
sweeps.

Logistics note: the combined run was killed by Colab twice at ~2.5–3.5h with no checkpoints, so
V4 ran as two sessions — V4a (jev-v8a) and V4b (jev-transformer) — with byte-identical cells,
seeds, and hyperparameters (generator: `make_notebook_jev_v4_split.py`).

```
=== JEV-V4 TEST RESULTS (taskC) ===
config                 task      acc  ece_raw  ece_cal   brier
jev-v8a                taskC   26.8%    0.221    0.049   0.755
jev-transformer        taskC   28.2%    0.079    0.024   0.751

backbone val loss: v8a 1.297 (ppl 3.7) | transformer 1.323 (ppl 3.8)
finetune val acc:  v8a 28.6/28.2/27.2/22.2/28.4/27.4 | transformer 26.8/23.6/23.0/24.8/26.8/27.8

=== JEV-V4 LATENCY SWEEP (ms/decision, eager fp32, T4) ===
T                  256    512   1024   2048   4096   8192
jev-v8a             14.1   16.4   22.6   35.1   60.7  108.6
jev-transformer     11.6   14.3   20.6   37.3   85.5  257.7

crossover: ~T=1556 tokens (log-log interpolation) — transformer faster below,
v8a faster above; at T=8192 v8a is 2.37x faster. No OOM at 8192 for either.
```

**Training speed (timestamp estimates from run logs):** jev-v8a's 6 taskC finetune epochs
completed in roughly half an hour; jev-transformer's took ~25–30 minutes *per epoch* — about a
6–7x per-epoch gap. Same data, same budget; the difference is the O(T²) attention tax on
2k–8k-char passages. (Estimates from session timestamps, not exact timers.)

**Reading:** taskC is brutal — both models sit near the 25% chance baseline (26.8% vs 28.2%
test), so the accuracy delta is noise, not a verdict. The verdict is structural: the latency
crossover lands at ~1.5k tokens, and beyond it the linear substrate pulls away — 2.37x faster
at 8k context. Combined with the ~6–7x training-speed gap, the efficiency case for linear
recurrence at long context is the finding here, independent of who wins accuracy.

## JEV-V3: 15M params + speed techniques (2026-10-14)

Same protocol scaled up: both models ~15.18M params (DIM 392 × 8 layers × 8 heads), totals matched
within ~2k. Plus a speed section on the v8a side: `torch.compile(mode="reduce-overhead")` + fp16
autocast inference, a 4-config latency matrix, and a fast-path accuracy check.

```
=== JEV-V3 TEST RESULTS ===
config                 task      acc  ece_raw  ece_cal   brier
jev-transformer        taskA   45.2%    0.398    0.095   0.667
jev-transformer        taskB   61.0%    0.297    0.081   0.512
jev-v8a                taskA   47.4%    0.393    0.074   0.647
jev-v8a                taskB   66.2%    0.283    0.059   0.463

substrate effect (taskA):  jev-v8a 47.4% vs jev-transformer 45.2%
substrate effect (taskB):  jev-v8a 66.2% vs jev-transformer 61.0%

=== JEV-V3 LATENCY MATRIX ===
jev-v8a/eager/fp32               17.2 ms/decision
jev-v8a/compiled/fp16            25.3 ms/decision
jev-transformer/eager/fp32       7.4 ms/decision
jev-transformer/compiled/fp16    15.1 ms/decision

=== JEV-V3 FAST-PATH ACCURACY ===
task A: eager fp32 47.4% | compiled fp16 47.8% | delta +0.40pp
task B: eager fp32 66.2% | compiled fp16 66.2% | delta +0.00pp
```

**Reading:** the accuracy win survives scale — v8a takes both tasks again at 15M. The gap narrowed
on taskA (+4.2pp → +2.2pp) but *widened* on taskB (+3.6pp → +5.2pp, with v8a's absolute accuracy up
63.4% → 66.2%). The linear decision-training edge did not wash out with capacity.

The speed story went the other way: `torch.compile` + fp16 made **both** models *slower*
(v8a 17.2→25.3 ms, transformer 7.4→15.1 ms). At batch=1 on tiny models, CUDA-graph overhead
dominates — compilation only pays off at larger batch/sizes. The fp16 path keeps full accuracy
(+0.4pp / +0.0pp), but it isn't a fast path here. The ~2.3× latency gap in attention's favor
remains at this scale; the structural O(T)-vs-O(T²) win for linear only materializes at longer
contexts, which these 200-char tasks don't exercise — that's the next study.

## Context

This follows JEV-V1 (in `linear-attention-lab/experiments/jev-v1/`), which compared Jev-style vs
linear decision *heads* on *frozen* backbones. V2 answers the corrected question: two full Jev-style
models, which substrate performs better.

## JEV-V5: does directionality help? (bidirectional × architecture, 2026-10-15)

V4 left a question open: causal-only models, ~25%-chance accuracy. V5 tests whether letting the
encoder see the full context (bidirectional) changes the picture — a 2×2 of directionality ×
architecture at the same 15M budget (DIM 352×8L×8H, byte-identical to V4; bidirectionality adds
zero params):

| | causal (V4 reference) | bidirectional (V5) |
|---|---|---|
| **v8a (linear)** | 26.8% test | **29.0% test** ✅ V5b |
| **transformer (attention)** | 28.2% test | pending — V5d next |

**V5b (jev-v8a-bidir, 15.09M)** — shared-weight forward+backward scans summed, same taskC, same
protocol as V4a. Results (`results_jev_v5b_v8a_bidir.txt`):

```
=== JEV-V5b TEST RESULTS (jev-v8a-bidir) ===
config                 task      acc  ece_raw  ece_cal   brier
jev-v8a-bidir          taskC   29.0%    0.062    0.025   0.746

=== JEV-V5b LATENCY SWEEP (jev-v8a-bidir) ===
 T (chars)   jev-v8a-bidir ms
       256               24.1
       512               26.1
      1024               35.7
      2048               56.0
      4096               97.5
      8192              178.2
```

**Reading:** directionality helps the linear model — 26.8% → 29.0% test (+2.2pp), the first
above-noise movement in this task series (val sat at chance 24–25% all 6 epochs; the test lift
came with calibration: ECE 0.062→0.025). The backward pass costs latency: bidir is ~1.6–1.7×
slower than causal v8a at every T (24.1 vs 14.1 ms at T=256; 178.2 vs 108.6 ms at T=8192), but
still O(T)-linear — the scaling shape is preserved, only the constant moved.

**Methodological note:** the bidir backbone's Stage-0 LM pre-train reached val 0.098 (ppl 1.1) vs
1.297 for the causal V4 backbone — expected, since a bidirectional mixer sees future tokens
during next-token pre-training and the task becomes near-trivial. Stage 0 is only
initialization; both V5 conditions share the advantage, so the directionality comparison stays
fair. Hypotheses H1–H4 and full protocol: `linear-attention-lab/experiments/jev-v5/SPEC.md`.
