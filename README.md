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

- `linear_attention_jev_v2.ipynb` — the experiment notebook (runs on free Colab T4)
- `make_notebook_jev_v2.py` — generator that produced the notebook
- `results_jev_v2.txt` / `results_jev_v2.json` — the test results

## Context

This follows JEV-V1 (in `linear-attention-lab/experiments/jev-v1/`), which compared Jev-style vs
linear decision *heads* on *frozen* backbones. V2 answers the corrected question: two full Jev-style
models, which substrate performs better.
