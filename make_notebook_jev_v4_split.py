"""Generate SPLIT Colab notebooks JEV-V4a (v8a) and JEV-V4b (transformer).

Why split: V4's combined ~3.5h run doesn't fit Colab's runtime limit — the
runtime was reclaimed twice mid-finetune (first at ~3.5h, second at ~2.5h),
and Drive checkpointing is broken in this environment, so each model now
trains in its own short session. V4a (v8a) ~= 1.5h; V4b (transformer) ~= 2.5h.

Comparability contract (must hold for the two sessions to be directly
comparable to each other and to the V4 spec):
  - taskC data generation: byte-identical (random.Random(0) only).
  - Explicit seeds byte-identical: torch.manual_seed(0) at setup,
    torch.manual_seed(1234) before head init, torch.Generator().manual_seed(0)
    for batch order.
  - Hyperparams byte-identical: DIM 352, 8 layers, 8 heads, CTX_LONG 8192,
    ffn_hidden=1343 (v8a), state_dim=256, forget floors, LR 3e-4, micro 2,
    accum 4, 6 epochs, 1500 stage-0 steps, EVAL_B 8.
  - Known unavoidable difference: only one backbone is constructed per
    session (V4 built both), so implicit global-RNG draws for weight init
    and stage-0 batches differ. No dropout anywhere (0.0), so nothing else
    consumes global RNG. This washes out over 1500 pretrain steps; stage 1
    (the actual experiment) is controlled by the explicit seeds above.

Each notebook prints the same markers the monitor watches for:
BACKBONETRAININGDONE, FINETUNEDONE, CALIBRATIONDONE, LATENCYDONE,
TEST RESULTS table, LATENCY SWEEP table, crossover line, SAVE DONE (no space).
Results files (txt+json) carry the tables so the parent can combine the two
sessions' sweeps into V4's crossover table locally.
"""
import json
import os

OUTDIR = "/home/hatch/workspace/linear-attention-lab/experiments/jev-v4"
os.makedirs(OUTDIR, exist_ok=True)


def new_nb():
    return {"nbformat": 4, "nbformat_minor": 0,
            "metadata": {"kernelspec": {"display_name": "Python 3",
                                        "language": "python",
                                        "name": "python3"}},
            "cells": []}


def md(NB, src):
    NB["cells"].append({"cell_type": "markdown", "metadata": {},
                        "source": src.splitlines(keepends=True)})


def code(NB, src):
    NB["cells"].append({"cell_type": "code", "metadata": {},
                        "source": src.splitlines(keepends=True),
                        "outputs": [], "execution_count": None})


# ---------------------------------------------------------------- setup ---
# Byte-identical to V4 (cell text verified by diff against the V4 notebook).
SETUP_SRC = """import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "JEV-V4 needs the T4 GPU"
torch.manual_seed(0)
import math, random, re, time, json
random.seed(0)
"""

# ------------------------------------------------------------------ data ---
# Byte-identical to V4 (cell text verified by diff against the V4 notebook).
DATA_SRC = """import os, urllib.request
os.makedirs("data", exist_ok=True)
URLS = {
    "tinyshakespeare.txt": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "alice.txt":           "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "frankenstein.txt":    "https://www.gutenberg.org/cache/epub/84/pg84.txt",
    "pride.txt":           "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
}
parts = []
for name, url in URLS.items():
    p = f"data/{name}"
    if not os.path.exists(p):
        urllib.request.urlretrieve(url, p)
    with open(p, encoding="utf-8", errors="ignore") as f:
        parts.append(f.read())
text = "\\n".join(parts)
chars = sorted(set(text)); vocab = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.95 * len(data)); train_data, val_data = data[:n], data[n:]
nl_id = stoi.get("\\n")
print(f"corpus {len(text)/1e6:.2f} MB, vocab {vocab}, newline_id={nl_id}")
assert nl_id is not None, "newline not in vocab!"

SEQ, BS = 128, 32   # stage-0 pre-training stays on SHORT windows (as V3)
def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    i = torch.randint(0, len(d) - SEQ - 1, (BS,))
    x = torch.stack([d[j:j+SEQ] for j in i]).to(device)
    y = torch.stack([d[j+1:j+SEQ+1] for j in i]).to(device)
    return x, y
"""


def models_cell_src():
    # Same inline recipe as the V4 generator (verified by diff).
    _models_src = open(
        "/home/hatch/workspace/linear-attention-lab/models.py").read().replace(
        '"""Linear-recurrent replacement for Transformer self-attention.',
        '"""(inlined) Linear-recurrent replacement for Transformer self-attention.').replace(
        "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
        "import torch.nn as nn\nimport torch.nn.functional as F\n")
    _kern_src = open(
        "/home/hatch/workspace/linear-attention-lab/triton_kernels.py").read().replace(
        '"""V6: Triton-fused selective scan for the gated segmented state space.',
        '"""(inlined) V6/V7/V8-A: Triton-fused scans.').replace(
        "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
        "import triton\nimport triton.language as tl\n")
    return _models_src + "\n\n" + _kern_src


# ----------------------------------------------------------------- taskC ---
# Byte-identical to V4 (cell text verified by diff against the V4 notebook).
TASKC_SRC = """# taskC: "which of these 4 sentences appears verbatim in the passage?"
# Passages concatenate corpus sentences to a uniform-random 2000-8000 chars.
# 1 TRUE option = a sentence drawn verbatim from THIS passage; 3 distractors =
# sentences from elsewhere (rejected if they appear in this passage, so every
# example is unambiguous). Fully automatic, no LLM needed.
N_TRAIN, N_VAL, N_TEST = 2000, 500, 500
K_OPTS, OPT_C = 4, 224   # options are full sentences (<=200 chars)
rng = random.Random(0)

sents = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if 30 <= len(s.strip()) <= 200]
print(f"usable sentences: {len(sents)}")
# Worst case per passage: 8000-char target / 30-char min sentence ~= 267
# distinct sentences. Passages sample WITH replacement across the 3000
# examples (the corpus is ~5MB; 3000 x ~5000-char passages would need ~15MB
# of unique text), but WITHOUT replacement within a passage.
assert len(sents) >= 1000, (
    f"too few distinct sentences for 3000 long passages: have {len(sents)}, "
    f"need >= 1000 (worst case ~267 distinct 30-char sentences per 8000-char passage)")

def make_passage():
    target = rng.randint(2000, 8000)
    idxs = rng.sample(range(len(sents)), 400)
    chosen, total = [], 0
    for i in idxs:
        s = sents[i]
        if total + len(s) + 1 > CTX_LONG:
            continue
        chosen.append(s)
        total += len(s) + 1
        if total >= target:
            break
    passage = " ".join(chosen)
    assert 2000 <= len(passage) <= CTX_LONG, f"bad passage length {len(passage)}"
    return passage, chosen

def make_taskC(n):
    ctx, opts, lab = [], [], []
    for _ in range(n):
        passage, constituents = make_passage()
        const_set = set(constituents)
        true = rng.choice(constituents)
        distract = []
        while len(distract) < K_OPTS - 1:
            d = sents[rng.randrange(len(sents))]
            if d not in const_set and d not in distract and d not in passage:
                distract.append(d)
        options = [true] + distract
        rng.shuffle(options)
        ctx.append(passage); opts.append(options); lab.append(options.index(true))
    return ctx, opts, lab

def tok_pad(s, maxlen):
    ids = [stoi.get(c, 0) for c in s[:maxlen]]
    L = len(ids)
    return ids + [0] * (maxlen - L), L

def to_tensors_c(ctx, opts, lab):
    Bc = len(ctx)
    C = torch.zeros(Bc, CTX_LONG, dtype=torch.long)
    Cl = torch.zeros(Bc, dtype=torch.long)
    O = torch.zeros(Bc, K_OPTS, OPT_C, dtype=torch.long)
    Ol = torch.zeros(Bc, K_OPTS, dtype=torch.long)
    for b in range(Bc):
        ids, L = tok_pad(ctx[b], CTX_LONG); C[b] = torch.tensor(ids); Cl[b] = L
        for k in range(K_OPTS):
            ids, L = tok_pad(opts[b][k], OPT_C); O[b, k] = torch.tensor(ids); Ol[b, k] = L
    return {"ctx": C, "ctx_len": Cl, "opts": O, "opt_len": Ol,
            "label": torch.tensor(lab, dtype=torch.long)}

tasks = {}
for sname, n in [("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST)]:
    tc, to, tl = make_taskC(n)
    tasks[sname] = to_tensors_c(tc, to, tl)
    _lens = tasks[sname]["ctx_len"]
    print(f"taskC {sname}: {n} examples, ctx_len min {_lens.min().item()} "
          f"max {_lens.max().item()} mean {_lens.float().mean().item():.0f}", flush=True)
n_pass = sum(tasks[s]["label"].shape[0] for s in ["train", "val", "test"])
assert n_pass == N_TRAIN + N_VAL + N_TEST == 3000, f"passage count wrong: {n_pass}"
print("random-guess baseline accuracy: 25.0%")
"""

# ------------------------------------------------- JevChoiceHead + decide ---
# Byte-identical to V4 except the smoke test runs on this session's backbone
# (BB is bound in the build cell of each notebook).
HEAD_SRC = """class JevChoiceHead(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = torch.nn.Linear(dim, dim, bias=False)
        self.k = torch.nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
    def forward(self, H, Hmask, O):
        Q = self.q(O)                                   # (B,K,D)
        K = self.k(H)                                   # (B,T,D)
        s = torch.einsum('bkd,btd->bkt', Q, K) * self.scale
        s = s.masked_fill(~Hmask[:, None, :], float('-inf'))
        a = torch.softmax(s, dim=-1)                    # (B,K,T)
        C = torch.einsum('bkt,btd->bkd', a, H)          # option-specific ctx
        return torch.einsum('bkd,bkd->bk', C, O)        # (B,K) logits

def bb_forward(model, idx):
    B, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=idx.device))[None]
    reset = (idx == model.newline_id).float() if model.newline_id is not None else None
    for blk in model.blocks:
        x = blk(x, reset)
    return model.ln_f(x)

def encode_opts(model, O, Ol):
    b, K, To = O.shape
    Ho = bb_forward(model, O.reshape(b * K, To))
    ar = torch.arange(To, device=O.device)[None, :]
    om = (ar < Ol.reshape(b * K, 1)).float()
    Ov = (Ho * om[..., None]).sum(1) / Ol.reshape(b * K, 1).clamp_min(1).float()
    return Ov.reshape(b, K, -1)

def decide(model, head, C, Cl, O, Ol):
    H = bb_forward(model, C)
    Ov = encode_opts(model, O, Ol)
    Hmask = torch.arange(H.shape[1], device=H.device)[None, :] < Cl[:, None]
    return head(H, Hmask, Ov)

# shape smoke test on a real long example (no_grad, then release)
_h = JevChoiceHead(DIM).to(device)
_t = tasks["train"]
with torch.no_grad():
    _lg = decide(BB, _h, _t["ctx"][:1].to(device), _t["ctx_len"][:1].to(device),
                 _t["opts"][:1].to(device), _t["opt_len"][:1].to(device))
assert _lg.shape == (1, K_OPTS), f"bad decide shape {_lg.shape}"
print("decide smoke OK (long context):", tuple(_lg.shape),
      f"{sum(p.numel() for p in _h.parameters())} head params")
del _h, _lg
torch.cuda.empty_cache()
"""

# ------------------------------------------------------------- calibrate ---
# Byte-identical to V4 (cell text verified by diff against the V4 notebook).
CALIB_SRC = """def fit_temperature(logits, labels):
    T = torch.nn.Parameter(torch.ones(1, device=logits.device))
    opt_t = torch.optim.LBFGS([T], lr=0.1, max_iter=50)
    def closure():
        opt_t.zero_grad()
        loss = F.cross_entropy(logits / T.clamp_min(1e-3), labels)
        loss.backward()
        return loss
    opt_t.step(closure)
    return float(T.clamp_min(1e-3))

def ece_score(probs, labels, n_bins=15):
    conf, pred = probs.max(-1)
    bins = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            acc_b = (pred[m] == labels[m]).float().mean()
            con_b = conf[m].mean()
            e += (m.float().mean() * (acc_b - con_b).abs()).item()
    return e

def brier_score(probs, labels, K=K_OPTS):
    oh = F.one_hot(labels, K).float()
    return float(((probs - oh) ** 2).sum(-1).mean())

def forward_all(model, head, split, batch=EVAL_B):
    t = tasks[split]
    outs = []
    model.eval(); head.eval()
    with torch.no_grad():
        for i in range(0, t["label"].shape[0], batch):
            b = slice(i, i + batch)
            outs.append(decide(model, head,
                               t["ctx"][b].to(device), t["ctx_len"][b].to(device),
                               t["opts"][b].to(device), t["opt_len"][b].to(device)))
    return torch.cat(outs)

results = {}
for mname, (model, head) in trained.items():
    test_logits = forward_all(model, head, "test")
    val_logits = forward_all(model, head, "val")
    te, va = tasks["test"], tasks["val"]
    T = fit_temperature(val_logits, va["label"].to(device))
    raw_p = torch.softmax(test_logits, -1)
    cal_p = torch.softmax(test_logits / T, -1)
    acc = float((test_logits.argmax(-1) == te["label"].to(device)).float().mean())
    results[f"{mname}/taskC"] = {
        "acc": acc, "ece_raw": ece_score(raw_p, te["label"].to(device)),
        "ece_cal": ece_score(cal_p, te["label"].to(device)),
        "brier": brier_score(cal_p, te["label"].to(device)), "temp": T}
    r = results[f"{mname}/taskC"]
    print(f"{mname} taskC: acc {r['acc']*100:.1f}% | "
          f"ECE {r['ece_raw']:.3f} -> {r['ece_cal']:.3f} (T={r['temp']:.2f}) | "
          f"Brier {r['brier']:.3f}", flush=True)
print("CALIBRATIONDONE")
"""

# ============================================================ model cells ===

TITLE_MD = {
    "v8a": """# JEV-V4a — long-context Jev model: V8-A (split session)

V4's combined ~3.5h run doesn't fit Colab's runtime limit (reclaimed twice
mid-finetune), so V4 runs as two single-model sessions: **V4a** trains
jev-v8a here, **V4b** trains jev-transformer separately. Protocol, seeds,
data, and hyperparams are identical to the V4 spec — the two sessions are
directly comparable, and their result files combine into V4's tables.

This session: **jev-v8a** only.
- **taskC** (new): long-context retrieval-decision — a 2k–8k-char passage,
  4 options, exactly one sentence taken verbatim from the passage.
  2000/500/500 train/val/test. Fully automatic, no LLM needed.
- Model: ~15.09M params (DIM 352 × 8 layers × 8 heads; pos_emb sized to 8192
  for long contexts).
- Stage 0: char-LM pre-train, 1500 steps, short 128-token windows (as V3).
- Stage 1: end-to-end decision finetune on taskC only (backbone unfrozen),
  micro-batch 2 with ×4 gradient accumulation, 6 epochs.
- Stage 2: temperature scaling.
- **Latency sweep**: T = 256…8192, batch=1, eager fp32, 10 warmup / 100 timed.
  CUDA OOM is recorded as a finding.
""",
    "transformer": """# JEV-V4b — long-context Jev model: Transformer (split session)

V4's combined ~3.5h run doesn't fit Colab's runtime limit (reclaimed twice
mid-finetune), so V4 runs as two single-model sessions: **V4a** trains
jev-v8a separately, **V4b** trains jev-transformer here. Protocol, seeds,
data, and hyperparams are identical to the V4 spec — the two sessions are
directly comparable, and their result files combine into V4's tables.

This session: **jev-transformer** only.
- **taskC** (new): long-context retrieval-decision — a 2k–8k-char passage,
  4 options, exactly one sentence taken verbatim from the passage.
  2000/500/500 train/val/test. Fully automatic, no LLM needed.
- Model: ~15.09M params (DIM 352 × 8 layers × 8 heads; pos_emb sized to 8192
  for long contexts).
- Stage 0: char-LM pre-train, 1500 steps, short 128-token windows (as V3).
- Stage 1: end-to-end decision finetune on taskC only (backbone unfrozen),
  micro-batch 2 with ×4 gradient accumulation, 6 epochs.
- Stage 2: temperature scaling.
- **Latency sweep**: T = 256…8192, batch=1, eager fp32, 10 warmup / 100 timed.
  CUDA OOM is recorded as a finding.
""",
}

BUILD_SRC = {
    "v8a": """# V4a: v8a backbone ONLY (jev-transformer trains in the V4b session).
# Same construction recipe as V4: DIM 352 x 8 layers x 8 heads, pos_emb sized
# to CTX_LONG=8192 (V3's 128-row table would IndexError past T=128).
# ffn_hidden=1343 holds the 14.5-15.5M band; state_dim stays 256.
# Comparability note: torch.manual_seed(0) is set in Setup exactly as in V4,
# but only one backbone is constructed here, so implicit global-RNG draws for
# weight init / stage-0 batches differ from the combined run. All EXPLICIT
# seeds are identical: taskC uses random.Random(0), head init uses
# torch.manual_seed(1234), batch order uses torch.Generator().manual_seed(0).
DIM, LAYERS, HEADS, CTX_LONG = 352, 8, 8, 8192
v8a = TinyLM(vocab, DIM, LAYERS, HEADS, CTX_LONG,
    mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
    ffn_hidden=1343, newline_id=nl_id).to(device)
_floors = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
for _i, _blk in enumerate(v8a.blocks):
    _blk.mix.forget_floor.data.fill_(_floors[_i])
print("key_proj bias (expect 2.0):", float(v8a.blocks[0].mix.key_proj.bias[0]))
print("write_proj bias (expect 0.0):", float(v8a.blocks[0].mix.write_proj.bias[0]))
# JevChoiceHead (defined in section 6): q,k = Linear(DIM,DIM,bias=False)
HEAD_P = 2 * DIM * DIM
pv = sum(p.numel() for p in v8a.parameters()) + HEAD_P
print(f"jev-v8a total (backbone+head): {pv/1e6:.3f}M params")
assert 14.5e6 <= pv <= 15.5e6, f"v8a outside 15M band: {pv}"
print("BUDGET GATE PASSED (15M band)")
# Long-context smoke: catches pos_emb indexing and Triton-scan issues at
# T=8192 BEFORE the expensive training stages.
with torch.no_grad():
    _probe = torch.randint(0, vocab, (1, CTX_LONG)).to(device)
    _lg, _ = v8a(_probe)
    assert _lg.shape == (1, CTX_LONG, vocab), ("v8a", _lg.shape)
    print("long-context forward smoke OK (T=8192: pos_emb + Triton scan)")
del _probe
torch.cuda.empty_cache()
MNAME, BB = "jev-v8a", v8a
""",
    "transformer": """# V4b: transformer backbone ONLY (jev-v8a trains in the V4a session).
# Same construction recipe as V4: DIM 352 x 8 layers x 8 heads, pos_emb sized
# to CTX_LONG=8192 (V3's 128-row table would IndexError past T=128).
# Comparability note: torch.manual_seed(0) is set in Setup exactly as in V4,
# but only one backbone is constructed here, so implicit global-RNG draws for
# weight init / stage-0 batches differ from the combined run. All EXPLICIT
# seeds are identical: taskC uses random.Random(0), head init uses
# torch.manual_seed(1234), batch order uses torch.Generator().manual_seed(0).
DIM, LAYERS, HEADS, CTX_LONG = 352, 8, 8, 8192
attention = TinyLM(vocab, DIM, LAYERS, HEADS, CTX_LONG,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h)).to(device)
# JevChoiceHead (defined in section 6): q,k = Linear(DIM,DIM,bias=False)
HEAD_P = 2 * DIM * DIM
pa = sum(p.numel() for p in attention.parameters()) + HEAD_P
print(f"jev-transformer total (backbone+head): {pa/1e6:.3f}M params")
assert 14.5e6 <= pa <= 15.5e6, f"transformer outside 15M band: {pa}"
print("BUDGET GATE PASSED (15M band)")
# Long-context smoke: catches pos_emb indexing issues at T=8192 BEFORE the
# expensive training stages.
with torch.no_grad():
    _probe = torch.randint(0, vocab, (1, CTX_LONG)).to(device)
    _lg, _ = attention(_probe)
    assert _lg.shape == (1, CTX_LONG, vocab), ("attention", _lg.shape)
    print("long-context forward smoke OK (T=8192: pos_emb)")
del _probe
torch.cuda.empty_cache()
MNAME, BB = "jev-transformer", attention
""",
}

STAGE0_SRC = """STEPS, EVAL, LR = 1500, 250, 3e-4
opt = torch.optim.AdamW(BB.parameters(), lr=LR)

def val_loss(m):
    m.eval()
    with torch.no_grad():
        return sum(float(m(*get_batch("val"))[1]) for _ in range(10)) / 10

for step in range(1, STEPS + 1):
    BB.train()
    xb, yb = get_batch("train")
    _, loss = BB(xb, yb)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(BB.parameters(), 1.0)
    opt.step()
    if step % EVAL == 0 or step == 1:
        vl = val_loss(BB)
        print(f"step {step:5d} | {MNAME}: val {vl:.3f} (ppl {math.exp(vl):.1f})",
              flush=True)
print("BACKBONETRAININGDONE")
print(f"FINAL {MNAME}: val {val_loss(BB):.3f}")
# NOTE: backbone stays UNFROZEN — V4 finetunes end-to-end on taskC (like V2/V3)
"""

STAGE1_SRC = """# Long contexts (up to 8192 tokens) x 15M params on a 16GB T4:
# micro-batch 2 with x4 gradient accumulation -> effective batch 8.
# 6 epochs x 250 optimizer steps = 1500 steps (V3 used 200 steps at short
# context; 25 epochs here would be ~3h — infeasible, and unnecessary).
import copy
EPOCHS, MICRO, ACCUM, LR_FT = 6, 2, 4, 3e-4
EVAL_B = 8
trained = {}
mname, bb = MNAME, BB
model = copy.deepcopy(bb).to(device)
torch.manual_seed(1234)  # identical head init across both sessions
head = JevChoiceHead(DIM).to(device)
opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                        lr=LR_FT)
tr = tasks["train"]; va = tasks["val"]
g = torch.Generator().manual_seed(0)  # identical batch order across sessions
best_acc, best_state = 0.0, None
n_steps = tr["label"].shape[0] // MICRO
assert n_steps % ACCUM == 0, "micro-steps must divide evenly into accum groups"
for ep in range(EPOCHS):
    model.train(); head.train()
    perm = torch.randperm(tr["label"].shape[0], generator=g)
    opt.zero_grad()
    for si in range(n_steps):
        b = perm[si*MICRO:(si+1)*MICRO]
        logits = decide(model, head,
                        tr["ctx"][b].to(device), tr["ctx_len"][b].to(device),
                        tr["opts"][b].to(device), tr["opt_len"][b].to(device))
        loss = F.cross_entropy(logits, tr["label"][b].to(device)) / ACCUM
        loss.backward()
        if (si + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            opt.zero_grad()
    model.eval(); head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, va["label"].shape[0], EVAL_B):
            b = slice(i, i + EVAL_B)
            vl = decide(model, head,
                        va["ctx"][b].to(device), va["ctx_len"][b].to(device),
                        va["opts"][b].to(device), va["opt_len"][b].to(device))
            lbl = va["label"][b].to(device)
            correct += (vl.argmax(-1) == lbl).sum().item()
            total += lbl.shape[0]
    acc = correct / total
    print(f"{mname} taskC epoch {ep+1}/{EPOCHS}: val acc {acc*100:.1f}%",
          flush=True)
    if acc > best_acc:
        best_acc = acc
        best_state = (
            {k: v.cpu().clone() for k, v in model.state_dict().items()},
            {k: v.cpu().clone() for k, v in head.state_dict().items()})
ms, hs = best_state
model.load_state_dict({k: v.to(device) for k, v in ms.items()})
head.load_state_dict({k: v.to(device) for k, v in hs.items()})
trained[mname] = (model, head)
print(f"{mname} taskC: best val acc {best_acc*100:.1f}% "
      f"({sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())} params)",
      flush=True)
del ms, hs
torch.cuda.empty_cache()
print("FINETUNEDONE")
"""

SWEEP_TMPL = """# The core V4 question: where does attention's latency curve bend?
# Inference-only, batch=1, torch.no_grad, eager fp32 — the honest comparable
# mode (V3 showed compile+fp16 backfires for both at batch=1 tiny scale).
# Each measurement: 10 warmup + 100 timed, cuda.synchronize.
# A CUDA OOM is caught and recorded as "OOM" — an OOM IS a finding
# (attention may OOM where the linear substrate doesn't).
SWEEP_T = [256, 512, 1024, 2048, 4096, 8192]

def sweep_batch(T):
    passage = text[:T]
    assert len(passage) == T, f"passage length {len(passage)} != {T}"
    ids = [stoi.get(c, 0) for c in passage]
    C = torch.tensor([ids], dtype=torch.long).to(device)
    Cl = torch.tensor([T], dtype=torch.long).to(device)
    O = torch.zeros(1, K_OPTS, OPT_C, dtype=torch.long).to(device)
    Ol = torch.zeros(1, K_OPTS, dtype=torch.long).to(device)
    for k in range(K_OPTS):
        oids = [stoi.get(c, 0) for c in sents[k][:OPT_C]]
        O[0, k, :len(oids)] = torch.tensor(oids, device=device)
        Ol[0, k] = len(oids)
    return C, Cl, O, Ol

def bench_T(model, head, T, iters=100):
    model.eval(); head.eval()
    C, Cl, O, Ol = sweep_batch(T)
    try:
        with torch.no_grad():
            for _ in range(10):
                decide(model, head, C, Cl, O, Ol)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(iters):
                decide(model, head, C, Cl, O, Ol)
            torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return None
        raise

sweep = {}
model, head = trained[MNAME]
for T in SWEEP_T:
    ms = bench_T(model, head, T)
    sweep[(MNAME, T)] = ms
    print(f"{MNAME} T={T:5d}: {'OOM' if ms is None else f'{ms:7.1f} ms/decision'}",
          flush=True)
torch.cuda.empty_cache()

print()
print(f"=== JEV-__TAG__ LATENCY SWEEP ({MNAME}) ===")
print(f"{'T (chars)':>10s} {(MNAME + ' ms'):>18s}")
for T in SWEEP_T:
    a = sweep[(MNAME, T)]
    sa = "OOM" if a is None else f"{a:.1f}"
    print(f"{T:>10d} {sa:>18s}")

# Crossover needs BOTH models' sweeps; the two sessions' result files are
# combined locally into V4's crossover table. Recorded here as pending.
crossover_line = ("crossover: single-model session — combine the V4a and V4b "
                  "sweeps to compute the crossover")
print(crossover_line)
print("LATENCYDONE")
"""

SAVE_TMPL = """print()
print(f"=== JEV-__TAG__ TEST RESULTS ({MNAME}) ===")
print(f"{'config':22s} {'task':6s} {'acc':>6s} {'ece_raw':>8s} {'ece_cal':>8s} {'brier':>7s}")
r = results[f"{MNAME}/taskC"]
print(f"{MNAME:22s} {'taskC':6s} {r['acc']*100:5.1f}% {r['ece_raw']:8.3f} {r['ece_cal']:8.3f} {r['brier']:7.3f}")
print()
print(f"=== JEV-__TAG__ LATENCY SWEEP ({MNAME}) ===")
print(f"{'T (chars)':>10s} {(MNAME + ' ms'):>18s}")
for T in SWEEP_T:
    a = sweep[(MNAME, T)]
    sa = "OOM" if a is None else f"{a:.1f}"
    print(f"{T:>10d} {sa:>18s}")
print(crossover_line)

sweep_json = {str(T): {MNAME: sweep[(MNAME, T)]} for T in SWEEP_T}
results["latency_sweep"] = sweep_json
results["crossover"] = crossover_line

with open("__RESULTS__.json", "w") as f:
    json.dump(results, f, indent=1)
with open("__RESULTS__.txt", "w") as f:
    f.write("JEV-__TAG__ RESULTS (test, eager fp32)\\n")
    f.write(f"{MNAME}/taskC: " + json.dumps(results[f"{MNAME}/taskC"]) + "\\n")
    f.write("latency_sweep: " + json.dumps(sweep_json) + "\\n")
    f.write("crossover: " + crossover_line + "\\n")
print()
print("SAVE DONE")
"""

SPECS = {
    "v8a": {
        "nb_file": "linear_attention_jev_v4a_v8a.ipynb",
        "tag": "V4a",
        "results": "results_jev_v4a_v8a",
    },
    "transformer": {
        "nb_file": "linear_attention_jev_v4b_transformer.ipynb",
        "tag": "V4b",
        "results": "results_jev_v4b_transformer",
    },
}

_models_cell = models_cell_src()

for which, spec in SPECS.items():
    NB = new_nb()
    md(NB, TITLE_MD[which])
    md(NB, "## 0. Setup")
    code(NB, SETUP_SRC)
    md(NB, "## 1. Data — identical corpus recipe to V1/V2/V3")
    code(NB, DATA_SRC)
    md(NB, "## 2. Models — inline models.py + triton_kernels.py (same as V1/V2/V3)")
    code(NB, _models_cell)
    md(NB, f"## 3. Build backbone ({spec['tag']}) — ~15M params, "
            "long-context positions, budget gate")
    code(NB, BUILD_SRC[which])
    md(NB, "## 4. Stage 0 — LM pre-train backbone (1500 steps, short windows)")
    code(NB, STAGE0_SRC)
    md(NB, "## 5. taskC — long-context retrieval-decision (2000–8000 char passages)")
    code(NB, TASKC_SRC)
    md(NB, "## 6. Jev decision model — JevChoiceHead + end-to-end decision forward")
    code(NB, HEAD_SRC)
    md(NB, "## 7. Stage 1 — end-to-end decision finetune on taskC (backbone UNFROZEN)")
    code(NB, STAGE1_SRC)
    md(NB, "## 8. Stage 2 — temperature scaling (calibration) + ECE / Brier")
    code(NB, CALIB_SRC)
    md(NB, "## 9. Latency sweep — O(T) vs O(T^2), T=256..8192")
    code(NB, SWEEP_TMPL.replace("__TAG__", spec["tag"]))
    md(NB, "## 10. Save results")
    code(NB, SAVE_TMPL.replace("__TAG__", spec["tag"]).replace(
        "__RESULTS__", spec["results"]))
    out = os.path.join(OUTDIR, spec["nb_file"])
    with open(out, "w") as f:
        json.dump(NB, f)
    print(f"notebook written: {out} ({len(NB['cells'])} cells)")
