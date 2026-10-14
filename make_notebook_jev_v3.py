"""Generate Colab notebook JEV-V3: 15M-param Jev models + speed techniques.

V2 (6.37M params) showed jev-v8a beats jev-transformer on decision accuracy
on both tasks, losing only on raw speed (14.8ms vs 7.1ms/decision).
V3 scales the SAME protocol to ~15M params per model and adds speed
techniques for the V8-A side:

  - "jev-v8a":         V8-A linear backbone + Jev decision head, ~15.18M
  - "jev-transformer": attention backbone   + Jev decision head, ~15.18M
  (totals include the Jev head; matched within 10k params)

Protocol (identical to V2 otherwise):
  Stage 0: LM pre-train both backbones from scratch (1500 steps).
  Stage 1: attach Jev head, UNFREEZE backbone, finetune END-TO-END on the
           decision tasks (same optimizer, LR, epochs, batch, seeds).
  Stage 2: temperature scaling per model.
  NEW: speed techniques for jev-v8a — torch.compile (reduce-overhead) +
       fp16 autocast; 4-config latency matrix; fast-path accuracy check
       (compiled fp16 vs eager fp32) to verify the accuracy win survives.
Metrics per model per task: accuracy, ECE raw/calibrated, Brier, latency.
"""
import json
import os

# Root of the linear-attention-lab checkout this generator builds against.
# Set the LINEAR_ATTENTION_LAB_DIR environment variable to point at yours.
LAB_DIR = os.environ.get("LINEAR_ATTENTION_LAB_DIR", "/path/to/linear-attention-lab")

os.makedirs(LAB_DIR + "/experiments/jev-v3",
            exist_ok=True)

NB = {"nbformat": 4, "nbformat_minor": 0,
      "metadata": {"kernelspec": {"display_name": "Python 3",
                                  "language": "python", "name": "python3"}},
      "cells": []}

def md(src):
    NB["cells"].append({"cell_type": "markdown", "metadata": {},
                        "source": src.splitlines(keepends=True)})

def code(src):
    NB["cells"].append({"cell_type": "code", "metadata": {},
                        "source": src.splitlines(keepends=True),
                        "outputs": [], "execution_count": None})

md("""# JEV-V3 — 15M-param Jev models: V8-A vs Transformer + speed techniques

V2 (6.4M params) showed a complete end-to-end Jev-style decision model on the
V8-A linear substrate beats the same model on the Transformer substrate on
decision accuracy — both tasks — losing only on raw per-decision speed.

V3 scales the identical protocol to **~15M params per model** (budget band
14.5–15.5M, matched within 10k) and fights back on speed: the V8-A side gets
`torch.compile(mode="reduce-overhead")` + fp16 autocast, a 4-config latency
matrix, and a fast-path accuracy check proving the accuracy win survives the
fast path.

Recipe:
- Model A "jev-v8a": V8-A backbone (DIM 392, 8 layers) + Jev head.
- Model B "jev-transformer": attention backbone (DIM 392, 8 layers) + Jev head.
- Stage 0: char-LM pre-train both backbones from scratch (1500 steps).
- Stage 1: decision finetune END-TO-END (backbone unfrozen), same optimizer,
  LR, epochs, batch size, seeds, batch order for both models.
- Stage 2: temperature scaling per model (calibration).
- Speed: torch.compile + fp16 autocast on jev-v8a; latency matrix
  (eager fp32 vs compiled fp16, both models); fast-path accuracy check.
- Metrics: accuracy, ECE (raw + calibrated), Brier, per-decision latency.
""")

md("## 0. Setup")
code("""import torch
import triton
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
device = "cuda" if torch.cuda.is_available() else "cpu"
assert torch.cuda.is_available(), "JEV-V3 needs the T4 GPU"
torch.manual_seed(0)
import math, random, re, time, json
random.seed(0)
""")

md("## 1. Data — identical corpus recipe to V8-A / JEV-V1 / JEV-V2")
code("""import os, urllib.request
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

SEQ, BS = 128, 32
def get_batch(split="train"):
    d = train_data if split == "train" else val_data
    i = torch.randint(0, len(d) - SEQ - 1, (BS,))
    x = torch.stack([d[j:j+SEQ] for j in i]).to(device)
    y = torch.stack([d[j+1:j+SEQ+1] for j in i]).to(device)
    return x, y
""")

md("## 2. Models — inline models.py + triton_kernels.py (same as V1/V2)")
_models_src = open(LAB_DIR + "/models.py").read().replace(
    '"""Linear-recurrent replacement for Transformer self-attention.',
    '"""(inlined) Linear-recurrent replacement for Transformer self-attention.').replace(
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n",
    "import torch.nn as nn\nimport torch.nn.functional as F\n")
_kern_src = open(LAB_DIR + "/triton_kernels.py").read().replace(
    '"""V6: Triton-fused selective scan for the gated segmented state space.',
    '"""(inlined) V6/V7/V8-A: Triton-fused scans.').replace(
    "import torch\nimport torch.nn as nn\nimport triton\nimport triton.language as tl\n",
    "import triton\nimport triton.language as tl\n")
code(_models_src + "\n\n" + _kern_src)

md("## 3. Build backbones — V8-A + attention at ~15M params, budget gate")
code("""# V3 scale-up: DIM 392 x 8 layers. V8-A ffn_hidden=1583 chosen so the two
# model totals (backbone + Jev head) match within 10k; V8-A state_dim stays
# 256 (Triton kernel fuses a 256-wide state — scaling comes from DIM/LAYERS).
DIM, LAYERS, HEADS = 392, 8, 8
attention = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: CausalSelfAttention(d, h)).to(device)
v8a = TinyLM(vocab, DIM, LAYERS, HEADS, SEQ,
    mixer_fn=lambda d, h: SelectiveSegmentedStateV8A(d, state_dim=256),
    ffn_hidden=1583, newline_id=nl_id).to(device)
_floors = torch.linspace(math.log(0.3/0.7), math.log(0.9/0.1), LAYERS)
for _i, _blk in enumerate(v8a.blocks):
    _blk.mix.forget_floor.data.fill_(_floors[_i])
print("key_proj bias (expect 2.0):", float(v8a.blocks[0].mix.key_proj.bias[0]))
print("write_proj bias (expect 0.0):", float(v8a.blocks[0].mix.write_proj.bias[0]))
# JevChoiceHead (defined in section 6): q,k = Linear(DIM,DIM,bias=False)
HEAD_P = 2 * DIM * DIM
pa = sum(p.numel() for p in attention.parameters()) + HEAD_P
pv = sum(p.numel() for p in v8a.parameters()) + HEAD_P
print(f"jev-transformer total (backbone+head): {pa/1e6:.3f}M params")
print(f"jev-v8a         total (backbone+head): {pv/1e6:.3f}M params")
assert 14.5e6 <= pa <= 15.5e6, f"transformer outside 15M band: {pa}"
assert 14.5e6 <= pv <= 15.5e6, f"v8a outside 15M band: {pv}"
assert abs(pa - pv) <= 10000, f"budget not matched within 10k: diff {abs(pa-pv)}"
print("BUDGET GATE PASSED (15M band, matched within 10k)")
backbones = {"attention": attention, "v8a": v8a}
""")

md("## 4. Stage 0 — LM pre-train both backbones (1500 steps, identical to V1/V2)")
code("""STEPS, EVAL, LR = 1500, 250, 3e-4
opt = {n: torch.optim.AdamW(m.parameters(), lr=LR) for n, m in backbones.items()}

def val_loss(m):
    m.eval()
    with torch.no_grad():
        return sum(float(m(*get_batch("val"))[1]) for _ in range(10)) / 10

for step in range(1, STEPS + 1):
    for name, m in backbones.items():
        m.train()
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt[name].zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt[name].step()
    if step % EVAL == 0 or step == 1:
        msg = f"step {step:5d}"
        for name, m in backbones.items():
            vl = val_loss(m)
            msg += f" | {name}: val {vl:.3f} (ppl {math.exp(vl):.1f})"
        print(msg, flush=True)
print("BACKBONETRAININGDONE")
for name, m in backbones.items():
    print(f"FINAL {name}: val {val_loss(m):.3f}")
# NOTE: backbones stay UNFROZEN — V3 finetunes end-to-end (like V2)
""")

md("## 5. Decision tasks — generated in-domain from the same corpus (char-level)")
code("""N_TRAIN, N_VAL, N_TEST = 2000, 500, 500
K_OPTS, CTX_MAX, OPT_MAX = 4, 64, 32
rng = random.Random(0)

sents = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if 30 <= len(s.strip()) <= 200]
print(f"usable sentences: {len(sents)}")
# task B samples 3*n sentences per split (worst case 3*N_TRAIN); require 2x headroom
# (V1 bugfix: corpus yields ~15,765 usable sentences, NOT > 20,000)
assert len(sents) >= 6 * N_TRAIN, f"too few sentences! need {6 * N_TRAIN}, have {len(sents)}"

def far_sample(i, n, k, gap=5):
    out = []
    while len(out) < k:
        j = rng.randrange(n)
        if abs(j - i) > gap and j not in out:
            out.append(j)
    return out

def make_taskA(n):
    ctx, opts, lab = [], [], []
    idxs = rng.sample(range(len(sents) - 1), n)
    short = [s[:OPT_MAX] for s in sents]
    for i in idxs:
        c = sents[i][-CTX_MAX:]
        true = short[i + 1]
        d = [short[j] for j in far_sample(i, len(sents), K_OPTS - 1)]
        options = [true] + d
        rng.shuffle(options)
        ctx.append(c); opts.append(options); lab.append(options.index(true))
    return ctx, opts, lab

_word_pool = list({w for s in sents[:20000] for w in re.findall(r"[A-Za-z']{4,}", s)})
print(f"word pool: {len(_word_pool)}")
def make_taskB(n):
    ctx, opts, lab = [], [], []
    made = 0
    idxs = rng.sample(range(len(sents)), n * 3)
    for i in idxs:
        if made >= n: break
        s = sents[i]
        words = re.findall(r"[A-Za-z']{4,}", s)
        if not words: continue
        w = rng.choice(words)
        masked = re.sub(r'\\b' + re.escape(w) + r'\\b', '____', s, count=1)
        if '____' not in masked or len(masked) > CTX_MAX + 10: continue
        d = []
        while len(d) < K_OPTS - 1:
            w2 = rng.choice(_word_pool)
            if w2 != w and w2 not in d: d.append(w2)
        options = [w] + d
        rng.shuffle(options)
        ctx.append(masked[-CTX_MAX:]); opts.append(options); lab.append(options.index(w))
        made += 1
    assert made == n, f"taskB short: {made}/{n}"
    return ctx, opts, lab

def tok_pad(s, maxlen):
    ids = [stoi.get(c, 0) for c in s[:maxlen]]
    L = len(ids)
    return ids + [0] * (maxlen - L), L

def to_tensors(ctx, opts, lab, ctx_max=CTX_MAX, opt_max=OPT_MAX):
    Bc = len(ctx)
    C = torch.zeros(Bc, ctx_max, dtype=torch.long)
    Cl = torch.zeros(Bc, dtype=torch.long)
    O = torch.zeros(Bc, K_OPTS, opt_max, dtype=torch.long)
    Ol = torch.zeros(Bc, K_OPTS, dtype=torch.long)
    for b in range(Bc):
        ids, L = tok_pad(ctx[b], ctx_max); C[b] = torch.tensor(ids); Cl[b] = L
        for k in range(K_OPTS):
            ids, L = tok_pad(opts[b][k], opt_max); O[b, k] = torch.tensor(ids); Ol[b, k] = L
    return {"ctx": C, "ctx_len": Cl, "opts": O, "opt_len": Ol,
            "label": torch.tensor(lab, dtype=torch.long)}

tasks = {}
for tname, fn in [("A", make_taskA), ("B", make_taskB)]:
    tc, to, tl = fn(N_TRAIN); vc, vo, vl = fn(N_VAL); ec, eo, el = fn(N_TEST)
    tasks[tname] = {"train": to_tensors(tc, to, tl), "val": to_tensors(vc, vo, vl),
                    "test": to_tensors(ec, eo, el)}
    print(f"task {tname}: train {N_TRAIN} / val {N_VAL} / test {N_TEST}")
print("random-guess baseline accuracy: 25.0%")
""")

md("## 6. Jev decision model — JevChoiceHead + end-to-end decision forward")
code("""class JevChoiceHead(torch.nn.Module):
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

# shape smoke test (grad-enabled path, no_grad just for the smoke)
_h = JevChoiceHead(DIM).to(device)
_t = tasks["A"]["train"]
with torch.no_grad():
    _lg = decide(v8a, _h, _t["ctx"][:8].to(device), _t["ctx_len"][:8].to(device),
                 _t["opts"][:8].to(device), _t["opt_len"][:8].to(device))
assert _lg.shape == (8, K_OPTS), f"bad decide shape {_lg.shape}"
print("decide smoke OK:", tuple(_lg.shape),
      f"{sum(p.numel() for p in _h.parameters())} head params")
""")

md("## 7. Stage 1 — end-to-end decision finetune (backbone UNFROZEN)")
code("""import copy
EPOCHS, BATCH, LR_FT = 25, 256, 3e-4
trained = {}
for mname, bb in [("jev-v8a", v8a), ("jev-transformer", attention)]:
    for tname in ["A", "B"]:
        model = copy.deepcopy(bb).to(device)
        torch.manual_seed(1234)  # identical head init across all 4 runs
        head = JevChoiceHead(DIM).to(device)
        opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                                lr=LR_FT)
        tr = tasks[tname]["train"]; va = tasks[tname]["val"]
        g = torch.Generator().manual_seed(0)  # identical batch order across runs
        best_acc, best_state = 0.0, None
        for ep in range(EPOCHS):
            model.train(); head.train()
            perm = torch.randperm(tr["label"].shape[0], generator=g)
            for i in range(0, len(perm), BATCH):
                b = perm[i:i+BATCH]
                logits = decide(model, head,
                                tr["ctx"][b].to(device), tr["ctx_len"][b].to(device),
                                tr["opts"][b].to(device), tr["opt_len"][b].to(device))
                loss = F.cross_entropy(logits, tr["label"][b].to(device))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(head.parameters()), 1.0)
                opt.step()
            model.eval(); head.eval()
            with torch.no_grad():
                vl = decide(model, head,
                            va["ctx"].to(device), va["ctx_len"].to(device),
                            va["opts"].to(device), va["opt_len"].to(device))
                acc = (vl.argmax(-1) == va["label"].to(device)).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = (
                    {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    {k: v.cpu().clone() for k, v in head.state_dict().items()})
        ms, hs = best_state
        model.load_state_dict({k: v.to(device) for k, v in ms.items()})
        head.load_state_dict({k: v.to(device) for k, v in hs.items()})
        trained[(mname, tname)] = (model, head)
        print(f"{mname} task {tname}: best val acc {best_acc*100:.1f}% "
              f"({sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())} params)",
              flush=True)
print("FINETUNEDONE")
""")

md("## 8. Stage 2 — temperature scaling (calibration) + ECE / Brier")
code("""def fit_temperature(logits, labels):
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

results = {}
for (mname, tname), (model, head) in trained.items():
    te = tasks[tname]["test"]; va = tasks[tname]["val"]
    model.eval(); head.eval()
    with torch.no_grad():
        test_logits = decide(model, head,
                             te["ctx"].to(device), te["ctx_len"].to(device),
                             te["opts"].to(device), te["opt_len"].to(device))
        val_logits = decide(model, head,
                            va["ctx"].to(device), va["ctx_len"].to(device),
                            va["opts"].to(device), va["opt_len"].to(device))
    T = fit_temperature(val_logits, va["label"].to(device))
    raw_p = torch.softmax(test_logits, -1)
    cal_p = torch.softmax(test_logits / T, -1)
    acc = float((test_logits.argmax(-1) == te["label"].to(device)).float().mean())
    results[f"{mname}/task{tname}"] = {
        "acc": acc, "ece_raw": ece_score(raw_p, te["label"].to(device)),
        "ece_cal": ece_score(cal_p, te["label"].to(device)),
        "brier": brier_score(cal_p, te["label"].to(device)), "temp": T}
    r = results[f"{mname}/task{tname}"]
    print(f"{mname} task {tname}: acc {r['acc']*100:.1f}% | "
          f"ECE {r['ece_raw']:.3f} -> {r['ece_cal']:.3f} (T={r['temp']:.2f}) | "
          f"Brier {r['brier']:.3f}", flush=True)
print("CALIBRATIONDONE")
""")

md("## 9. Speed techniques — torch.compile + fp16 autocast, latency matrix")
code("""# V3 fights back on speed: compile the models (reduce-overhead, stable on
# free T4) and run inference under fp16 autocast. Latency matrix covers both
# models x {eager fp32, compiled fp16}, same methodology as V2
# (batch=1, 10 warmup, 100 timed, cuda.synchronize).
fast = {}
def get_fast(mname, tname):
    if (mname, tname) not in fast:
        model, head = trained[(mname, tname)]
        fm = torch.compile(copy.deepcopy(model), mode="reduce-overhead")
        fh = torch.compile(copy.deepcopy(head), mode="reduce-overhead")
        fast[(mname, tname)] = (fm, fh)
    return fast[(mname, tname)]

def timed_decide(model, head, cb, cbl, ob, obl, fp16):
    if fp16:
        with torch.autocast("cuda", dtype=torch.float16):
            return decide(model, head, cb, cbl, ob, obl)
    return decide(model, head, cb, cbl, ob, obl)

def bench(model, head, fp16, iters=100):
    model.eval(); head.eval()
    t = tasks["A"]["test"]
    cb, cbl = t["ctx"][:1].to(device), t["ctx_len"][:1].to(device)
    ob, obl = t["opts"][:1].to(device), t["opt_len"][:1].to(device)
    with torch.no_grad():
        for _ in range(10):  # warmup (also triggers compilation)
            timed_decide(model, head, cb, cbl, ob, obl, fp16)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(iters):
            timed_decide(model, head, cb, cbl, ob, obl, fp16)
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000

lat = {}
configs = {}
for mname in ["jev-v8a", "jev-transformer"]:
    model, head = trained[(mname, "A")]
    configs[f"{mname}/eager/fp32"] = (model, head, False)
    fm, fh = get_fast(mname, "A")
    configs[f"{mname}/compiled/fp16"] = (fm, fh, True)
for cname, (m, h, fp16) in configs.items():
    ms = bench(m, h, fp16)
    lat[cname] = ms
    print(f"{cname}: {ms:.1f} ms/decision", flush=True)
results["latency_matrix"] = lat
print()
print("=== JEV-V3 LATENCY MATRIX ===")
print(f"{'config':32s} {'ms/decision':>12s}")
for cname in ["jev-v8a/eager/fp32", "jev-v8a/compiled/fp16",
              "jev-transformer/eager/fp32", "jev-transformer/compiled/fp16"]:
    print(f"{cname:32s} {lat[cname]:12.1f}")
print("LATENCYDONE")
""")

md("## 10. Fast-path accuracy — does the accuracy win survive compile+fp16?")
code("""# Re-run taskA/taskB test accuracy for jev-v8a in compiled-fp16 mode and
# compare against the eager-fp32 numbers from section 8.
fp_acc = {}
for tname in ["A", "B"]:
    te = tasks[tname]["test"]
    fm, fh = get_fast("jev-v8a", tname)
    fm.eval(); fh.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            logits = decide(fm, fh,
                            te["ctx"].to(device), te["ctx_len"].to(device),
                            te["opts"].to(device), te["opt_len"].to(device))
    acc = float((logits.argmax(-1) == te["label"].to(device)).float().mean())
    base = results[f"jev-v8a/task{tname}"]["acc"]
    dpp = (acc - base) * 100
    fp_acc[tname] = {"fp32_eager": base, "fp16_compiled": acc, "delta_pp": dpp}
    flag = "  <-- >1pp shift, NOTE" if abs(dpp) > 1 else ""
    print(f"jev-v8a task {tname}: eager fp32 {base*100:.1f}% vs "
          f"compiled fp16 {acc*100:.1f}% (delta {dpp:+.2f}pp){flag}", flush=True)
results["fast_path_accuracy"] = fp_acc
print()
print("=== JEV-V3 FAST-PATH ACCURACY ===")
for tname in ["A", "B"]:
    d = fp_acc[tname]
    print(f"task {tname}: eager fp32 {d['fp32_eager']*100:.1f}% | "
          f"compiled fp16 {d['fp16_compiled']*100:.1f}% | delta {d['delta_pp']:+.2f}pp")
""")

md("## 11. Save results")
code("""print()
print("=== JEV-V3 TEST RESULTS ===")
print(f"{'config':22s} {'task':6s} {'acc':>6s} {'ece_raw':>8s} {'ece_cal':>8s} {'brier':>7s}")
for key in sorted(results):
    if "/" not in key: continue
    r = results[key]
    cfg, tsk = key.split("/")
    print(f"{cfg:22s} {tsk:6s} {r['acc']*100:5.1f}% {r['ece_raw']:8.3f} {r['ece_cal']:8.3f} {r['brier']:7.3f}")
print()
print("substrate effect (taskA): ",
      f"jev-v8a {results['jev-v8a/taskA']['acc']*100:.1f}% vs "
      f"jev-transformer {results['jev-transformer/taskA']['acc']*100:.1f}%")
print("substrate effect (taskB): ",
      f"jev-v8a {results['jev-v8a/taskB']['acc']*100:.1f}% vs "
      f"jev-transformer {results['jev-transformer/taskB']['acc']*100:.1f}%")
print()
print("=== JEV-V3 LATENCY MATRIX ===")
for cname in ["jev-v8a/eager/fp32", "jev-v8a/compiled/fp16",
              "jev-transformer/eager/fp32", "jev-transformer/compiled/fp16"]:
    print(f"{cname:32s} {lat[cname]:.1f} ms/decision")
print()
print("=== JEV-V3 FAST-PATH ACCURACY ===")
for tname in ["A", "B"]:
    d = fp_acc[tname]
    print(f"task {tname}: eager fp32 {d['fp32_eager']*100:.1f}% | "
          f"compiled fp16 {d['fp16_compiled']*100:.1f}% | delta {d['delta_pp']:+.2f}pp")

with open("results_jev_v3.json", "w") as f:
    json.dump(results, f, indent=1)
with open("results_jev_v3.txt", "w") as f:
    f.write("JEV-V3 RESULTS (test, eager fp32)\\n")
    for key in sorted(results):
        if "/" not in key: continue
        f.write(f"{key}: {json.dumps(results[key])}\\n")
    f.write("latency_matrix: " + json.dumps(results["latency_matrix"]) + "\\n")
    f.write("fast_path_accuracy: " + json.dumps(results["fast_path_accuracy"]) + "\\n")
print()
print("SAVE DONE")
""")

with open(LAB_DIR + "/experiments/jev-v3/linear_attention_jev_v3.ipynb", "w") as f:
    json.dump(NB, f)
print(f"notebook written: {len(NB['cells'])} cells")
