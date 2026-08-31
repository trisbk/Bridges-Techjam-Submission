""" Run 17: Final Performance Improvements

Run 17 tested three approaches to further improve the score.

R17a – Weighted Ensemble: Searched for the best weighting between the MLP and FM using 
validation data only.

R17b – Diverse Ensemble: Combined MLPs with different configurations and multiple FMs to 
increase model diversity.

R17c – Learning Rate Decay: Trained the MLP longer while gradually reducing the learning 
rate every eight epochs.
"""
import time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from mlp import MLPRank, infonce_mlp_step
from harness import _append, BASELINE, run_experiment

BANKED = 0.5986

print("loading ...")
splits = load('./KuaiRand-Pure/data')
enc, dim = encode(splits)
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)
F = len(FIELDS)


def train_model(seed, kind, k=16, H=64, lr=None, decay=False):
    if kind == 'mlp':
        m = MLPRank(dim, F, k=k, H=H, lr=lr or 0.0003, seed=seed)
        step = lambda p, n: infonce_mlp_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
        gs, ls = m.state, m.load_state
    else:
        m = FM(dim, k=k, lr=lr or 0.001, seed=seed)
        step = lambda p, n: infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
        gs = lambda: (m.V.copy(), m.W.copy(), np.float32(m.b))
        def ls(st): m.V, m.W, m.b = st
    rng = np.random.default_rng(seed)
    patience = 8 if decay else 4
    best, best_state, bad = -1, None, 0
    for ep in range(1, 61 if decay else 41):
        if decay and ep % 8 == 0:
            m.lr *= 0.5
        P, N = sample_lists(pairs_users, rng, 4)
        for i in range(0, len(P), 8192):
            step(P[i:i + 8192], N[i:i + 8192])
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = gs()
        else:
            bad += 1
            if bad >= patience:
                break
    ls(best_state)
    pv, pt = m.predict(Xva), m.predict(Xte)
    return ((pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std())


def log_result(name, valid_p, test_p, note):
    d = test_p - BASELINE
    mark = '✅ BETTER than banked' if test_p > BANKED else '❌ not better'
    _append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': name, 'valid_mean': round(float(valid_p), 5),
             'test_mean': round(float(test_p), 5), 'test_std': None,
             'd_baseline': round(float(d), 5),
             'verdict': 'WIN' if d > 0.002 else 'NOISE', 'note': note})
    print(f"{name}: valid {valid_p:.5f}  test {test_p:.4f}  "
          f"vs banked {test_p - BANKED:+.4f} {mark}")


# ---- base committee models (reused across a and b) ----
mlp_va, mlp_te, fm_va, fm_te = [], [], [], []
for s in range(5):
    pv, pt = train_model(s, 'mlp'); mlp_va.append(pv); mlp_te.append(pt)
    print(f"mlp seed {s} done")
for s in range(5):
    pv, pt = train_model(s, 'fm'); fm_va.append(pv); fm_te.append(pt)
    print(f"fm seed {s} done")

# ---- R17a: alpha searched on VALIDATION only ----
Mva, Fva = np.mean(mlp_va, 0), np.mean(fm_va, 0)
Mte, Fte = np.mean(mlp_te, 0), np.mean(fm_te, 0)
best_a, best_v = 0.5, -1
for a in np.arange(0.0, 1.01, 0.05):
    v = evaluate(uva, yva, a * Mva + (1 - a) * Fva)['primary']
    if v > best_v:
        best_v, best_a = v, a
test_p = evaluate(ute, yte, best_a * Mte + (1 - best_a) * Fte)['primary']
log_result(f'R17a weighted committee alpha={best_a:.2f}', best_v, test_p,
           'alpha chosen on validation grid; test measured once')

# ---- R17b: diverse-config MLP committee + FMs ----
div_va, div_te = [], []
for (H, k) in ((48, 12), (64, 16), (96, 24)):
    for s in range(2):
        pv, pt = train_model(s, 'mlp', k=k, H=H)
        div_va.append(pv); div_te.append(pt)
        print(f"mlp H={H} k={k} seed {s} done")
v = evaluate(uva, yva, np.mean(div_va + fm_va, 0))['primary']
t = evaluate(ute, yte, np.mean(div_te + fm_te, 0))['primary']
log_result('R17b diverse-config committee (6 MLP cfgs + 5 FM)', v, t,
           'H in 48/64/96, k in 12/16/24 diagonal')

# ---- R17c: lr decay, long patience (through harness, 3 seeds) ----
def train_fn(seed):
    pv, pt = None, None
    m = MLPRank(dim, F, k=16, H=64, lr=0.0003, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 61):
        if ep % 8 == 0:
            m.lr *= 0.5
        P, N = sample_lists(pairs_users, rng, 4)
        for i in range(0, len(P), 8192):
            infonce_mlp_step(m, Xtr[P[i:i + 8192]], Xtr[N[i:i + 8192].reshape(-1)],
                             len(P[i:i + 8192]), 4)
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= 8:
                break
    m.load_state(best_state)
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test': evaluate(ute, yte, m.predict(Xte))}

run_experiment(
    name='R17c MLP lr-decay long-train',
    hypothesis='Halving lr every 8 epochs with patience 8 lets the MLP settle '
               'into a finer optimum than fixed-lr early stopping allows.',
    rationale='Only training-schedule lever untried; costs nothing but time.',
    train_fn=train_fn, seeds=3, config={'lr0': 0.0003, 'halve_every': 8})