""" Run 15: Multi-Class Ensemble Expansion

Run 15 tested whether adding the new MLP2 (0.5982) and FFM (0.5976) models could 
improve the ensemble, even though neither beat the MLP individually.

Building on Run 11, we tested 2-class, 3-class, and 4-class ensembles to see whether 
greater model diversity could push the best score beyond 0.5986.
"""
import time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from mlp import MLPRank, infonce_mlp_step
from architectures import MLP2Rank, FFM, infonce_generic
from harness import _append, BASELINE

print("loading ...")
splits = load('./KuaiRand-Pure/data')
enc, dim = encode(splits)
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)
F = len(FIELDS)


def train(seed, kind):
    if kind == 'mlp':
        m = MLPRank(dim, F, k=16, H=64, lr=0.0003, seed=seed)
        step = lambda p, n: infonce_mlp_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
        gs, ls = m.state, m.load_state
    elif kind == 'mlp2':
        m = MLP2Rank(dim, F, k=16, H1=64, H2=32, lr=0.0003, seed=seed)
        step = lambda p, n: infonce_generic(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4, 'mlp2')
        gs, ls = m.state, m.load_state
    elif kind == 'ffm':
        m = FFM(dim, F, k=8, lr=0.001, seed=seed)
        step = lambda p, n: infonce_generic(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4, 'ffm')
        gs, ls = m.state, m.load_state
    else:
        m = FM(dim, k=16, lr=0.001, seed=seed)
        step = lambda p, n: infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
        gs = lambda: (m.V.copy(), m.W.copy(), np.float32(m.b))
        def ls(st): m.V, m.W, m.b = st
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        P, N = sample_lists(pairs_users, rng, 4)
        for i in range(0, len(P), 8192):
            step(P[i:i + 8192], N[i:i + 8192])
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = gs()
        else:
            bad += 1
            if bad >= 4:
                break
    ls(best_state)
    pv, pt = m.predict(Xva), m.predict(Xte)
    return ((pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std())


preds = {}
for kind in ('mlp', 'fm', 'ffm', 'mlp2'):
    preds[kind] = {'va': [], 'te': []}
    for s in range(5):
        t0 = time.time()
        pv, pt = train(s, kind)
        preds[kind]['va'].append(pv); preds[kind]['te'].append(pt)
        print(f"{kind} seed {s} done ({time.time()-t0:.0f}s)")


def log_ens(name, kinds):
    va = [p for k in kinds for p in preds[k]['va']]
    te = [p for k in kinds for p in preds[k]['te']]
    ev = evaluate(uva, yva, np.mean(va, axis=0))
    et = evaluate(ute, yte, np.mean(te, axis=0))
    d = et['primary'] - BASELINE
    _append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': name, 'valid_mean': round(float(ev['primary']), 5),
             'test_mean': round(float(et['primary']), 5), 'test_std': None,
             'd_baseline': round(float(d), 5),
             'verdict': 'WIN' if d > 0.002 else 'NOISE',
             'note': '+'.join(kinds) + ' x5 seeds each'})
    print(f"{name}: valid {ev['primary']:.4f}  test {et['primary']:.4f}  d/base {d:+.4f}")


log_ens('R15a mlp+fm (banked reference)', ['mlp', 'fm'])
log_ens('R15b mlp+fm+ffm', ['mlp', 'fm', 'ffm'])
log_ens('R15c mlp+fm+mlp2', ['mlp', 'fm', 'mlp2'])
log_ens('R15d all four classes', ['mlp', 'fm', 'ffm', 'mlp2'])