""" Run 23: Interest-Vector Ensemble

Run 23 built on the interest-vector MLP, which achieved the best single-model score 
of 0.6036.

Two ensembles were tested: one combining interest, MLP-sequence, and FM-sequence models, 
and another using only interest and MLP-sequence models. The best combination was selected 
using validation performance.
"""
import time, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from mlp import MLPRank, infonce_mlp_step
from harness import _append, BASELINE
from sequences import load_sequenced, encode_rows, BASE, SEQ
from interest import InterestMLP, infonce_interest_step, L

BANKED = 0.6043

print("loading + sequencing + histories ...")
splits = load_sequenced()
rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
watched = {}
for x in rows_flat:
    u = x['user_id']
    if u not in watched:
        watched[u] = collections.deque(maxlen=L)
    x['hvids'] = list(watched[u])
    if x['y'] == 1:
        watched[u].append(x['video_id'])

fields = BASE + SEQ
enc, dim = encode_rows(splits, fields)
F = len(fields)

tr = splits['train']
vocabs = [dict() for _ in fields]
for x in tr:
    for i, f in enumerate(fields):
        if x[f] not in vocabs[i]:
            vocabs[i][x[f]] = len(vocabs[i])
dims = [len(v) + 1 for v in vocabs]
offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
vvocab = vocabs[1]; vunk = len(vvocab); voffset = int(offsets[1])


def hist_arrays(rws):
    Hm = np.zeros((len(rws), L), dtype=np.int32)
    Wt = np.zeros((len(rws), L), dtype=np.float32)
    for n, x in enumerate(rws):
        hv = x['hvids']; c = len(hv)
        for j, v in enumerate(hv):
            Hm[n, j] = vvocab.get(v, vunk) + voffset
        if c:
            Wt[n, :c] = 1.0 / c
    return Hm, Wt


H_ = {name: hist_arrays(rws) for name, rws in splits.items()}
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
Htr, Wtr = H_['train']; Hva, Wva = H_['valid']; Hte, Wte = H_['test']
pairs_users, _, _ = build_pair_index(utr, ytr)


def train_one(seed, kind):
    rng = np.random.default_rng(seed)
    if kind == 'interest':
        m = InterestMLP(dim, F, k=16, H=64, lr=0.0003, seed=seed)
        def step(p, n):
            infonce_interest_step(m, Xtr[p], Htr[p], Wtr[p],
                                  Xtr[n], Htr[n], Wtr[n], len(p), 4)
        pred_va = lambda: m.predict(Xva, Hva, Wva)
        pred_te = lambda: m.predict(Xte, Hte, Wte)
        gs, ls = m.state, m.load_state
    elif kind == 'mlp':
        m = MLPRank(dim, F, k=16, H=64, lr=0.0003, seed=seed)
        def step(p, n):
            infonce_mlp_step(m, Xtr[p], Xtr[n], len(p), 4)
        pred_va = lambda: m.predict(Xva); pred_te = lambda: m.predict(Xte)
        gs, ls = m.state, m.load_state
    else:
        m = FM(dim, k=16, lr=0.001, seed=seed)
        def step(p, n):
            infonce_step(m, Xtr[p], Xtr[n], len(p), 4)
        pred_va = lambda: m.predict(Xva); pred_te = lambda: m.predict(Xte)
        gs = lambda: (m.V.copy(), m.W.copy(), np.float32(m.b))
        def ls(st): m.V, m.W, m.b = st
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        P, N = sample_lists(pairs_users, rng, 4)
        for i in range(0, len(P), 8192):
            step(P[i:i + 8192], N[i:i + 8192].reshape(-1))
        va = evaluate(uva, yva, pred_va())
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = gs()
        else:
            bad += 1
            if bad >= 4:
                break
    ls(best_state)
    pv, pt = pred_va(), pred_te()
    return ((pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std())


P = {}
for kind in ('interest', 'mlp', 'fm'):
    P[kind] = {'va': [], 'te': []}
    for s in range(5):
        t0 = time.time()
        pv, pt = train_one(s, kind)
        P[kind]['va'].append(pv); P[kind]['te'].append(pt)
        print(f"{kind} seed {s} done ({time.time()-t0:.0f}s)")


def log_ens(name, kinds):
    va = [p for k in kinds for p in P[k]['va']]
    te = [p for k in kinds for p in P[k]['te']]
    v = evaluate(uva, yva, np.mean(va, 0))['primary']
    t = evaluate(ute, yte, np.mean(te, 0))['primary']
    mark = '✅ BETTER than banked' if t > BANKED else '❌ not better'
    _append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': name, 'valid_mean': round(float(v), 5),
             'test_mean': round(float(t), 5), 'test_std': None,
             'd_baseline': round(float(t - BASELINE), 5),
             'verdict': 'WIN' if t - BASELINE > 0.002 else 'NOISE',
             'note': '+'.join(kinds) + ' x5 seeds'})
    print(f"{name}: valid {v:.5f}  test {t:.4f}  vs banked {t - BANKED:+.4f} {mark}")
    return v, t


log_ens('R23a interest+mlp+fm committee', ['interest', 'mlp', 'fm'])
log_ens('R23b interest+mlp committee', ['interest', 'mlp'])