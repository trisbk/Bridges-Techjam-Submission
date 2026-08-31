"""FROZEN FINAL MODEL — the banked recipe, reproducible in one command.

Recipe (banked 31 Aug 2026, Run 33c — campaign 5's converged checkpoint;
previous champion R24b, 29 Aug, differed only by the absence of tab_n):
  - Features: kit base fields + causal sequence features, all computed from
    strictly-prior events (prev1, hist10, hist_n, auth_hist, hist30,
    tag_hist, gap), plus tab_n — a per-(user, surface) impression count
    (log-bucketed, label-free), the autonomous v2-loop's own
    residual-derived discovery
  - Objective: listwise InfoNCE, K=4 sampled within-user negatives
  - Model: Factorization Machine, k=16, lr=0.001, Adam, patience 4
  - Committee: seeds 0-4, per-model z-scored predictions averaged

Running this file retrains everything from raw data (~5 min, numpy only),
saves weights + predictions to frozen_model/, and prints the final scores.
No test label influences any training or selection step. Test-row
FEATURES use the same strictly-prior-events rule as everywhere else,
which under continuous updates includes earlier test-window events (the
streaming serving assumption, stated and measured in the README's
serving-assumption section; the frozen and daily regimes are measured in
staleness_ablation.py and protocolB_retrain.py).
"""
import os, csv, collections, time
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

OUT = './frozen_model'
os.makedirs(OUT, exist_ok=True)

print("loading + sequencing ...")
splits = load_sequenced()
vid2tag = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
tab_ct = {}
for x in rows_flat:
    u = x['user_id']
    if u not in hist:
        hist[u] = {'last30': collections.deque(maxlen=30),
                   'tag': collections.Counter(), 'last_t': None}
    h = hist[u]
    x['hist30'] = 'none' if not h['last30'] else str(int(10 * sum(h['last30']) / len(h['last30'])))
    tg = vid2tag.get(x['video_id'], 'UNK')
    tc = h['tag'][tg]
    x['tag_hist'] = str(tc) if tc < 3 else '3+'
    if h['last_t'] is None:
        x['gap'] = 'none'
    else:
        d = (x['date'] - h['last_t'][0]) * 86400_000 + (x['t'] - h['last_t'][1])
        x['gap'] = ('<1m' if d < 60_000 else '<1h' if d < 3_600_000
                    else '<1d' if d < 86_400_000 else '1d+')
    # tab_n: prior impression count on this row's surface (label-free,
    # self-exclusive: featurized before increment)
    kt = (u, x['tab'])
    n = tab_ct.get(kt, 0)
    x['tab_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
                  else '11-30' if n <= 30 else '31-100' if n <= 100
                  else '100+')
    tab_ct[kt] = n + 1
    h['last30'].append(x['y']); h['tag'][tg] += x['y']; h['last_t'] = (x['date'], x['t'])

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap', 'tab_n']
enc, dim = encode_rows(splits, RICH)
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)

va_preds, te_preds = [], []
for seed in range(5):
    t0 = time.time()
    m = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        P, N = sample_lists(pairs_users, rng, 4)
        for i in range(0, len(P), 8192):
            infonce_step(m, Xtr[P[i:i + 8192]], Xtr[N[i:i + 8192].reshape(-1)],
                         len(P[i:i + 8192]), 4)
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4:
                break
    m.V, m.W, m.b = best_state
    np.savez_compressed(os.path.join(OUT, f'fm_seed{seed}.npz'),
                        V=m.V, W=m.W, b=m.b)
    pv, pt = m.predict(Xva), m.predict(Xte)
    va_preds.append((pv - pv.mean()) / pv.std())
    te_preds.append((pt - pt.mean()) / pt.std())
    print(f"seed {seed}: single test {evaluate(ute, yte, pt)['primary']:.4f} "
          f"({time.time()-t0:.0f}s)")

ens_va = np.mean(va_preds, 0); ens_te = np.mean(te_preds, 0)
np.savez_compressed(os.path.join(OUT, 'ensemble_predictions.npz'),
                    valid=ens_va, test=ens_te)
rv = evaluate(uva, yva, ens_va); rt = evaluate(ute, yte, ens_te)
print("\n=== FROZEN FINAL MODEL ===")
print(f"valid : GAUC {rv['GAUC']:.4f} | nDCG@5 {rv['nDCG@5']:.4f} | primary {rv['primary']:.4f}")
print(f"test  : GAUC {rt['GAUC']:.4f} | nDCG@5 {rt['nDCG@5']:.4f} | primary {rt['primary']:.4f}")
print(f"published baseline 0.5946 -> delta {rt['primary'] - 0.5946:+.4f}")
