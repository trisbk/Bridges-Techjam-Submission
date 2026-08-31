""" Run 29: Final Ensemble

Run 29 built ensembles using the improved FM-rich model, after Run 28 increased the 
single-model score from 0.6101 to 0.6113.

R29a: Combined five FM-rich models trained with lr=5e-4.

R29b: Blended these with five models trained at lr=1e-3, with the best weighting 
selected using validation performance.

The final model was selected based on validation results.
"""
import time, csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from harness import _append, BASELINE
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

BANKED = 0.6116

print("loading + sequencing ...")
splits = load_sequenced()
vid2tag = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
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
    h['last30'].append(x['y']); h['tag'][tg] += x['y']; h['last_t'] = (x['date'], x['t'])

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']
enc, dim = encode_rows(splits, RICH)
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)


def train_fm(seed, lr, patience):
    m = FM(dim, k=16, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 61):
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
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    pv, pt = m.predict(Xva), m.predict(Xte)
    return ((pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std())


lo_va, lo_te, hi_va, hi_te = [], [], [], []
for s in range(5):
    pv, pt = train_fm(s, 0.0005, 6)
    lo_va.append(pv); lo_te.append(pt)
    print(f"lr5e-4 seed {s} done")
for s in range(5):
    pv, pt = train_fm(s, 0.001, 4)
    hi_va.append(pv); hi_te.append(pt)
    print(f"lr1e-3 seed {s} done")


def log_result(name, v, t, note):
    mark = '✅ BETTER than banked' if t > BANKED else '❌ not better'
    _append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': name, 'valid_mean': round(float(v), 5),
             'test_mean': round(float(t), 5), 'test_std': None,
             'd_baseline': round(float(t - BASELINE), 5),
             'verdict': 'WIN' if t - BASELINE > 0.002 else 'NOISE', 'note': note})
    print(f"{name}: valid {v:.5f}  test {t:.4f}  vs banked {t - BANKED:+.4f} {mark}")


Lva, Lte = np.mean(lo_va, 0), np.mean(lo_te, 0)
Hva, Hte = np.mean(hi_va, 0), np.mean(hi_te, 0)

v = evaluate(uva, yva, Lva)['primary']; t = evaluate(ute, yte, Lte)['primary']
log_result('R29a lr5e-4 FM-rich committee', v, t, '5 seeds, patience 6')

best_a, best_v = 1.0, -1
for a in np.arange(0.0, 1.01, 0.05):
    vv = evaluate(uva, yva, a * Lva + (1 - a) * Hva)['primary']
    if vv > best_v:
        best_v, best_a = vv, a
tt = evaluate(ute, yte, best_a * Lte + (1 - best_a) * Hte)['primary']
log_result(f'R29b lr blend alpha={best_a:.2f}', best_v, tt,
           'alpha on validation over lr5e-4 x lr1e-3 committees')