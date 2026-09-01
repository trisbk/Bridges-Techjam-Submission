"""Score the frozen model directly from its saved weights.

Loads the five committee models from frozen_model/, rebuilds the same causal
sequence features used during training, and reports the official validation
and test metrics. No retraining or model updates occur.
"""
import os, csv, collections
import numpy as np

def _frozen_dir():
    for c in ('frozen_model',
              os.path.join('..', 'final_output', 'frozen_model')):
        if os.path.exists(os.path.join(c, 'fm_seed0.npz')):
            return c
    raise FileNotFoundError(
        'frozen_model not found; expected ./frozen_model or '
        '../final_output/frozen_model')

from evaluate import evaluate
from baseline import FM
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

print("rebuilding features (no training) ...")
splits = load_sequenced()

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

    kt = (u, x['tab'])
    n = tab_ct.get(kt, 0)
    x['tab_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
                  else '11-30' if n <= 30 else '31-100' if n <= 100
                  else '100+')
    tab_ct[kt] = n + 1
    h['last30'].append(x['y']); h['tag'][tg] += x['y']; h['last_t'] = (x['date'], x['t'])

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap', 'tab_n']
enc, dim = encode_rows(splits, RICH)
Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

va_preds, te_preds = [], []
for seed in range(5):
    z = np.load(os.path.join(_frozen_dir(), f'fm_seed{seed}.npz'))
    m = FM(dim, k=16, seed=seed)
    assert m.V.shape == z['V'].shape, \
        f"weight/feature mismatch: {m.V.shape} vs {z['V'].shape}"
    m.V, m.W, m.b = z['V'], z['W'], np.float32(z['b'])
    pv, pt = m.predict(Xva), m.predict(Xte)
    va_preds.append((pv - pv.mean()) / pv.std())
    te_preds.append((pt - pt.mean()) / pt.std())
    print(f"loaded seed {seed}: single test "
          f"{evaluate(ute, yte, pt)['primary']:.4f}")

rv = evaluate(uva, yva, np.mean(va_preds, 0))
rt = evaluate(ute, yte, np.mean(te_preds, 0))
print("\n=== FROZEN MODEL (from saved weights) ===")
print(f"valid : GAUC {rv['GAUC']:.4f} | nDCG@5 {rv['nDCG@5']:.4f} | primary {rv['primary']:.4f}")
print(f"test  : GAUC {rt['GAUC']:.4f} | nDCG@5 {rt['nDCG@5']:.4f} | primary {rt['primary']:.4f}")
