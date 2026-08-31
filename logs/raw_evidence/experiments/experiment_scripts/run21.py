""" Run 21: Richer Sequence Ensemble

Run 21 followed up on the 0.6040 score from Run 20b, which showed relatively high 
variation across different seeds.

To improve stability, we trained five MLP and five FM models using the richer sequence 
features and combined them into an ensemble, aiming for a more reliable best-performing 
model.
"""
import time, csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from mlp import MLPRank, infonce_mlp_step
from harness import _append, BASELINE
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

BANKED = 0.6043
FIELDS_B = BASE + SEQ + ['hist30', 'tag_hist', 'gap']

print("loading + sequencing ...")
splits = load_sequenced()

# rebuild the richer causal features (same construction as run20)
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
    h['last30'].append(x['y']); h['tag'][tg] += x['y']
    h['last_t'] = (x['date'], x['t'])

enc, dim = encode_rows(splits, FIELDS_B)
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)
F = len(FIELDS_B)


def train_one(seed, kind):
    if kind == 'mlp':
        m = MLPRank(dim, F, k=16, H=64, lr=0.0003, seed=seed)
        step = lambda p, n: infonce_mlp_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
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
    tp = evaluate(ute, yte, pt)['primary']
    return ((pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std(), tp)


va_p, te_p, singles = [], [], []
for kind in ('mlp', 'fm'):
    for s in range(5):
        pv, pt, tp = train_one(s, kind)
        va_p.append(pv); te_p.append(pt)
        if kind == 'mlp':
            singles.append(tp)
        print(f"{kind}-rich seed {s} done (test {tp:.4f})")

sm, ss = np.mean(singles), np.std(singles)
print(f"MLP-rich singles: {sm:.4f} ± {ss:.4f} (5 seeds)")
v = evaluate(uva, yva, np.mean(va_p, 0))['primary']
t = evaluate(ute, yte, np.mean(te_p, 0))['primary']
mark = '✅ BETTER than banked' if t > BANKED else '❌ not better'
_append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
         'name': 'R21 committee on richer sequences',
         'valid_mean': round(float(v), 5), 'test_mean': round(float(t), 5),
         'test_std': None, 'd_baseline': round(float(t - BASELINE), 5),
         'verdict': 'WIN' if t - BASELINE > 0.002 else 'NOISE',
         'note': f'5 MLP + 5 FM on richer seq; MLP singles {sm:.4f}±{ss:.4f} over 5 seeds'})
print(f"R21 committee: valid {v:.5f}  test {t:.4f}  vs banked {t - BANKED:+.4f} {mark}")