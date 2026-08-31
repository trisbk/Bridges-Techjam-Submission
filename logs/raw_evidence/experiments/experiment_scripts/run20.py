""" Run 20: Expanding Sequence Features

Run 20 built on the improvements from using sequence-based features.

R20a – Sequence Ensemble: Combined five MLP and five FM models using the new sequence 
features.

R20b – Richer Sequences: Added longer viewing history, tag-specific history, and time 
since the user's previous impression to better capture user behavior.
"""
import time, csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from mlp import MLPRank, infonce_mlp_step
from harness import _append, BASELINE, run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

BANKED = 0.5986
R18 = 0.6016

print("loading + sequencing ...")
splits = load_sequenced()

# ---- extra causal features for R20b ----
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

FIELDS_A = BASE + SEQ
FIELDS_B = BASE + SEQ + ['hist30', 'tag_hist', 'gap']


def train_one(enc, dim, F, seed, kind):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
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
    return ((pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std())


# ---- R20a: mixed committee on seq features ----
enc_a, dim_a = encode_rows(splits, FIELDS_A)
va_p, te_p = [], []
for kind in ('mlp', 'fm'):
    for s in range(5):
        pv, pt = train_one(enc_a, dim_a, len(FIELDS_A), s, kind)
        va_p.append(pv); te_p.append(pt)
        print(f"{kind}-seq seed {s} done")
uva = enc_a['valid'][2]; yva = enc_a['valid'][1]
ute = enc_a['test'][2]; yte = enc_a['test'][1]
v = evaluate(uva, yva, np.mean(va_p, 0))['primary']
t = evaluate(ute, yte, np.mean(te_p, 0))['primary']
mark = '✅ BETTER than banked' if t > BANKED else '❌ not better'
_append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
         'name': 'R20a mixed committee on seq features',
         'valid_mean': round(float(v), 5), 'test_mean': round(float(t), 5),
         'test_std': None, 'd_baseline': round(float(t - BASELINE), 5),
         'verdict': 'WIN' if t - BASELINE > 0.002 else 'NOISE',
         'note': '5 MLP-seq + 5 FM-seq'})
print(f"R20a committee: valid {v:.5f}  test {t:.4f}  vs banked {t-BANKED:+.4f} {mark}")

# ---- R20b: richer sequence set, through harness ----
enc_b, dim_b = encode_rows(splits, FIELDS_B)
Xtr, ytr, utr = enc_b['train']; Xva2, yva2, uva2 = enc_b['valid']; Xte2, yte2, ute2 = enc_b['test']
pairs_users_b, _, _ = build_pair_index(utr, ytr)

def train_fn(seed):
    m = MLPRank(dim_b, len(FIELDS_B), k=16, H=64, lr=0.0003, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        P, N = sample_lists(pairs_users_b, rng, 4)
        for i in range(0, len(P), 8192):
            infonce_mlp_step(m, Xtr[P[i:i + 8192]], Xtr[N[i:i + 8192].reshape(-1)],
                             len(P[i:i + 8192]), 4)
        va = evaluate(uva2, yva2, m.predict(Xva2))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state(best_state)
    return {'valid': evaluate(uva2, yva2, m.predict(Xva2)),
            'test': evaluate(ute2, yte2, m.predict(Xte2))}

run_experiment(
    name='R20b richer sequences (+hist30, tag_hist, gap)',
    hypothesis='Longer window, causal per-tag taste, and session recency add '
               'signal beyond the four R18 features.',
    rationale='The sequence family just proved itself (+0.0030 over banked); '
              'the immediate question is how much more the family holds.',
    train_fn=train_fn, seeds=3,
    config={'fields': FIELDS_B, 'H': 64, 'lr': 0.0003})