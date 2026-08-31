""" Run 28: FM-Rich Retuning

Run 28 retuned the FM-rich model's training settings, since the original settings were 
optimized before the richer features were introduced.

The experiment retested key parameters such as learning rate, negative samples (K), 
and patience to see if the richer model could be trained more effectively.
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from harness import run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

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


def make_fn(lr, K, patience):
    def train_fn(seed, bs=8192):
        m = FM(dim, k=16, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 61):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                infonce_step(m, Xtr[P[i:i + bs]], Xtr[N[i:i + bs].reshape(-1)],
                             len(P[i:i + bs]), K)
            va = evaluate(uva, yva, m.predict(Xva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= patience:
                    break
        m.V, m.W, m.b = best_state
        return {'valid': evaluate(uva, yva, m.predict(Xva)),
                'test': evaluate(ute, yte, m.predict(Xte))}
    return train_fn


for name, lr, K, pat in (('lr5e-4 K4 pat6', 0.0005, 4, 6),
                         ('lr2e-3 K4 pat4', 0.002, 4, 4),
                         ('lr1e-3 K8 pat4', 0.001, 8, 4),
                         ('lr1e-3 K4 pat8', 0.001, 4, 8)):
    run_experiment(
        name=f'R28 FM-rich retune {name}',
        hypothesis='Recipe knobs were tuned on starved features; rich '
                   'features may prefer different lr/K/patience.',
        rationale='Premise change justifies one cheap retest of settled '
                  'knobs (same logic that legitimately reopened capacity).',
        train_fn=make_fn(lr, K, pat), seeds=3,
        config={'lr': lr, 'K': K, 'patience': pat})