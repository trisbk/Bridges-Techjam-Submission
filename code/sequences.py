"""Run 18: causal sequence features (unparked by owner, 29 Aug night).

Features per impression, from STRICTLY EARLIER events only (rows sorted by
(date, time_ms) per user; a row's own label is excluded; nothing from later):

  prev1     : label of the user's previous impression (0/1/none)
  hist10    : rolling long_view count over last <=10 impressions (0..10/none)
  hist_n    : log-bucketed count of prior impressions
  auth_hist : user's prior long_view count with THIS author (0/1/2/3+)

Legality: at serving time the platform observes past behavior — standard
sequential-recsys practice. Contrast with failed Runs 6-7: those were static,
whole-window, self-inclusive; these are causal, self-exclusive, and dynamic.
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from mlp import MLPRank, infonce_mlp_step
from harness import run_experiment

DATA = './KuaiRand-Pure/data'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}


def load_sequenced():
    vid2author = {}
    with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(DATA, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    'date': int(r['date']), 't': int(r['time_ms']),
                    'user_id': r['user_id'], 'video_id': r['video_id'],
                    'author_id': vid2author.get(r['video_id'], 'UNK'),
                    'tab': r['tab'], 'duration': float(r['duration_ms']),
                    'y': 1 if r['long_view'] != '0' else 0,
                })
    rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    hist = {}
    for x in rows:
        u = x['user_id']
        if u not in hist:
            hist[u] = {'last10': collections.deque(maxlen=10), 'n': 0,
                       'prev1': None, 'auth': collections.Counter()}
        h = hist[u]
        x['prev1'] = 'none' if h['prev1'] is None else str(h['prev1'])
        x['hist10'] = 'none' if not h['last10'] else str(sum(h['last10']))
        n = h['n']
        x['hist_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
                       else '11-30' if n <= 30 else '31-100' if n <= 100 else '100+')
        a = h['auth'][x['author_id']]
        x['auth_hist'] = str(a) if a < 3 else '3+'
        # update AFTER featurising — own label never in own features
        h['prev1'] = x['y']; h['last10'].append(x['y']); h['n'] += 1
        h['auth'][x['author_id']] += x['y']
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x['date'] <= hi]
    return out


def encode_rows(splits, fields):
    tr = splits['train']
    edges = np.quantile(np.array([x['duration'] for x in tr]),
                        np.linspace(0, 1, 11)[1:-1])
    for rws in splits.values():
        for x in rws:
            x['dur_bucket'] = str(int(np.searchsorted(edges, x['duration'])))
    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, f in enumerate(fields):
            if x[f] not in vocabs[i]:
                vocabs[i][x[f]] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, f in enumerate(fields):
                X[n, i] = vocabs[i].get(x[f], unk[i]) + offsets[i]
            y[n] = x['y']
            users.append(x['user_id'])
        enc[name] = (X, y, users)
    return enc, int(sum(dims))


BASE = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
SEQ = ['prev1', 'hist10', 'hist_n', 'auth_hist']

if __name__ == '__main__':
    print("loading + sequencing ...")
    splits = load_sequenced()
    enc, dim = encode_rows(splits, BASE + SEQ)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    F = len(BASE + SEQ)

    def train_fn(seed, K=4, k=16, H=64, lr=0.0003, epochs=40, bs=8192, patience=4):
        m = MLPRank(dim, F, k=k, H=H, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs]
                infonce_mlp_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K)
            va = evaluate(uva, yva, m.predict(Xva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = m.state()
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state(best_state)
        return {'valid': evaluate(uva, yva, m.predict(Xva)),
                'test': evaluate(ute, yte, m.predict(Xte))}

    run_experiment(
        name='R18 MLP + causal sequence features',
        hypothesis='Recency dynamics (last-impression outcome, rolling rate, '
                   'causal per-author history) carry signal static profiles '
                   'cannot: a user mid-binge ranks differently than mid-skip.',
        rationale='Production-recsys #1 lever, unparked by owner. Causal and '
                  'self-exclusive by construction, unlike Runs 6-7. Best model '
                  'class (MLP k16 H64 lr 3e-4) as base.',
        train_fn=train_fn, seeds=3,
        config={'fields': BASE + SEQ, 'H': 64, 'lr': 0.0003})
