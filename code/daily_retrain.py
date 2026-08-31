"""Post-freeze analysis 1: retrain the full recipe under a daily-batch
feature regime, so the staleness table's deployable row is a measured
number instead of a mismatch-penalized one.

Difference from staleness_ablation.py: there, only test featurization
changed while the weights stayed trained on continuous features (a
train/serve mismatch). Here, EVERY row's features are built under the
daily regime (a row sees the user's history from completed days strictly
before its own date), and the 5-seed committee is retrained from scratch
on those features. This is the model a team running a daily feature
refresh would actually train and ship.

This analysis does not touch the designated submission; the frozen
checkpoint remains the model of record.

Run from code/:  python3 daily_retrain.py     (~8 min)
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA


def build_daily_features():
    splits = load_sequenced()   # loads rows; we recompute ALL seq features
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
                       'tag': collections.Counter(), 'last_t': None,
                       'last10': collections.deque(maxlen=10), 'n': 0,
                       'prev1': None, 'auth': collections.Counter(),
                       'pend': [], 'pend_day': None}
        h = hist[u]
        # flush days completed strictly before this row's date
        if h['pend_day'] is not None and x['date'] > h['pend_day']:
            for p in h['pend']:
                _apply(h, p, vid2tag)
            h['pend'], h['pend_day'] = [], None

        x['prev1'] = 'none' if h['prev1'] is None else str(h['prev1'])
        x['hist10'] = 'none' if not h['last10'] else str(sum(h['last10']))
        n = h['n']
        x['hist_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
                       else '11-30' if n <= 30 else '31-100' if n <= 100 else '100+')
        a = h['auth'][x['author_id']]
        x['auth_hist'] = str(a) if a < 3 else '3+'
        x['hist30'] = ('none' if not h['last30']
                       else str(int(10 * sum(h['last30']) / len(h['last30']))))
        tg = vid2tag.get(x['video_id'], 'UNK')
        tc = h['tag'][tg]
        x['tag_hist'] = str(tc) if tc < 3 else '3+'
        if h['last_t'] is None:
            x['gap'] = 'none'
        else:
            d = (x['date'] - h['last_t'][0]) * 86400_000 + (x['t'] - h['last_t'][1])
            x['gap'] = ('<1m' if d < 60_000 else '<1h' if d < 3_600_000
                        else '<1d' if d < 86_400_000 else '1d+')

        h['pend'].append(x)
        h['pend_day'] = x['date']
    return splits


def _apply(h, x, vid2tag):
    h['prev1'] = x['y']; h['last10'].append(x['y']); h['n'] += 1
    h['auth'][x['author_id']] += x['y']
    h['last30'].append(x['y'])
    h['tag'][vid2tag.get(x['video_id'], 'UNK')] += x['y']
    h['last_t'] = (x['date'], x['t'])


RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']

if __name__ == '__main__':
    print("building daily-regime features for every row ...")
    splits = build_daily_features()
    enc, dim = encode_rows(splits, RICH)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    va_preds, te_preds, singles = [], [], []
    for seed in range(5):
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
        pv, pt = m.predict(Xva), m.predict(Xte)
        va_preds.append((pv - pv.mean()) / pv.std())
        te_preds.append((pt - pt.mean()) / pt.std())
        sp = float(evaluate(ute, yte, pt)['primary'])
        singles.append(sp)
        print(f"seed {seed}: single test {sp:.4f}")

    rv = evaluate(uva, yva, np.mean(va_preds, 0))
    rt = evaluate(ute, yte, np.mean(te_preds, 0))
    print("\n=== DAILY-REGIME RETRAINED COMMITTEE ===")
    print(f"valid : GAUC {rv['GAUC']:.4f} | nDCG@5 {rv['nDCG@5']:.4f} | primary {rv['primary']:.4f}")
    print(f"test  : GAUC {rt['GAUC']:.4f} | nDCG@5 {rt['nDCG@5']:.4f} | primary {rt['primary']:.4f}")
    print(f"vs official baseline: {rt['primary'] - 0.5946:+.4f}")
    print("(compare: shipped weights served daily without retraining: 0.60828)")
