"""Pillar 2: testing exposure-debiased negative sampling.

Logged impressions reflect choices made by the previous recommender, which can
introduce exposure bias. This experiment reduces that bias by sampling
over-exposed videos less often as negatives, using training-only exposure
counts.

Different correction strengths are evaluated on both the standard and
random-exposure test sets to measure the trade-off between logged-ranking
performance and less-biased exposure performance.

These are diagnostic experiments only; the final submission is unchanged.
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import infonce_step
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

SEQ_KEYS = ['prev1', 'hist10', 'hist_n', 'auth_hist', 'hist30', 'tag_hist', 'gap']
RICH = BASE + SEQ_KEYS
LAMBDAS = (0.0, 0.5, 1.0)
SEEDS = 3


def build_all():
    splits = load_sequenced()
    vid2author, vid2tag = {}, {}
    with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
            vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
    random_rows = []
    with open(os.path.join(DATA, 'log_random_4_22_to_5_08_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            random_rows.append({
                'date': int(r['date']), 't': int(r['time_ms']),
                'user_id': r['user_id'], 'video_id': r['video_id'],
                'author_id': vid2author.get(r['video_id'], 'UNK'),
                'tab': r['tab'], 'duration': float(r['duration_ms']),
                'y': 1 if r['long_view'] != '0' else 0})
    all_rows = [x for rws in splits.values() for x in rws] + random_rows
    all_rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    hist = {}
    for x in all_rows:
        u = x['user_id']
        if u not in hist:
            hist[u] = {'last30': collections.deque(maxlen=30),
                       'tag': collections.Counter(), 'last_t': None,
                       'last10': collections.deque(maxlen=10), 'n': 0,
                       'prev1': None, 'auth': collections.Counter()}
        h = hist[u]
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
        h['prev1'] = x['y']; h['last10'].append(x['y']); h['n'] += 1
        h['auth'][x['author_id']] += x['y']
        h['last30'].append(x['y']); h['tag'][tg] += x['y']
        h['last_t'] = (x['date'], x['t'])
    rand_test = [x for x in random_rows if x['date'] >= 20220429]
    return splits, rand_test


def sample_lists_weighted(pairs_users, neg_w, rng, K):
    for (pos, neg), cw in zip(pairs_users, neg_w):
        p = np.repeat(pos, K)
        draws = rng.random(len(p))
        idx = np.searchsorted(cw, draws)
        N.append(neg[idx].reshape(len(pos), K))
        P.append(pos)
    P = np.concatenate(P)
    N = np.concatenate(N, axis=0)
    order = rng.permutation(len(P))
    return P[order], N[order]


if __name__ == '__main__':
    print("building features for standard + random-exposure sets ...")
    splits, rand_test = build_all()

    enc, dim = encode_rows(splits, RICH)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    enc_r, dim_r = encode_rows({'train': splits['train'], 'valid': splits['valid'],
                                'test': rand_test}, RICH)
    assert dim_r == dim
    Xrt, yrt, urt = enc_r['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    exp_cnt = collections.Counter(x['video_id'] for x in splits['train'])
    vid_of_row = [x['video_id'] for x in splits['train']]

    print(f"{'lambda':>7} {'std test':>9} {'unbiased':>9}")
    results = []
    for lam in LAMBDAS:
        neg_w = []
        for pos, neg in pairs_users:
            w = np.array([1.0 / (exp_cnt[vid_of_row[i]] ** lam) for i in neg])
            cw = np.cumsum(w / w.sum())
            cw[-1] = 1.0
            neg_w.append(cw)
        te_p, rt_p = [], []
        for seed in range(SEEDS):
            m = FM(dim, k=16, lr=0.001, seed=seed)
            rng = np.random.default_rng(seed)
            best, best_state, bad = -1, None, 0
            for ep in range(1, 41):
                P, N = sample_lists_weighted(pairs_users, neg_w, rng, 4)
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
            pt = m.predict(Xte); pr = m.predict(Xrt)
            te_p.append((pt - pt.mean()) / pt.std())
            rt_p.append((pr - pr.mean()) / pr.std())
        std = evaluate(ute, yte, np.mean(te_p, 0))['primary']
        unb = evaluate(urt, yrt, np.mean(rt_p, 0))['primary']
        results.append((lam, std, unb))
        print(f"{lam:>7.1f} {std:>9.5f} {unb:>9.5f}")

    print("\n=== BIAS FRONTIER (3-seed committees) ===")
    l0 = results[0]
    for lam, std, unb in results:
        print(f"lambda {lam:.1f}: std {std:.5f} ({std-l0[1]:+.5f})  "
              f"unbiased {unb:.5f} ({unb-l0[2]:+.5f})")