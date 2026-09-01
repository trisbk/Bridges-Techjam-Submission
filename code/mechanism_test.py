"""Mechanism test: why do the causal sequence features help?

This experiment tests whether the gain comes from recent behavioral timing,
general user-history patterns, or simply extra feature capacity. It compares
the full sequence model against three controls: no sequence features,
time-shuffled history, and matched random features.

The resulting score differences separate the contribution of temporal
alignment, persistent user-history signal, and added capacity. These models
are diagnostic only and do not change the frozen submission.
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA
from controls import time_shuffle, capacity_noise

SEQ_KEYS = ['prev1', 'hist10', 'hist_n', 'auth_hist', 'hist30', 'tag_hist', 'gap']
RICH = BASE + SEQ_KEYS
A_FULL = 0.61164  


def build_rich_rows():
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
        h['last30'].append(x['y']); h['tag'][tg] += x['y']
        h['last_t'] = (x['date'], x['t'])
    return splits


def split_of(row):
    if row['date'] <= 20220421:
        return 'train'
    if row['date'] <= 20220428:
        return 'valid'
    return 'test'


def train_committee(splits, fields, label):
    enc, dim = encode_rows(splits, fields)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    va_p, te_p = [], []
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
        va_p.append((pv - pv.mean()) / pv.std())
        te_p.append((pt - pt.mean()) / pt.std())
    rv = evaluate(uva, yva, np.mean(va_p, 0))
    rt = evaluate(ute, yte, np.mean(te_p, 0))
    print(f"{label:<44} valid {rv['primary']:.5f}  test {rt['primary']:.5f}")
    return rt['primary']


if __name__ == '__main__':
    print("building rich features (continuous, as shipped) ...")
    splits = build_rich_rows()
    all_rows = [x for rws in splits.values() for x in rws]

    print("\narm B: base fields only (no sequence features)")
    B = train_committee(splits, BASE, 'B base-only committee')

    print("\narm C: sequence features time-shuffled within user+split")
    shuffled = time_shuffle(all_rows, SEQ_KEYS, split_of, seed=0)
    spl_C = {'train': [r for r in shuffled if split_of(r) == 'train'],
             'valid': [r for r in shuffled if split_of(r) == 'valid'],
             'test': [r for r in shuffled if split_of(r) == 'test']}
    C = train_committee(spl_C, RICH, 'C time-shuffled committee')

    print("\narm D: sequence features replaced by matched-cardinality noise")
    noised = capacity_noise(all_rows, SEQ_KEYS, seed=0)
    spl_D = {'train': [r for r in noised if split_of(r) == 'train'],
             'valid': [r for r in noised if split_of(r) == 'valid'],
             'test': [r for r in noised if split_of(r) == 'test']}
    D = train_committee(spl_D, RICH, 'D capacity-noise committee')

    print("\n=== MECHANISM DECOMPOSITION (test primary) ===")
    print(f"A full (frozen submission)  : {A_FULL:.5f}")
    print(f"B base-only                 : {B:.5f}")
    print(f"C time-shuffled             : {C:.5f}")
    print(f"D capacity-noise            : {D:.5f}")
    total = A_FULL - B
    timing = A_FULL - C
    fingerprint = C - B
    capacity = D - B
    print(f"\ntotal sequence gain     A-B = {total:+.5f}")
    print(f"timing component        A-C = {timing:+.5f} ({timing/total:.0%} of gain)")
    print(f"fingerprint component   C-B = {fingerprint:+.5f} ({fingerprint/total:.0%} of gain)")
    print(f"pure-capacity control   D-B = {capacity:+.5f} (expected ~0)")