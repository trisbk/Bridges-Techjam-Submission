""" Testing Two Metric-Based Improvements

Two ideas were tested against the official FM baseline:

B1 – User-Weight Alignment: Reweighted training so each user contributes more equally, 
preventing users with many impressions from dominating the model.

B2 – Dense Target: Used the continuous play time/duration ratio instead of the binary 
long_view target to provide more detailed engagement information.

The original evaluate.py and data.py were left unchanged to keep the official baseline 
reproducible.
"""
import argparse, csv, os, time, collections
import numpy as np
from data import SPLITS, LABEL, encode, FIELDS
from evaluate import evaluate
from baseline import FM, sigmoid


def load_with_ratio(data_dir):
    """data.load(), plus the play_time/duration ratio per row (same row order)."""
    vid2author = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
    rows, ratios = [], []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                dur = float(r['duration_ms'])
                rows.append((int(r['date']), r['user_id'], r['video_id'],
                             vid2author.get(r['video_id'], 'UNK'), r['tab'],
                             dur, 1 if r[LABEL] != '0' else 0))
                ratios.append(float(r['play_time_ms']) / dur if dur > 0 else 0.0)
    splits, ratio_by_split = {}, {}
    for name, (lo, hi) in SPLITS.items():
        keep = [i for i, x in enumerate(rows) if lo <= x[0] <= hi]
        splits[name] = [rows[i] for i in keep]
        ratio_by_split[name] = np.array([ratios[i] for i in keep], dtype=np.float32)
    return splits, ratio_by_split


def weighted_step(m, X, y, w):
    """FM.step with a per-row weight. Same maths otherwise."""
    B = len(y)
    z, E, S = m.logits(X)
    g = ((sigmoid(z) - y) * w / B).astype(np.float32)
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g.sum()
    return 0.0


def run(splits, ratios, use_weights, target, k=16, lr=0.001, epochs=40,
        bs=8192, patience=4, seed=0, verbose=False):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']

    # ---- B2: soft target from play ratio ----
    if target == 'binary':
        ttr = ytr
    elif target == 'ratio':
        ttr = np.clip(ratios['train'], 0.0, 1.0).astype(np.float32)
    elif target == 'blend':
        ttr = (0.5 * ytr + 0.5 * np.clip(ratios['train'], 0.0, 1.0)).astype(np.float32)
    else:
        raise ValueError(target)

    # ---- B1: per-row weight = 1 / impressions(user), mean-normalised ----
    if use_weights:
        cnt = collections.Counter(utr)
        w = np.array([1.0 / cnt[u] for u in utr], dtype=np.float32)
        w /= w.mean()
    else:
        w = np.ones(len(ytr), dtype=np.float32)

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ttr))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            weighted_step(m, Xtr[j], ttr[j], w[j])
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"    epoch {ep:2d} valid primary {va['primary']:.4f}")
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--seeds', type=int, default=3)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits, ratios = load_with_ratio(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()})

    variants = [
        ("baseline      (binary, unweighted)", False, 'binary'),
        ("B1            (binary, user-weighted)", True,  'binary'),
        ("B2            (ratio,  unweighted)", False, 'ratio'),
        ("B2-blend      (blend,  unweighted)", False, 'blend'),
        ("B1+B2         (ratio,  user-weighted)", True,  'ratio'),
        ("B1+B2-blend   (blend,  user-weighted)", True,  'blend'),
    ]
    print(f"\n{a.seeds} seeds each. Official published FM test primary = 0.5946 (std 0.0008)\n")
    print(f"{'variant':<40} {'valid':>18}   {'test primary':>22}")
    results = {}
    for name, uw, tgt in variants:
        vs, ts = [], []
        t0 = time.time()
        for s in range(a.seeds):
            r = run(splits, ratios, uw, tgt, seed=s)
            vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
        vm, tm = np.mean(vs), np.mean(ts)
        tsd = np.std(ts)
        results[name] = (vm, tm, tsd)
        print(f"{name:<40} {vm:>10.4f}        {tm:>10.4f} ± {tsd:.4f}   ({time.time()-t0:.0f}s)")

    base = results["baseline      (binary, unweighted)"][1]
    print(f"\n--- delta vs our own baseline run ({base:.4f}) ---")
    for name, (vm, tm, tsd) in results.items():
        d = tm - base
        flag = "SIGNIFICANT" if abs(d) > 0.002 else ("noise" if abs(d) < 0.0016 else "marginal")
        print(f"{name:<40} {d:+.4f}   {flag}")