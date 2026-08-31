"""P-series: match the training objective to the ranking metric (organizers' #1
suggested direction, previously untested).

P1  BPR pairwise    : within-user (pos, neg) pairs, loss = -log sigmoid(s_pos - s_neg)
P2  BPR + pointwise : interleave pairwise and pointwise batches each epoch

Rationale: primary = mean(GAUC, nDCG@5) only ever compares scores within one
user. Pointwise logloss spends model capacity calibrating scores across users,
which the metric cannot see. Pairwise loss trains on exactly the comparison the
metric grades.

Look-ahead hygiene: pairs are sampled from TRAIN rows only; encode() vocab is
built from train only (kit behaviour, unchanged); evaluate.py untouched.

Edge cases handled explicitly:
- users whose train rows are all-positive or all-negative yield no pairs; they
  are counted and reported. Under pure BPR (P1) their user embedding stays at
  init; P2 covers them via the pointwise component.
- pair sampling is reseeded per epoch from the run seed (reproducible).
- the FM bias term cancels in the pairwise difference and gets no update from
  pairwise steps (correct, not a bug).
"""
import argparse, time, collections
import numpy as np
from data import load, encode
from evaluate import evaluate
from baseline import FM, sigmoid


def build_pair_index(utr, ytr):
    """Per-user positive/negative train row indices, plus edge-case counts."""
    by_user = collections.defaultdict(lambda: ([], []))
    for i, (u, y) in enumerate(zip(utr, ytr)):
        by_user[u][0 if y > 0.5 else 1].append(i)
    pairs_users, all_pos, all_neg = [], 0, 0
    for u, (pos, neg) in by_user.items():
        if pos and neg:
            pairs_users.append((np.array(pos, dtype=np.int64),
                                np.array(neg, dtype=np.int64)))
        elif pos:
            all_pos += 1
        else:
            all_neg += 1
    return pairs_users, all_pos, all_neg


def sample_pairs(pairs_users, rng, neg_per_pos):
    """One epoch worth of (pos_idx, neg_idx) arrays, shuffled."""
    P, N = [], []
    for pos, neg in pairs_users:
        p = np.repeat(pos, neg_per_pos)
        n = neg[rng.integers(0, len(neg), size=len(p))]
        P.append(p); N.append(n)
    P = np.concatenate(P); N = np.concatenate(N)
    order = rng.permutation(len(P))
    return P[order], N[order]


def pairwise_step(m, Xp, Xn):
    """BPR step on the FM: d = z_pos - z_neg, loss = -log sigmoid(d)."""
    B = len(Xp)
    zp, Ep, Sp = m.logits(Xp)
    zn, En, Sn = m.logits(Xn)
    e = ((sigmoid(zp - zn) - 1.0) / B).astype(np.float32)   # dL/dd, batch-avg
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, Xp, e[:, None])
    np.add.at(gW, Xn, -e[:, None])
    np.add.at(gV, Xp, e[:, None, None] * (Sp[:, None, :] - Ep))
    np.add.at(gV, Xn, -e[:, None, None] * (Sn[:, None, :] - En))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for Pm, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        Pm -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    # bias cancels in z_pos - z_neg: no b update from pairwise steps


def run(splits, mode, k=16, lr=0.001, epochs=40, bs=8192, patience=4,
        neg_per_pos=4, seed=0, verbose=False):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, n_allpos, n_allneg = build_pair_index(utr, ytr)
    if verbose:
        print(f"    pairable users {len(pairs_users)} | all-pos {n_allpos} | all-neg {n_allneg}")

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        P, N = sample_pairs(pairs_users, rng, neg_per_pos)
        if mode == 'bpr':
            for i in range(0, len(P), bs):
                pairwise_step(m, Xtr[P[i:i + bs]], Xtr[N[i:i + bs]])
        elif mode == 'blend':
            idx = rng.permutation(len(ytr))
            pw_batches = [(P[i:i + bs], N[i:i + bs]) for i in range(0, len(P), bs)]
            pt_batches = [idx[i:i + bs] for i in range(0, len(idx), bs)]
            nb = max(len(pw_batches), len(pt_batches))
            for i in range(nb):
                if i < len(pt_batches):
                    j = pt_batches[i]
                    m.step(Xtr[j], ytr[j])
                if i < len(pw_batches):
                    p, n = pw_batches[i]
                    pairwise_step(m, Xtr[p], Xtr[n])
        else:
            raise ValueError(mode)
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
    from baseline import run_fm
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--neg_per_pos', type=int, default=4)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()})

    variants = [
        ("baseline (pointwise logloss)", lambda s: run_fm(splits, seed=s, verbose=False)),
        ("P1 BPR pairwise",              lambda s: run(splits, 'bpr',   neg_per_pos=a.neg_per_pos, seed=s)),
        ("P2 BPR + pointwise blend",     lambda s: run(splits, 'blend', neg_per_pos=a.neg_per_pos, seed=s)),
    ]
    print(f"\n{a.seeds} seeds each. Published FM test primary = 0.5946 (std 0.0008)\n")
    print(f"{'variant':<32} {'valid':>10}   {'test primary':>20}")
    results = {}
    for name, fn in variants:
        vs, ts = [], []
        t0 = time.time()
        for s in range(a.seeds):
            r = fn(s)
            vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
        vm, tm, tsd = np.mean(vs), np.mean(ts), np.std(ts)
        results[name] = (vm, tm, tsd)
        print(f"{name:<32} {vm:>10.4f}   {tm:>12.4f} ± {tsd:.4f}   ({time.time()-t0:.0f}s)")

    base = results["baseline (pointwise logloss)"][1]
    print(f"\n--- delta vs our own baseline run ({base:.4f}) ---")
    for name, (vm, tm, tsd) in results.items():
        d = tm - base
        flag = "SIGNIFICANT" if abs(d) > 0.002 else ("noise" if abs(d) < 0.0016 else "marginal")
        print(f"{name:<32} {d:+.4f}   {flag}")
