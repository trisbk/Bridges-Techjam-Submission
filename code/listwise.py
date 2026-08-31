"""Run 4: two attacks on the BPR gain cap found in Run 3.

L-series  listwise/InfoNCE : softmax over (1 positive + K sampled negatives)
                             per user. Unlike BPR's uniform pair gradient,
                             softmax upweights hard negatives automatically.
C-series  capacity         : k=32 embeddings. C-control = pointwise at k=32,
                             so any BPR@k=32 gain is attributed to the
                             objective x capacity interaction, not capacity.

Hygiene identical to pairwise.py: negatives sampled within-user from TRAIN
rows only; all-pos/all-neg users yield no lists (skipped for the listwise
loss); softmax rows sum to zero gradient for the bias (no bias update).
"""
import time
import numpy as np
from data import load, encode
from evaluate import evaluate
from baseline import FM, run_fm
from pairwise import build_pair_index, run as run_bpr


def sample_lists(pairs_users, rng, K):
    """Per positive occurrence: (pos_idx, K neg idxs). Shuffled."""
    P, N = [], []
    for pos, neg in pairs_users:
        p = pos
        n = neg[rng.integers(0, len(neg), size=(len(p), K))]
        P.append(p); N.append(n)
    P = np.concatenate(P); N = np.concatenate(N, axis=0)
    order = rng.permutation(len(P))
    return P[order], N[order]


def infonce_step(m, Xp, Xn_flat, B, K):
    """Softmax CE over [pos, K negs]. Xn_flat is (B*K, F)."""
    zp, Ep, Sp = m.logits(Xp)
    zn, En, Sn = m.logits(Xn_flat)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    en = e[:, 1:].reshape(B * K)
    np.add.at(gW, Xp, e[:, 0][:, None])
    np.add.at(gW, Xn_flat, en[:, None])
    np.add.at(gV, Xp, e[:, 0][:, None, None] * (Sp[:, None, :] - Ep))
    np.add.at(gV, Xn_flat, en[:, None, None] * (Sn[:, None, :] - En))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for Pm, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        Pm -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    # softmax rows sum to zero gradient for the bias: no b update


def run_listwise(splits, K=4, k=16, lr=0.001, epochs=40, bs=8192, patience=4,
                 seed=0, verbose=False):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        P, N = sample_lists(pairs_users, rng, K)
        for i in range(0, len(P), bs):
            p = P[i:i + bs]
            n = N[i:i + bs]
            infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K)
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
    print("loading ./KuaiRand-Pure/data ...")
    splits = load('./KuaiRand-Pure/data')

    configs = [
        ("L1 InfoNCE K=4  k=16", lambda s: run_listwise(splits, K=4, k=16, seed=s)),
        ("L2 InfoNCE K=8  k=16", lambda s: run_listwise(splits, K=8, k=16, seed=s)),
        ("C1 BPR k=32 (lr .001 npp 4)", lambda s: run_bpr(splits, 'bpr', k=32, seed=s)),
        ("C0 pointwise k=32 (control)", lambda s: run_fm(splits, k=32, seed=s, verbose=False)),
    ]
    SEEDS = 3
    BASE = 0.5950
    print(f"\n{SEEDS} seeds each. Same-session baseline = {BASE} | best so far P1 = 0.5967 (+0.0017)\n")
    print(f"{'config':<30} {'valid':>10}   {'test primary':>20}   {'delta':>8}")
    for name, fn in configs:
        vs, ts = [], []
        t0 = time.time()
        for s in range(SEEDS):
            r = fn(s)
            vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
        vm, tm, tsd = np.mean(vs), np.mean(ts), np.std(ts)
        d = tm - BASE
        flag = "SIGNIFICANT+" if d > 0.002 else ("SIGNIFICANT-" if d < -0.002 else "within noise band")
        print(f"{name:<30} {vm:>10.4f}   {tm:>12.4f} ± {tsd:.4f}   {d:+.4f}  {flag}  ({time.time()-t0:.0f}s)")
