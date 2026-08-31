"""Iteration 1: align FM training more closely with the ranking metric.

GAUC and nDCG@5 evaluate ordering within each user, while the baseline uses
pointwise BCE over all impressions. This experiment adds a BPR pairwise term
so the model directly learns to rank a user's positive impressions above
their negatives.

Objective: BCE(pointwise impressions) + λ * BPR(within-user positive/negative pairs)

The BCE term keeps the stable signal from the full training set, while BPR
adds pressure in the same within-user direction measured by the ranking
metrics. Users are sampled uniformly for the pairwise term.

All pairs come from TRAIN only, the official five input fields are unchanged,
and the model still scores individual impressions normally at inference.
Pure BPR is retained as an ablation for comparison.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from data import load, encode
from baseline import FM, sigmoid
from evaluate import evaluate
from harness import run_experiment

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'KuaiRand-Pure', 'data')


class PairFM(FM):

    def step_pair(self, Xp, Xn):
        B = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        d = zp - zn
        c = (sigmoid(-d) / B).astype(np.float32)      # -dL/dd
        gp, gn = -c, c
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, Xp, gp[:, None])
        np.add.at(gW, Xn, gn[:, None])
        np.add.at(gV, Xp, gp[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, gn[:, None, None] * (Sn[:, None, :] - En))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        return float(np.mean(np.logaddexp(0.0, -np.clip(d, -30, 30))))


def _ragged(groups):
    counts = np.array([len(g) for g in groups], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    flat = np.concatenate(groups).astype(np.int64) if len(groups) else np.zeros(0, np.int64)
    return flat, offs, counts


def build_pairs_index(y, users):
    pos, neg = {}, {}
    for i, (u, yi) in enumerate(zip(users, y)):
        (pos if yi > 0.5 else neg).setdefault(u, []).append(i)
    keep = [u for u in pos if u in neg]
    return (_ragged([np.asarray(pos[u]) for u in keep]),
            _ragged([np.asarray(neg[u]) for u in keep]), len(keep))


def run_pair_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4,
                seed=0, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    (pf, po, pc), (nf, no, nc), U = build_pairs_index(ytr, utr)
    if verbose:
        print(f"  pairable users {U} / {len(set(utr))}")
    m = PairFM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(len(ytr) / bs))     # same gradient-step budget as baseline
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        losses = []
        for _ in range(n_steps):
            u = rng.integers(0, U, bs)                      # users uniformly
            ip = pf[po[u] + (rng.random(bs) * pc[u]).astype(np.int64)]
            iN = nf[no[u] + (rng.random(bs) * nc[u]).astype(np.int64)]
            losses.append(m.step_pair(Xtr[ip], Xtr[iN]))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | bpr {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}



class HybridFM(PairFM):

    def step_hybrid(self, Xb, yb, Xp, Xn, lam):
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)

        Bb = len(yb)
        zb, Eb, Sb = self.logits(Xb)
        gb = ((sigmoid(zb) - yb) / Bb).astype(np.float32)
        np.add.at(gW, Xb, gb[:, None])
        np.add.at(gV, Xb, gb[:, None, None] * (Sb[:, None, :] - Eb))
        self.b -= self.lr * gb.sum()

        Bp = len(Xp)
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn)
        d = zp - zn
        c = (lam * sigmoid(-d) / Bp).astype(np.float32)
        np.add.at(gW, Xp, -c[:, None])
        np.add.at(gW, Xn, c[:, None])
        np.add.at(gV, Xp, -c[:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn, c[:, None, None] * (Sn[:, None, :] - En))

        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        pw = float(-np.mean(yb * np.log(sigmoid(zb) + 1e-9)
                            + (1 - yb) * np.log(1 - sigmoid(zb) + 1e-9)))
        return pw + lam * float(np.mean(np.logaddexp(0.0, -np.clip(d, -30, 30))))


def run_hybrid_fm(splits, lam=0.3, k=16, lr=0.001, epochs=40, bs=8192,
                  patience=4, seed=0, verbose=True):
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    (pf, po, pc), (nf, no, nc), U = build_pairs_index(ytr, utr)
    if verbose:
        print(f"  pairable users {U} / {len(set(utr))}")
    m = HybridFM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        losses = []
        for i in range(0, len(idx), bs):
            sl = idx[i:i + bs]
            u = rng.integers(0, U, len(sl))                 # users uniformly
            ip = pf[po[u] + (rng.random(len(sl)) * pc[u]).astype(np.int64)]
            iN = nf[no[u] + (rng.random(len(sl)) * nc[u]).astype(np.int64)]
            losses.append(m.step_hybrid(Xtr[sl], ytr[sl], Xtr[ip], Xtr[iN], lam))
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='graded', choices=['graded', 'lam_scan'])
    ap.add_argument('--lam', type=float, default=0.3)
    a = ap.parse_args()
    splits = load(DATA)

    if a.mode == 'lam_scan':
        for lam in (0.1, 0.3, 1.0, 3.0):
            r = run_hybrid_fm(splits, lam=lam, seed=0, verbose=False)
            print(f"lam {lam:<5} valid {r['valid']['primary']:.5f} "
                  f"(test {r['test']['primary']:.5f})", flush=True)
        raise SystemExit

    run_experiment(
        name='exp01_within_user_pairwise_aux_loss',
        hypothesis=f'Adding a within-user pairwise (BPR) auxiliary term to the '
                   f'FM\'s pointwise BCE objective, weight lam={a.lam}, raises '
                   f'the primary metric above the official baseline, because '
                   f'the metric reads only within-user ordering while BCE alone '
                   f'optimises pooled, cross-user, calibrated probabilities.',
        rationale='GAUC and nDCG@5 are both computed inside each user\'s own '
                  'impression list (test median 5 impressions/user), so score '
                  'mass that is constant within a user is invisible to them, '
                  'and pooled BCE weights a user by impression count while the '
                  'metric weights users far more evenly. A within-user pairwise '
                  'term applies gradient pressure exactly along the direction '
                  'the metric reads. Keeping BCE alongside it matters: the '
                  'pure-BPR ablation (this file\'s run_pair_fm) lost on '
                  'validation at every lr in {3e-4, 1e-3, 3e-3} (best 0.59938 '
                  'vs baseline 0.60144) because the pairwise signal alone is '
                  'high-variance and overfits within ~2 epochs, so the dense '
                  'pointwise term is kept as the stabiliser and lam controls '
                  'the trade-off. Only the objective changes: same FM, same 5 '
                  'official fields, same k/lr/l2/batch/epochs/patience and the '
                  'same pointwise pass over all train rows.',
        train_fn=lambda s: run_hybrid_fm(splits, lam=a.lam, seed=s, verbose=False),
        seeds=3,
        config={'model': 'FM', 'k': 16, 'lr': 0.001, 'l2': 1e-6, 'epochs': 40,
                'bs': 8192, 'patience': 4, 'fields': 'official 5',
                'loss': 'BCE + lam * within-user BPR', 'lam': a.lam,
                'pair_sampling': 'user ~ Uniform, then pos ~ U(user pos), neg ~ U(user neg)',
                'lam_selected_on': 'validation, seed 0'})