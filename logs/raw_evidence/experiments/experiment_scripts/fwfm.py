""" Run 9: Field-Weighted FM

Run 9 tested FwFM using the listwise objective. FwFM extends the standard FM by learning 
a separate weight for each pair of feature fields.

The model starts identically to the FM, so any performance improvement can be attributed 
to learning these field-pair weights.
""" 
import time
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from baseline import FM, sigmoid
from pairwise import build_pair_index
from listwise import sample_lists
from harness import run_experiment


class FwFM(FM):
    def __init__(self, dim, F, k=16, lr=0.001, l2=1e-6, seed=0):
        super().__init__(dim, k=k, lr=lr, l2=l2, seed=seed)
        self.F = F
        self.R = np.ones((F, F), dtype=np.float32)
        np.fill_diagonal(self.R, 0.0)
        self.mR = np.zeros_like(self.R); self.vR = np.zeros_like(self.R)

    def logits(self, X):
        E = self.V[X]                                    # (B,F,k)
        D = np.einsum('bik,bjk->bij', E, E)              # (B,F,F)
        inter = 0.5 * np.einsum('ij,bij->b', self.R, D)
        return self.b + self.W[X].sum(1) + inter, E, D

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])


def infonce_fwfm_step(m, Xp, Xn_flat, B, K):
    zp, Ep, Dp = m.logits(Xp)
    zn, En, Dn = m.logits(Xn_flat)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    ep, en = e[:, 0], e[:, 1:].reshape(B * K)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    REp = np.einsum('ij,bjk->bik', m.R, Ep)              # dz/dE
    REn = np.einsum('ij,bjk->bik', m.R, En)
    np.add.at(gW, Xp, ep[:, None])
    np.add.at(gW, Xn_flat, en[:, None])
    np.add.at(gV, Xp, ep[:, None, None] * REp)
    np.add.at(gV, Xn_flat, en[:, None, None] * REn)
    gR = 0.5 * (np.einsum('b,bij->ij', ep, Dp) + np.einsum('b,bij->ij', en, Dn))
    np.fill_diagonal(gR, 0.0)

    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for Pm, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW),
                         (m.R, gR, m.mR, m.vR)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        Pm -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    np.fill_diagonal(m.R, 0.0)


if __name__ == '__main__':
    print("loading ...")
    splits = load('./KuaiRand-Pure/data')
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    F = len(FIELDS)

    def train_fn(seed, K=4, k=16, lr=0.001, epochs=40, bs=8192, patience=4):
        m = FwFM(dim, F, k=k, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs]
                infonce_fwfm_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K)
            va = evaluate(uva, yva, m.predict(Xva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b), m.R.copy())
            else:
                bad += 1
                if bad >= patience:
                    break
        m.V, m.W, m.b, m.R = best_state
        return {'valid': evaluate(uva, yva, m.predict(Xva)),
                'test': evaluate(ute, yte, m.predict(Xte))}

    rec = run_experiment(
        name='R9 FwFM listwise K=4',
        hypothesis='Field-pair weights let user x video interactions dominate '
                   'low-signal pairs (e.g. tab x dur_bucket), which uniform FM '
                   'interactions cannot express.',
        rationale='R init = ones makes FwFM identical to FM at step zero, so '
                  'any departure is data-driven; 10 extra params, negligible '
                  'overfit risk. Feature/capacity levers are exhausted (Runs '
                  '4-8), leaving model expressiveness as the open frontier.',
        train_fn=train_fn, seeds=3,
        config={'K': 4, 'k': 16, 'lr': 0.001, 'fields': FIELDS})

    # peek at the learned field-pair weights of the last seed for the log
    m = FwFM(dim, F)