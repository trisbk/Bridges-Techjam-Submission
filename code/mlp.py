"""Run 10: MLP head over field embeddings, under the listwise objective.

First nonlinear model in the series. Embeddings (F*k=80) -> hidden ReLU
layer (H=64) -> scalar score. The FM can only express second-order
multiplicative interactions; the MLP can carve arbitrary interaction shapes.
Higher ceiling, higher variance risk — needs its own lr search (two lrs
tested, each 3 seeds, each through the harness).
"""
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from harness import run_experiment


class MLPRank:
    def __init__(self, dim, F, k=16, H=64, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W1 = (rng.normal(0, 1, (F * k, H)) *
                   np.sqrt(2.0 / (F * k))).astype(np.float32)
        self.b1 = np.zeros(H, dtype=np.float32)
        self.w2 = (rng.normal(0, 1, H) * np.sqrt(2.0 / H)).astype(np.float32)
        self.b2 = np.float32(0.0)
        self.F, self.k, self.H, self.lr, self.l2 = F, k, H, lr, l2
        self.params = ['V', 'W1', 'b1', 'w2']
        for p in self.params:
            arr = getattr(self, p)
            setattr(self, 'm_' + p, np.zeros_like(arr))
            setattr(self, 'v_' + p, np.zeros_like(arr))
        self.t = 0

    def forward(self, X):
        E = self.V[X]                                   # (B,F,k)
        flat = E.reshape(len(X), self.F * self.k)
        pre = flat @ self.W1 + self.b1
        h = np.maximum(pre, 0.0)
        z = h @ self.w2 + self.b2
        return z, flat, pre, h

    def predict(self, X, bs=200_000):
        return np.concatenate([self.forward(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])

    def backward(self, X, flat, pre, h, e):
        """Accumulate grads for scores z given dL/dz = e."""
        gw2 = h.T @ e
        dh = np.outer(e, self.w2); dh[pre <= 0] = 0.0
        gW1 = flat.T @ dh
        gb1 = dh.sum(0)
        dflat = dh @ self.W1.T
        dE = dflat.reshape(len(X), self.F, self.k)
        gV = np.zeros_like(self.V)
        np.add.at(gV, X, dE)
        return {'V': gV, 'W1': gW1, 'b1': gb1, 'w2': gw2}

    def adam(self, grads):
        self.t += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            P = getattr(self, p)
            G = grads[p] + (self.l2 * P if p in ('V', 'W1', 'w2') else 0.0)
            M = getattr(self, 'm_' + p); Vv = getattr(self, 'v_' + p)
            M *= b1_; M += (1 - b1_) * G
            Vv *= b2_; Vv += (1 - b2_) * (G * G)
            P -= self.lr * (M / (1 - b1_ ** self.t)) / (np.sqrt(Vv / (1 - b2_ ** self.t)) + eps)

    def state(self):
        return tuple(getattr(self, p).copy() for p in self.params) + (np.float32(self.b2),)

    def load_state(self, st):
        for p, arr in zip(self.params, st[:-1]):
            setattr(self, p, arr)
        self.b2 = st[-1]


def infonce_mlp_step(m, Xp, Xn_flat, B, K):
    zp, fp, pp, hp = m.forward(Xp)
    zn, fn, pn, hn = m.forward(Xn_flat)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    gp = m.backward(Xp, fp, pp, hp, e[:, 0])
    gn = m.backward(Xn_flat, fn, pn, hn, e[:, 1:].reshape(B * K))
    m.adam({p: gp[p] + gn[p] for p in gp})


if __name__ == '__main__':
    print("loading ...")
    splits = load('./KuaiRand-Pure/data')
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    F = len(FIELDS)

    def make_train_fn(lr):
        def train_fn(seed, K=4, k=16, epochs=40, bs=8192, patience=4):
            m = MLPRank(dim, F, k=k, lr=lr, seed=seed)
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
        return train_fn

    for lr in (0.001, 0.0003):
        run_experiment(
            name=f'R10 MLP head H=64 lr={lr}',
            hypothesis='A nonlinear head over the embeddings can express '
                       'interaction shapes the second-order FM cannot.',
            rationale='All second-order levers are exhausted (Runs 3-9: '
                      'hyperparams, capacity, features, field weights). '
                      'Model class is the only remaining frontier. Two lrs '
                      'because nonlinear heads are lr-sensitive.',
            train_fn=make_train_fn(lr), seeds=3,
            config={'H': 64, 'k': 16, 'lr': lr, 'K': 4})
