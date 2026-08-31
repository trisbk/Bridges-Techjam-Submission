""" Run 14: New Model Architectures

Run 14 tested two new model architectures using the existing features and listwise 
objective.

R14a – Two-Layer MLP: Extended the previous single-layer MLP with two hidden layers 
(64 → 32) to capture more complex feature relationships.

R14b – FFM: Tested field-aware embeddings, allowing feature interactions to be modeled 
differently depending on their fields. This is more expressive than the FwFM approach 
from Run 9.

Even if performance is similar, these models could add diversity to the mixed ensemble 
approach from Run 11.
"""
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from harness import run_experiment


class MLP2Rank:
    def __init__(self, dim, F, k=16, H1=64, H2=32, lr=0.0003, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W1 = (rng.normal(0, 1, (F * k, H1)) * np.sqrt(2.0 / (F * k))).astype(np.float32)
        self.b1 = np.zeros(H1, dtype=np.float32)
        self.W2 = (rng.normal(0, 1, (H1, H2)) * np.sqrt(2.0 / H1)).astype(np.float32)
        self.b2 = np.zeros(H2, dtype=np.float32)
        self.w3 = (rng.normal(0, 1, H2) * np.sqrt(2.0 / H2)).astype(np.float32)
        self.F, self.k, self.lr, self.l2 = F, k, lr, l2
        self.params = ['V', 'W1', 'b1', 'W2', 'b2', 'w3']
        for p in self.params:
            arr = getattr(self, p)
            setattr(self, 'm_' + p, np.zeros_like(arr))
            setattr(self, 'v_' + p, np.zeros_like(arr))
        self.t = 0

    def forward(self, X):
        E = self.V[X]
        flat = E.reshape(len(X), self.F * self.k)
        p1 = flat @ self.W1 + self.b1; h1 = np.maximum(p1, 0.0)
        p2 = h1 @ self.W2 + self.b2;   h2 = np.maximum(p2, 0.0)
        z = h2 @ self.w3
        return z, (flat, p1, h1, p2, h2)

    def predict(self, X, bs=200_000):
        return np.concatenate([self.forward(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])

    def backward(self, X, cache, e):
        flat, p1, h1, p2, h2 = cache
        gw3 = h2.T @ e
        dh2 = np.outer(e, self.w3); dh2[p2 <= 0] = 0.0
        gW2 = h1.T @ dh2; gb2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T; dh1[p1 <= 0] = 0.0
        gW1 = flat.T @ dh1; gb1 = dh1.sum(0)
        dE = (dh1 @ self.W1.T).reshape(len(X), self.F, self.k)
        gV = np.zeros_like(self.V)
        np.add.at(gV, X, dE)
        return {'V': gV, 'W1': gW1, 'b1': gb1, 'W2': gW2, 'b2': gb2, 'w3': gw3}

    def adam(self, grads):
        self.t += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            P = getattr(self, p)
            G = grads[p] + (self.l2 * P if p not in ('b1', 'b2') else 0.0)
            M = getattr(self, 'm_' + p); Vv = getattr(self, 'v_' + p)
            M *= b1_; M += (1 - b1_) * G
            Vv *= b2_; Vv += (1 - b2_) * (G * G)
            P -= self.lr * (M / (1 - b1_ ** self.t)) / (np.sqrt(Vv / (1 - b2_ ** self.t)) + eps)

    def state(self):
        return tuple(getattr(self, p).copy() for p in self.params)

    def load_state(self, st):
        for p, arr in zip(self.params, st):
            setattr(self, p, arr)


class FFM:
    def __init__(self, dim, F, k=8, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, F, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.F, self.k, self.lr, self.l2 = F, k, lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0
        self.pairs = [(i, j) for i in range(F) for j in range(i + 1, F)]

    def forward(self, X):
        E = self.V[X]                       # (B, F_feat, F_field, k)
        z = self.W[X].sum(1)
        for i, j in self.pairs:
            z = z + np.einsum('bk,bk->b', E[:, i, j], E[:, j, i])
        return z, E

    def predict(self, X, bs=100_000):
        return np.concatenate([self.forward(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])

    def backward(self, X, E, e):
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, e[:, None])
        dE = np.zeros_like(E)
        for i, j in self.pairs:
            dE[:, i, j] += e[:, None] * E[:, j, i]
            dE[:, j, i] += e[:, None] * E[:, i, j]
        np.add.at(gV, X, dE)
        return gV, gW

    def adam(self, gV, gW):
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def state(self):
        return (self.V.copy(), self.W.copy())

    def load_state(self, st):
        self.V, self.W = st


def infonce_generic(m, Xp, Xn_flat, B, K, kind):
    if kind == 'mlp2':
        zp, cp = m.forward(Xp); zn, cn = m.forward(Xn_flat)
    else:
        zp, Ep = m.forward(Xp); zn, En = m.forward(Xn_flat)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    ep, en = e[:, 0], e[:, 1:].reshape(B * K)
    if kind == 'mlp2':
        gp = m.backward(Xp, cp, ep); gn = m.backward(Xn_flat, cn, en)
        m.adam({p: gp[p] + gn[p] for p in gp})
    else:
        gVp, gWp = m.backward(Xp, Ep, ep)
        gVn, gWn = m.backward(Xn_flat, En, en)
        m.adam(gVp + gVn, gWp + gWn)


if __name__ == '__main__':
    print("loading ...")
    splits = load('./KuaiRand-Pure/data')
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    F = len(FIELDS)

    def make_fn(kind, **kw):
        def train_fn(seed, K=4, epochs=40, bs=8192, patience=4):
            m = (MLP2Rank(dim, F, seed=seed, **kw) if kind == 'mlp2'
                 else FFM(dim, F, seed=seed, **kw))
            rng = np.random.default_rng(seed)
            best, best_state, bad = -1, None, 0
            for ep in range(1, epochs + 1):
                P, N = sample_lists(pairs_users, rng, K)
                for i in range(0, len(P), bs):
                    p = P[i:i + bs]; n = N[i:i + bs]
                    infonce_generic(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K, kind)
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

    run_experiment(
        name='R14a two-layer MLP H=64-32',
        hypothesis='Depth lets the head compose interactions hierarchically; '
                   'one layer may be the expressiveness bottleneck.',
        rationale='Single-layer MLP is the banked best single model; depth is '
                  'the canonical next rung. Same lr (3e-4) as proven optimal.',
        train_fn=make_fn('mlp2', k=16, H1=64, H2=32, lr=0.0003), seeds=3,
        config={'k': 16, 'H1': 64, 'H2': 32, 'lr': 0.0003})

    run_experiment(
        name='R14b FFM k=8',
        hypothesis='Per-field embedding subspaces express asymmetric '
                   'interactions (user-as-seen-by-video vs by-author) that '
                   'shared embeddings conflate.',
        rationale='FwFM scalar weights failed (Run 9) but full field-aware '
                  'subspaces are strictly richer. k=8 keeps params comparable '
                  '(dim*5*8 = 2.5x baseline). Diverse class also feeds the '
                  'mixed ensemble even on a tie.',
        train_fn=make_fn('ffm', k=8, lr=0.001), seeds=3,
        config={'k': 8, 'lr': 0.001})