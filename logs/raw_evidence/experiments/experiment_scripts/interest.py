""" Run 22: User-Interest Vector

Run 22 introduced a DIN-lite user-interest vector based on the user's last 10 watched videos.

Their previous video embeddings were averaged into a taste vector, allowing the MLP to 
compare candidate videos with the user's recent viewing interests. Only past viewing 
history was used to prevent data leakage, while users with no history received a zero 
vector.
"""
import numpy as np
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from harness import run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ

L = 10  # history length


class InterestMLP:
    def __init__(self, dim, F, k=16, H=64, lr=0.0003, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        D = F * k + k
        self.W1 = (rng.normal(0, 1, (D, H)) * np.sqrt(2.0 / D)).astype(np.float32)
        self.b1 = np.zeros(H, dtype=np.float32)
        self.w2 = (rng.normal(0, 1, H) * np.sqrt(2.0 / H)).astype(np.float32)
        self.F, self.k, self.lr, self.l2 = F, k, lr, l2
        self.params = ['V', 'W1', 'b1', 'w2']
        for p in self.params:
            arr = getattr(self, p)
            setattr(self, 'm_' + p, np.zeros_like(arr))
            setattr(self, 'v_' + p, np.zeros_like(arr))
        self.t = 0

    def forward(self, X, Hm, Wt):
        E = self.V[X]                                    # (B,F,k)
        hist = self.V[Hm]                                # (B,L,k)
        P = np.einsum('blk,bl->bk', hist, Wt)            # mean over watched
        flat = np.concatenate([E.reshape(len(X), self.F * self.k), P], axis=1)
        pre = flat @ self.W1 + self.b1
        h = np.maximum(pre, 0.0)
        z = h @ self.w2
        return z, (flat, pre, h)

    def predict(self, X, Hm, Wt, bs=200_000):
        return np.concatenate([self.forward(X[i:i + bs], Hm[i:i + bs],
                                            Wt[i:i + bs])[0]
                               for i in range(0, len(X), bs)])

    def backward(self, X, Hm, Wt, cache, e):
        flat, pre, h = cache
        gw2 = h.T @ e
        dh = np.outer(e, self.w2); dh[pre <= 0] = 0.0
        gW1 = flat.T @ dh; gb1 = dh.sum(0)
        dflat = dh @ self.W1.T
        B = len(X)
        dE = dflat[:, :self.F * self.k].reshape(B, self.F, self.k)
        dP = dflat[:, self.F * self.k:]                  # (B,k)
        gV = np.zeros_like(self.V)
        np.add.at(gV, X, dE)
        np.add.at(gV, Hm, dP[:, None, :] * Wt[:, :, None])
        return {'V': gV, 'W1': gW1, 'b1': gb1, 'w2': gw2}

    def adam(self, grads):
        self.t += 1
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for p in self.params:
            Pm = getattr(self, p)
            G = grads[p] + (self.l2 * Pm if p != 'b1' else 0.0)
            M = getattr(self, 'm_' + p); Vv = getattr(self, 'v_' + p)
            M *= b1_; M += (1 - b1_) * G
            Vv *= b2_; Vv += (1 - b2_) * (G * G)
            Pm -= self.lr * (M / (1 - b1_ ** self.t)) / (np.sqrt(Vv / (1 - b2_ ** self.t)) + eps)

    def state(self):
        return tuple(getattr(self, p).copy() for p in self.params)

    def load_state(self, st):
        for p, arr in zip(self.params, st):
            setattr(self, p, arr)


def infonce_interest_step(m, Xp, Hp, Wp, Xn, Hn, Wn, B, K):
    zp, cp = m.forward(Xp, Hp, Wp)
    zn, cn = m.forward(Xn, Hn, Wn)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    gp = m.backward(Xp, Hp, Wp, cp, e[:, 0])
    gn = m.backward(Xn, Hn, Wn, cn, e[:, 1:].reshape(B * K))
    m.adam({p: gp[p] + gn[p] for p in gp})


if __name__ == '__main__':
    print("loading + sequencing + histories ...")
    splits = load_sequenced()

    # causal watched-video history per row (video ids, strictly prior)
    rows_flat = [x for rws in splits.values() for x in rws]
    rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    import collections
    watched = {}
    for x in rows_flat:
        u = x['user_id']
        if u not in watched:
            watched[u] = collections.deque(maxlen=L)
        x['hvids'] = list(watched[u])
        if x['y'] == 1:
            watched[u].append(x['video_id'])

    fields = BASE + SEQ
    enc, dim = encode_rows(splits, fields)
    F = len(fields)

    # encode history against the video field's train vocab
    tr = splits['train']
    vvocab, voffset = {}, None
    # rebuild the video vocab exactly as encode_rows did (field index 1)
    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, f in enumerate(fields):
            if x[f] not in vocabs[i]:
                vocabs[i][x[f]] = len(vocabs[i])
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    vvocab = vocabs[1]; vunk = len(vvocab); voffset = int(offsets[1])

    def hist_arrays(rws):
        Hm = np.zeros((len(rws), L), dtype=np.int32)
        Wt = np.zeros((len(rws), L), dtype=np.float32)
        for n, x in enumerate(rws):
            hv = x['hvids']
            c = len(hv)
            for j, v in enumerate(hv):
                Hm[n, j] = vvocab.get(v, vunk) + voffset
            if c:
                Wt[n, :c] = 1.0 / c
        return Hm, Wt

    H_ = {name: hist_arrays(rws) for name, rws in splits.items()}
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    Htr, Wtr = H_['train']; Hva, Wva = H_['valid']; Hte, Wte = H_['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    def train_fn(seed, K=4, epochs=40, bs=8192, patience=4):
        m = InterestMLP(dim, F, k=16, H=64, lr=0.0003, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs].reshape(-1)
                infonce_interest_step(m, Xtr[p], Htr[p], Wtr[p],
                                      Xtr[n], Htr[n], Wtr[n], len(p), K)
            va = evaluate(uva, yva, m.predict(Xva, Hva, Wva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = m.state()
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state(best_state)
        return {'valid': evaluate(uva, yva, m.predict(Xva, Hva, Wva)),
                'test': evaluate(ute, yte, m.predict(Xte, Hte, Wte))}

    run_experiment(
        name='R22 interest-vector MLP (mean-pooled watched history)',
        hypothesis='Content-based recency: pooling the embeddings of the last '
                   '10 watched videos lets the model match candidates to what '
                   'the user has actually been watching, beyond count-based '
                   'sequence features.',
        rationale='DIN-lite without attention — the mean-pooled interest '
                  'vector captures the core of the idea at a fraction of the '
                  'implementation risk. Attention refinement only if this '
                  'shows signal.',
        train_fn=train_fn, seeds=3,
        config={'L': L, 'fields': fields, 'H': 64, 'lr': 0.0003})