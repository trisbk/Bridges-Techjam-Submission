""" Run 26: Attention-Based User History

Run 26 tested better ways of using a user's recent watched-video history on the rich 
feature set.

R26a – Attention Pooling: Weighted the last 10 watched videos based on how relevant 
each one was to the candidate video.

R26b – Mean Pooling: Used the same history but averaged the video embeddings equally, 
allowing us to measure the specific benefit of attention.

Both used the MLP architecture and were tested across three seeds.
"""
import time, csv, os, collections
import numpy as np
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from harness import run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

L = 10

print("loading + sequencing ...")
splits = load_sequenced()
vid2tag = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
watched = {}
for x in rows_flat:
    u = x['user_id']
    if u not in hist:
        hist[u] = {'last30': collections.deque(maxlen=30),
                   'tag': collections.Counter(), 'last_t': None}
        watched[u] = collections.deque(maxlen=L)
    h = hist[u]
    x['hist30'] = 'none' if not h['last30'] else str(int(10 * sum(h['last30']) / len(h['last30'])))
    tg = vid2tag.get(x['video_id'], 'UNK')
    tc = h['tag'][tg]
    x['tag_hist'] = str(tc) if tc < 3 else '3+'
    if h['last_t'] is None:
        x['gap'] = 'none'
    else:
        d = (x['date'] - h['last_t'][0]) * 86400_000 + (x['t'] - h['last_t'][1])
        x['gap'] = ('<1m' if d < 60_000 else '<1h' if d < 3_600_000
                    else '<1d' if d < 86_400_000 else '1d+')
    x['hvids'] = list(watched[u])
    h['last30'].append(x['y']); h['tag'][tg] += x['y']; h['last_t'] = (x['date'], x['t'])
    if x['y'] == 1:
        watched[u].append(x['video_id'])

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']
enc, dim = encode_rows(splits, RICH)
F = len(RICH)

vocabs = [dict() for _ in RICH]
for x in splits['train']:
    for i, f in enumerate(RICH):
        if x[f] not in vocabs[i]:
            vocabs[i][x[f]] = len(vocabs[i])
dims_ = [len(v) + 1 for v in vocabs]
offsets = np.cumsum([0] + dims_[:-1]).astype(np.int32)
vvocab = vocabs[1]; vunk = len(vvocab); voffset = int(offsets[1])
VIDX = 1  # candidate video's field position


def hist_arrays(rws):
    Hm = np.zeros((len(rws), L), dtype=np.int32)
    Mk = np.zeros((len(rws), L), dtype=np.float32)
    for n, x in enumerate(rws):
        hv = x['hvids']
        for j, v in enumerate(hv):
            Hm[n, j] = vvocab.get(v, vunk) + voffset
            Mk[n, j] = 1.0
    return Hm, Mk


H_ = {name: hist_arrays(rws) for name, rws in splits.items()}
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
Htr, Mtr = H_['train']; Hva, Mva = H_['valid']; Hte, Mte = H_['test']
pairs_users, _, _ = build_pair_index(utr, ytr)


class AttnInterestMLP:
    def __init__(self, dim, F, k=16, H=64, lr=0.0003, l2=1e-6, seed=0, attn=True):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        D = F * k + k
        self.W1 = (rng.normal(0, 1, (D, H)) * np.sqrt(2.0 / D)).astype(np.float32)
        self.b1 = np.zeros(H, dtype=np.float32)
        self.w2 = (rng.normal(0, 1, H) * np.sqrt(2.0 / H)).astype(np.float32)
        self.F, self.k, self.lr, self.l2, self.attn = F, k, lr, l2, attn
        self.params = ['V', 'W1', 'b1', 'w2']
        for p in self.params:
            arr = getattr(self, p)
            setattr(self, 'm_' + p, np.zeros_like(arr))
            setattr(self, 'v_' + p, np.zeros_like(arr))
        self.t = 0

    def forward(self, X, Hm, Mk):
        B = len(X)
        E = self.V[X]                                    # (B,F,k)
        hist = self.V[Hm]                                # (B,L,k)
        if self.attn:
            cand = E[:, VIDX, :]                         # (B,k)
            s = np.einsum('bk,blk->bl', cand, hist) / np.sqrt(self.k)
            s = np.where(Mk > 0, s, -1e9)
            s -= s.max(axis=1, keepdims=True)
            a = np.exp(s) * (Mk > 0)
            asum = a.sum(axis=1, keepdims=True)
            a = np.where(asum > 0, a / np.maximum(asum, 1e-9), 0.0).astype(np.float32)
        else:
            c = Mk.sum(axis=1, keepdims=True)
            a = (Mk / np.maximum(c, 1.0)).astype(np.float32)
        P = np.einsum('blk,bl->bk', hist, a)
        flat = np.concatenate([E.reshape(B, self.F * self.k), P], axis=1)
        pre = flat @ self.W1 + self.b1
        h = np.maximum(pre, 0.0)
        z = h @ self.w2
        return z, (E, hist, a, flat, pre, h)

    def predict(self, X, Hm, Mk, bs=100_000):
        return np.concatenate([self.forward(X[i:i + bs], Hm[i:i + bs], Mk[i:i + bs])[0]
                               for i in range(0, len(X), bs)])

    def backward(self, X, Hm, Mk, cache, e):
        E, hist, a, flat, pre, h = cache
        B = len(X)
        gw2 = h.T @ e
        dh = np.outer(e, self.w2); dh[pre <= 0] = 0.0
        gW1 = flat.T @ dh; gb1 = dh.sum(0)
        dflat = dh @ self.W1.T
        dE = dflat[:, :self.F * self.k].reshape(B, self.F, self.k).copy()
        dP = dflat[:, self.F * self.k:]                  # (B,k)
        gV = np.zeros_like(self.V)
        # history grads through pooling weights
        dhist = a[:, :, None] * dP[:, None, :]           # (B,L,k)
        if self.attn:
            da = np.einsum('bk,blk->bl', dP, hist)       # (B,L)
            ds = a * (da - (a * da).sum(axis=1, keepdims=True))
            cand = E[:, VIDX, :]
            dhist += ds[:, :, None] * cand[:, None, :] / np.sqrt(self.k)
            dcand = np.einsum('bl,blk->bk', ds, hist) / np.sqrt(self.k)
            dE[:, VIDX, :] += dcand
        np.add.at(gV, X, dE)
        np.add.at(gV, Hm, dhist)
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


def step(m, Xp, Hp, Mp, Xn, Hn, Mn, B, K):
    zp, cp = m.forward(Xp, Hp, Mp)
    zn, cn = m.forward(Xn, Hn, Mn)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    gp = m.backward(Xp, Hp, Mp, cp, e[:, 0])
    gn = m.backward(Xn, Hn, Mn, cn, e[:, 1:].reshape(B * K))
    m.adam({p: gp[p] + gn[p] for p in gp})


def make_fn(attn):
    def train_fn(seed, K=4, bs=8192, patience=4):
        m = AttnInterestMLP(dim, F, k=16, H=64, lr=0.0003, seed=seed, attn=attn)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 41):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs].reshape(-1)
                step(m, Xtr[p], Htr[p], Mtr[p], Xtr[n], Htr[n], Mtr[n], len(p), K)
            va = evaluate(uva, yva, m.predict(Xva, Hva, Mva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = m.state()
            else:
                bad += 1
                if bad >= patience:
                    break
        m.load_state(best_state)
        return {'valid': evaluate(uva, yva, m.predict(Xva, Hva, Mva)),
                'test': evaluate(ute, yte, m.predict(Xte, Hte, Mte))}
    return train_fn


run_experiment(
    name='R26a attention-pooled interest on rich fields',
    hypothesis='Weighting watched history by relevance to the candidate '
               '(dot-product attention) extracts per-candidate taste the '
               'uniform mean pool blurs.',
    rationale='DIN-proper. Built on the winning rich feature set; attention '
              'is the canonical upgrade over R22 mean pooling.',
    train_fn=make_fn(True), seeds=3,
    config={'L': L, 'fields': 'rich', 'attn': True})

run_experiment(
    name='R26b mean-pooled interest on rich fields (control)',
    hypothesis='Isolates whether interest vectors help at all on rich fields '
               'and what attention specifically adds beyond the mean pool.',
    rationale='One-variable-at-a-time.',
    train_fn=make_fn(False), seeds=3,
    config={'L': L, 'fields': 'rich', 'attn': False})