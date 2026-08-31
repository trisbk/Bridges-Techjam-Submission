""" Run 27: Hybrid FM with Content History

Run 27 added the interest-vector idea from Run 22 to the stronger and more stable 
FM-rich model.

The FM now compares each candidate video with the user's recent watched-video history, 
allowing it to capture content-based interests.

The random-exposure dataset was not used because it overlaps with the validation/test 
period and would cause data leakage.
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM, sigmoid
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
VIDX = 1


def hist_arrays(rws):
    Hm = np.zeros((len(rws), L), dtype=np.int32)
    Wt = np.zeros((len(rws), L), dtype=np.float32)
    for n, x in enumerate(rws):
        hv = x['hvids']; c = len(hv)
        for j, v in enumerate(hv):
            Hm[n, j] = vvocab.get(v, vunk) + voffset
        if c:
            Wt[n, :c] = 1.0 / c
    return Hm, Wt


H_ = {name: hist_arrays(rws) for name, rws in splits.items()}
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
Htr, Wtr = H_['train']; Hva, Wva = H_['valid']; Hte, Wte = H_['test']
pairs_users, _, _ = build_pair_index(utr, ytr)


class HybridFM(FM):
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        super().__init__(dim, k=k, lr=lr, l2=l2, seed=seed)
        self.beta = np.float32(1.0)
        self.mB = 0.0; self.vB = 0.0

    def logits_h(self, X, Hm, Wt):
        z, E, S = self.logits(X)
        histE = self.V[Hm]                              # (B,L,k)
        P = np.einsum('blk,bl->bk', histE, Wt)          # mean watched emb
        cand = E[:, VIDX, :]
        dot = np.einsum('bk,bk->b', cand, P) / np.sqrt(self.V.shape[1])
        return z + self.beta * dot, E, S, histE, P, cand, dot

    def predict_h(self, X, Hm, Wt, bs=100_000):
        out = []
        for i in range(0, len(X), bs):
            out.append(self.logits_h(X[i:i + bs], Hm[i:i + bs], Wt[i:i + bs])[0])
        return np.concatenate(out)


def hybrid_step(m, Xp, Hp, Wp, Xn, Hn, Wn, B, K):
    k = m.V.shape[1]; rk = np.sqrt(k)
    zp, Ep, Sp, HEp, Pp, Cp, Dp = m.logits_h(Xp, Hp, Wp)
    zn, En, Sn, HEn, Pn, Cn, Dn = m.logits_h(Xn, Hn, Wn)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    ep, en = e[:, 0], e[:, 1:].reshape(B * K)

    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W); gB = 0.0
    for (X_, E_, S_, HE_, P_, C_, D_, e_, H_m, W_t) in (
            (Xp, Ep, Sp, HEp, Pp, Cp, Dp, ep, Hp, Wp),
            (Xn, En, Sn, HEn, Pn, Cn, Dn, en, Hn, Wn)):
        np.add.at(gW, X_, e_[:, None])
        dE = e_[:, None, None] * (S_[:, None, :] - E_)          # FM part
        dE[:, VIDX, :] += (m.beta / rk) * e_[:, None] * P_       # dot: cand side
        np.add.at(gV, X_, dE)
        dhist = (m.beta / rk) * (e_[:, None] * C_)[:, None, :] * W_t[:, :, None]
        np.add.at(gV, H_m, dhist)
        gB += float((e_ * D_).sum())

    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for Pm, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        Pm -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.mB = b1 * m.mB + (1 - b1) * gB
    m.vB = b2 * m.vB + (1 - b2) * gB * gB
    m.beta -= m.lr * (m.mB / (1 - b1 ** m.t)) / (np.sqrt(m.vB / (1 - b2 ** m.t)) + eps)


def train_fn(seed, K=4, bs=8192, patience=4):
    m = HybridFM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        P, N = sample_lists(pairs_users, rng, K)
        for i in range(0, len(P), bs):
            p = P[i:i + bs]; n = N[i:i + bs].reshape(-1)
            hybrid_step(m, Xtr[p], Htr[p], Wtr[p], Xtr[n], Htr[n], Wtr[n], len(p), K)
        va = evaluate(uva, yva, m.predict_h(Xva, Hva, Wva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b), np.float32(m.beta))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b, m.beta = best_state
    return {'valid': evaluate(uva, yva, m.predict_h(Xva, Hva, Wva)),
            'test': evaluate(ute, yte, m.predict_h(Xte, Hte, Wte))}


run_experiment(
    name='R27 hybrid FM (rich fields + candidate-history dot term)',
    hypothesis='Adding beta * <candidate emb, mean watched-history emb> gives '
               'the stable FM family the content-recency signal that made the '
               'interest MLP the best single model on base fields.',
    rationale='Interest signal worked (R22) but the MLP family is unstable on '
              'rich fields; grafting the dot term onto the FM keeps the '
              'winning family and adds the new signal. beta learned, init 1.',
    train_fn=train_fn, seeds=3,
    config={'L': L, 'fields': 'rich', 'beta_init': 1.0})