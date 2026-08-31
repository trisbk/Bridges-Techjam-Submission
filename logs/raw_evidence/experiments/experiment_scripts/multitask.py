""" Run 19: Multi-Task Learning

Run 19 added is_click and is_like as auxiliary training targets alongside the main 
long_view target.

These tasks share the same embeddings and hidden layer, helping the model learn a 
stronger representation of user engagement. The auxiliary tasks were weighted at 0.3 
and used training data only, while the main evaluation remained unchanged.
""" 
import csv, os, collections
import numpy as np
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from mlp import MLPRank
from harness import run_experiment
from sequences import SPLITS, DATA, encode_rows, BASE, SEQ


def load_seq_multilabel():
    vid2author = {}
    with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(DATA, f)) as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    'date': int(r['date']), 't': int(r['time_ms']),
                    'user_id': r['user_id'], 'video_id': r['video_id'],
                    'author_id': vid2author.get(r['video_id'], 'UNK'),
                    'tab': r['tab'], 'duration': float(r['duration_ms']),
                    'y': 1 if r['long_view'] != '0' else 0,
                    'yc': 1.0 if r['is_click'] != '0' else 0.0,
                    'yl': 1.0 if r['is_like'] != '0' else 0.0,
                })
    rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    hist = {}
    for x in rows:
        u = x['user_id']
        if u not in hist:
            hist[u] = {'last10': collections.deque(maxlen=10), 'n': 0,
                       'prev1': None, 'auth': collections.Counter()}
        h = hist[u]
        x['prev1'] = 'none' if h['prev1'] is None else str(h['prev1'])
        x['hist10'] = 'none' if not h['last10'] else str(sum(h['last10']))
        n = h['n']
        x['hist_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
                       else '11-30' if n <= 30 else '31-100' if n <= 100 else '100+')
        a = h['auth'][x['author_id']]
        x['auth_hist'] = str(a) if a < 3 else '3+'
        h['prev1'] = x['y']; h['last10'].append(x['y']); h['n'] += 1
        h['auth'][x['author_id']] += x['y']
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x['date'] <= hi]
    return out


class MultiMLP(MLPRank):
    def __init__(self, dim, F, k=16, H=64, lr=0.0003, l2=1e-6, seed=0):
        super().__init__(dim, F, k=k, H=H, lr=lr, l2=l2, seed=seed)
        rng = np.random.default_rng(seed + 999)
        self.wc = (rng.normal(0, 1, H) * np.sqrt(2.0 / H)).astype(np.float32)
        self.wl = (rng.normal(0, 1, H) * np.sqrt(2.0 / H)).astype(np.float32)
        for p in ('wc', 'wl'):
            self.params.append(p)
            setattr(self, 'm_' + p, np.zeros_like(getattr(self, p)))
            setattr(self, 'v_' + p, np.zeros_like(getattr(self, p)))

    def aux_step(self, X, yc, yl, lam):
        z, flat, pre, h = self.forward(X)
        B = len(X)
        zc = h @ self.wc; zl = h @ self.wl
        sc = 1.0 / (1.0 + np.exp(-np.clip(zc, -30, 30)))
        sl = 1.0 / (1.0 + np.exp(-np.clip(zl, -30, 30)))
        ec = (lam * (sc - yc) / B).astype(np.float32)
        el = (lam * (sl - yl) / B).astype(np.float32)
        gwc = h.T @ ec; gwl = h.T @ el
        dh = np.outer(ec, self.wc) + np.outer(el, self.wl)
        dh[pre <= 0] = 0.0
        gW1 = flat.T @ dh; gb1 = dh.sum(0)
        dE = (dh @ self.W1.T).reshape(B, self.F, self.k)
        gV = np.zeros_like(self.V)
        np.add.at(gV, X, dE)
        grads = {'V': gV, 'W1': gW1, 'b1': gb1,
                 'w2': np.zeros_like(self.w2), 'wc': gwc, 'wl': gwl}
        self.adam(grads)


def infonce_multi_step(m, Xp, Xn_flat, B, K):
    zp, fp, pp, hp = m.forward(Xp)
    zn, fn, pn, hn = m.forward(Xn_flat)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    gp = m.backward(Xp, fp, pp, hp, e[:, 0])
    gn = m.backward(Xn_flat, fn, pn, hn, e[:, 1:].reshape(B * K))
    grads = {p: gp[p] + gn[p] for p in gp}
    grads['wc'] = np.zeros_like(m.wc); grads['wl'] = np.zeros_like(m.wl)
    m.adam(grads)


if __name__ == '__main__':
    print("loading + sequencing + multilabels ...")
    splits = load_seq_multilabel()
    fields = BASE + SEQ
    enc, dim = encode_rows(splits, fields)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    yc = np.array([x['yc'] for x in splits['train']], dtype=np.float32)
    yl = np.array([x['yl'] for x in splits['train']], dtype=np.float32)
    pairs_users, _, _ = build_pair_index(utr, ytr)
    F = len(fields)

    def train_fn(seed, K=4, lam=0.3, epochs=40, bs=8192, patience=4):
        m = MultiMLP(dim, F, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            P, N = sample_lists(pairs_users, rng, K)
            aux_idx = rng.permutation(len(ytr))
            na = 0
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs]
                infonce_multi_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K)
                if i // bs % 2 == 0 and na + bs <= len(aux_idx):
                    j = aux_idx[na:na + bs]; na += bs
                    m.aux_step(Xtr[j], yc[j], yl[j], lam)
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

    run_experiment(
        name='R19 multi-task aux heads (click+like) on R18 recipe',
        hypothesis='Jointly predicting click/like regularizes the shared '
                   'embedding + hidden representation, improving long_view '
                   'ranking beyond what its own label can teach.',
        rationale='Aux labels are train-rows only; heads share the trunk, '
                  'evaluation unchanged. lambda=0.3, aux on alternate batches.',
        train_fn=train_fn, seeds=3,
        config={'lam': 0.3, 'fields': fields})