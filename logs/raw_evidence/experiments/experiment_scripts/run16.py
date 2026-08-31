""" Run 16: Cross Features and Temperature Tuning

Run 16 tested two final improvements.

R16a – User × Tab: Added a user_tab feature to capture how each user's preferences may 
differ across different surfaces.

R16b/c – InfoNCE Temperature: Tested τ=0.5 to focus more on difficult ranking comparisons 
and τ=2.0 as a softer comparison.
"""
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from mlp import MLPRank
from harness import run_experiment

print("loading ...")
splits = load('./KuaiRand-Pure/data')


def encode_cross(splits):
    """Kit fields + user_tab cross. Vocab from train; UNK for unseen."""
    tr = splits['train']
    edges = np.quantile(np.array([x[5] for x in tr]), np.linspace(0, 1, 11)[1:-1])

    def raw(x):
        return [x[1], x[2], x[3], x[4],
                str(int(np.searchsorted(edges, x[5]))),
                x[1] + '|' + x[4]]                     # user_tab cross

    fields = FIELDS + ['user_tab']
    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)
    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(dims)), len(fields)


def infonce_mlp_step_tau(m, Xp, Xn_flat, B, K, tau):
    zp, fp, pp, hp = m.forward(Xp)
    zn, fn, pn, hn = m.forward(Xn_flat)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1) / tau
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / (B * tau)).astype(np.float32)
    gp = m.backward(Xp, fp, pp, hp, e[:, 0])
    gn = m.backward(Xn_flat, fn, pn, hn, e[:, 1:].reshape(B * K))
    m.adam({p: gp[p] + gn[p] for p in gp})


def make_fn(enc, dim, F, tau):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    def train_fn(seed, K=4, k=16, H=64, lr=0.0003, epochs=40, bs=8192, patience=4):
        m = MLPRank(dim, F, k=k, H=H, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs]
                infonce_mlp_step_tau(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K, tau)
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


enc_c, dim_c, F_c = encode_cross(splits)
run_experiment(
    name='R16a MLP + user_tab cross field',
    hypothesis='Per-surface taste: tab long_view rates span 0.4%-49%, and a '
               'user may rank differently per surface; a user x tab embedding '
               'captures what flat user + tab fields conflate.',
    rationale='Tab stats measured first (73% tab 1; rates wildly uneven). '
              'Cross field is the cheapest per-surface conditioning; separate '
              'per-tab models would break cross-tab score comparability '
              'within a user.',
    train_fn=make_fn(enc_c, dim_c, F_c, 1.0), seeds=3,
    config={'fields': '+user_tab', 'H': 64, 'lr': 0.0003})

enc_b, dim_b = encode(splits)
for tau in (0.5, 2.0):
    run_experiment(
        name=f'R16{"b" if tau == 0.5 else "c"} InfoNCE temperature tau={tau}',
        hypothesis='tau<1 sharpens the softmax onto the hardest comparisons — '
                   'a cheap proxy for nDCG top-slot emphasis. tau>1 control.',
        rationale='Direct nDCG-weighted loss risks fighting the GAUC half '
                  '(Run 1 lesson); temperature is the mildest top-slot lever.',
        train_fn=make_fn(enc_b, dim_b, len(FIELDS), tau), seeds=3,
        config={'tau': tau, 'H': 64, 'lr': 0.0003})