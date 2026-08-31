""" Run 13: MLP Refinements

Run 13 tested two improvements to the best-performing MLP.

R13a – Annealing: Started with pointwise training for two epochs before switching to 
listwise training for a more stable start.

R13b – Hard Negatives: Focused half of the negative samples on videos the model currently 
found difficult to distinguish, while keeping the other half randomly sampled.
"""
import numpy as np
from data import load, encode, FIELDS
from evaluate import evaluate
from pairwise import build_pair_index
from listwise import sample_lists
from mlp import MLPRank, infonce_mlp_step
from harness import run_experiment

print("loading ...")
splits = load('./KuaiRand-Pure/data')
enc, dim = encode(splits)
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)
F = len(FIELDS)


def pointwise_mlp_step(m, X, y):
    z, flat, pre, h = m.forward(X)
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    e = ((p - y) / len(y)).astype(np.float32)
    m.adam(m.backward(X, flat, pre, h, e))


def sample_hard(pairs_users, rng, K, tr_scores):
    P, N = [], []
    for pos, neg in pairs_users:
        p = pos
        ns = tr_scores[neg]
        q = np.quantile(ns, 0.75)
        hard_pool = neg[ns >= q]
        if len(hard_pool) == 0:
            hard_pool = neg
        n_uni = neg[rng.integers(0, len(neg), size=(len(p), K))]
        n_hard = hard_pool[rng.integers(0, len(hard_pool), size=(len(p), K))]
        mask = rng.random((len(p), K)) < 0.5
        N.append(np.where(mask, n_hard, n_uni))
        P.append(p)
    P = np.concatenate(P); N = np.concatenate(N, axis=0)
    order = rng.permutation(len(P))
    return P[order], N[order]


def make_fn(mode):
    def train_fn(seed, K=4, k=16, H=64, lr=0.0003, epochs=40, bs=8192, patience=4):
        m = MLPRank(dim, F, k=k, H=H, lr=lr, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, epochs + 1):
            if mode == 'anneal' and ep <= 2:
                idx = rng.permutation(len(ytr))
                for i in range(0, len(idx), bs):
                    j = idx[i:i + bs]
                    pointwise_mlp_step(m, Xtr[j], ytr[j])
            else:
                if mode == 'hardneg' and ep > 1:
                    tr_scores = m.predict(Xtr)
                    P, N = sample_hard(pairs_users, rng, K, tr_scores)
                else:
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


run_experiment(
    name='R13a MLP pointwise-anneal then listwise',
    hypothesis='A 2-epoch pointwise warm start gives the listwise phase '
               'calibrated embeddings to refine, improving its optimum.',
    rationale='Cheap; Run 2 showed mixing objectives per-batch hurts, but '
              'sequential phases may not — tests whether the harm was '
              'simultaneity or the pointwise signal itself.',
    train_fn=make_fn('anneal'), seeds=3,
    config={'warm_epochs': 2, 'H': 64, 'lr': 0.0003})

run_experiment(
    name='R13b MLP hard-negative mining',
    hypothesis='Sampling half the negatives from the top-quartile of the '
               'user\'s negatives by current score sharpens the decision '
               'boundary where ranking errors live.',
    rationale='InfoNCE already upweights hard negatives in the gradient; '
              'this tests whether SAMPLING them more often adds anything '
              'or is redundant with that weighting.',
    train_fn=make_fn('hardneg'), seeds=3,
    config={'hard_frac': 0.5, 'quantile': 0.75, 'H': 64, 'lr': 0.0003})