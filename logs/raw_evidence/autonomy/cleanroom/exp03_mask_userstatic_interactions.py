"""Iteration 4: remove FM interactions that cannot affect within-user ranking.

GAUC and nDCG@5 only compare impressions belonging to the same user. In the
37-field model, 27 fields are constant for a given user, so interactions
between two user-static fields are also constant within that user's ranking
list. This makes 351 of the FM's 666 interaction pairs invisible to the metric.

This run masks those user-static x user-static interactions while keeping every
interaction between user features and item/context features. The goal is to
remove redundant capacity and prevent irrelevant gradients from interfering
with embeddings used for personalized ranking.

This is a metric-aware structural change rather than a new regularizer.
Everything else — data, features, objective, optimizer, hyperparameters, and
seeds — remains unchanged. An empty mask reproduces the original FM, with the
masked gradient implementation checked separately.
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from baseline import FM, sigmoid
from evaluate import evaluate
from harness import run_experiment
from exp02_content_context_fields import (load_rich, encode_rich, FIELDSETS,
                                          OFFICIAL, CONTENT, UPROFILE, ONEHOT)

WINNER = 'content+hour+uprofile+onehot'         

USER_STATIC = set(['user_id'] + UPROFILE + ONEHOT)


class MaskedFM(FM):

    def __init__(self, dim, user_static, drop_static_linear=False, **kw):
        super().__init__(dim, **kw)
        self.us = np.asarray(user_static, dtype=bool)
        self.any_us = bool(self.us.any())
        self.drop_lin = bool(drop_static_linear)

    def logits(self, X):
        E = self.V[X]                                    
        S = E.sum(1)                                   
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        if self.any_us:
            EU = E[:, self.us, :]
            SU = EU.sum(1)
            inter -= 0.5 * ((SU ** 2).sum(1) - (EU ** 2).sum((1, 2)))
        lin = self.W[X[:, ~self.us]].sum(1) if self.drop_lin else self.W[X].sum(1)
        return self.b + lin + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        P = S[:, None, :] - E                            # partner sum, all pairs
        if self.any_us:
            SI = S - E[:, self.us, :].sum(1)             # item/context side only
            P[:, self.us, :] = SI[:, None, :]
        if self.drop_lin:
            np.add.at(gW, X[:, ~self.us], g[:, None])
        else:
            np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * P)
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for Pm, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                             (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            Pm -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9)
                              + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))


def run_masked(enc, dim, user_static, drop_static_linear=False, k=16, lr=0.001,
               epochs=40, bs=8192, patience=4, seed=0, verbose=True):
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = MaskedFM(dim, user_static, drop_static_linear=drop_static_linear,
                 k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad, best_ep = -1, None, 0, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
                  for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"primary {va['primary']:.5f} | {time.time()-t0:.1f}s", flush=True)
        if va['primary'] > best + 1e-5:
            best, bad, best_ep = va['primary'], 0, ep
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    out = {'valid': evaluate(uva, yva, m.predict(Xva)),
           'test':  evaluate(ute, yte, m.predict(Xte))}
    out['best_epoch'] = best_ep
    return out


def mask_for(fields, which='uu'):
    if which == 'none':
        return np.zeros(len(fields), dtype=bool)
    return np.array([f in USER_STATIC for f in fields], dtype=bool)


def _selftest_grad():
    rng = np.random.default_rng(0)
    F, k, dim, B = 6, 4, 20, 7
    us = np.array([True, True, True, False, False, False])
    m = MaskedFM(dim, us, k=k, seed=1)
    m.V = rng.normal(0, 0.3, (dim, k)).astype(np.float32)
    m.W = rng.normal(0, 0.3, dim).astype(np.float32)
    X = rng.integers(0, dim, (B, F)).astype(np.int32)
    for r in range(B):
        X[r] = rng.permutation(dim)[:F]
    z0 = m.logits(X)[0].astype(np.float64)
    ref = np.zeros(B)
    for r in range(B):
        ref[r] = m.b + m.W[X[r]].sum()
        for i in range(F):
            for j in range(i + 1, F):
                if us[i] and us[j]:
                    continue
                ref[r] += float(m.V[X[r, i]] @ m.V[X[r, j]])
    assert np.allclose(z0, ref, atol=1e-4), (z0[:3], ref[:3])
    E = m.V[X]; S = E.sum(1)
    P = S[:, None, :] - E
    SI = S - E[:, us, :].sum(1)
    P[:, us, :] = SI[:, None, :]
    eps = 1e-3
    for _ in range(30):
        r = rng.integers(B); f = rng.integers(F); c = rng.integers(k)
        idx = X[r, f]
        old = m.V[idx, c].copy()
        m.V[idx, c] = old + eps; zp = float(m.logits(X[r:r + 1])[0][0])
        m.V[idx, c] = old - eps; zm = float(m.logits(X[r:r + 1])[0][0])
        m.V[idx, c] = old
        assert abs((zp - zm) / (2 * eps) - P[r, f, c]) < 2e-2, \
            ((zp - zm) / (2 * eps), P[r, f, c])
    print('selftest: masked interaction matches explicit pairwise sum and '
          'finite-difference gradients OK')


def _selftest_empty_mask(enc, dim):
    """With an empty mask, MaskedFM must equal baseline.FM step for step."""
    Xtr, ytr, _ = enc['train']
    a = FM(dim, k=8, lr=1e-3, seed=3)
    b = MaskedFM(dim, np.zeros(Xtr.shape[1], dtype=bool), k=8, lr=1e-3, seed=3)
    for i in range(0, 8192 * 5, 8192):
        la = a.step(Xtr[i:i + 8192], ytr[i:i + 8192])
        lb = b.step(Xtr[i:i + 8192], ytr[i:i + 8192])
        assert abs(la - lb) < 1e-9, (la, lb)
    assert np.array_equal(a.V, b.V) and np.array_equal(a.W, b.W)
    print('selftest: empty mask reproduces baseline.FM bit-for-bit OK')


def _audit(splits, fields):
    """Assert the user-static classification is true in the DATA, and report
    how much each field actually varies inside a test user's list."""
    print(f'{"field":24s} {"levels/user (test)":>19s}  {"class":>12s}')
    for f in fields:
        if f == 'dur_bucket':
            key = lambda x: round(x['dur'], 3)
        else:
            key = lambda x, f=f: x[f]
        byu = {}
        for x in splits['test']:
            byu.setdefault(x['user_id'], set()).add(key(x))
        mean_lv = np.mean([len(v) for v in byu.values()])
        cls = 'user-static' if f in USER_STATIC else 'varies'
        if f in USER_STATIC:
            assert mean_lv == 1.0, (f, mean_lv)
        print(f'{f:24s} {mean_lv:19.3f}  {cls:>12s}')
    n_us = sum(f in USER_STATIC for f in fields)
    F = len(fields)
    print(f'\nF={F}  user-static={n_us}  varying={F - n_us}')
    print(f'pairs total={F*(F-1)//2}  dead(UUxUU)={n_us*(n_us-1)//2} '
          f'({100*(n_us*(n_us-1)//2)/(F*(F-1)//2):.1f}%)  kept={F*(F-1)//2 - n_us*(n_us-1)//2}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='graded',
                    choices=['graded', 'ablate', 'selftest', 'audit'])
    ap.add_argument('--fieldset', default=WINNER)
    ap.add_argument('--variant', default='mask_uu',
                    choices=['none', 'mask_uu', 'mask_uu_lin'])
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--only', default='', help='ablate: comma-separated variants')
    ap.add_argument('--k', type=int, default=16)
    a = ap.parse_args()

    if a.mode == 'selftest':
        _selftest_grad()
        splits = load_rich()
        enc, dim = encode_rich(splits, FIELDSETS['official'])
        _selftest_empty_mask(enc, dim)
        raise SystemExit

    splits = load_rich()
    fields = FIELDSETS[a.fieldset]
    print({k_: len(v) for k_, v in splits.items()}, f'F={len(fields)}', flush=True)

    if a.mode == 'audit':
        _audit(splits, fields)
        raise SystemExit

    enc, dim = encode_rich(splits, fields)
    us_pos = mask_for(fields, 'uu')

    if a.mode == 'ablate':
        # VALIDATION-ONLY selection; test printed for the record, never chosen on.
        want = [w for w in a.only.split(',') if w] or ['none', 'mask_uu', 'mask_uu_lin']
        for name in want:
            mask = us_pos if name != 'none' else np.zeros(len(fields), dtype=bool)
            t0 = time.time()
            rs = [run_masked(enc, dim, mask, drop_static_linear=(name == 'mask_uu_lin'),
                             k=a.k, seed=s_, verbose=False) for s_ in range(a.seeds)]
            v = np.array([r['valid']['primary'] for r in rs])
            t = np.array([r['test']['primary'] for r in rs])
            eps_ = [r['best_epoch'] for r in rs]
            print(f'{name:14s} valid {v.mean():.5f} +-{v.std():.5f} '
                  f'(test {t.mean():.5f}) best_ep={eps_} '
                  f'n={a.seeds} {time.time()-t0:.0f}s', flush=True)
        raise SystemExit

    n_us = int(us_pos.sum()); F = len(fields)
    dead = n_us * (n_us - 1) // 2
    run_experiment(
        name=f'exp03_{a.variant}_{a.fieldset}',
        hypothesis=(
            'Removing from the FM interaction every pair of fields that are '
            'BOTH functions of the user alone raises the primary metric. Such '
            f'a pair contributes a constant to every impression in a user\'s '
            'list, and GAUC / nDCG@5 are computed strictly within a user, so '
            f'those {dead} of {F*(F-1)//2} pairs ({100*dead/(F*(F-1)//2):.0f}%) '
            'provably cannot change the metric; they only add collinear '
            'capacity (the per-user constant they express is already free in '
            'W[user_id]) and, worse, they contaminate the user-side embeddings '
            'with gradient that optimises a within-user-invisible direction. '
            'Nothing else changes - same fields, same k=16/lr=1e-3/l2=1e-6/'
            'bs=8192/epochs=40/patience=4, same BCE, same seeds - so any gain '
            'is attributable to the structural mask.'),
        rationale=(
            'Iteration 3 won by adding user-side fields (8 profile + 18 onehot) '
            f'to user_id, which also created all {dead} dead pairs: {n_us} of '
            f'{F} fields are user-static, so 53% of the interaction term is '
            'structurally invisible to the metric. The mask is exact rather '
            'than heuristic: inter_masked = 0.5(|S|^2 - sum|E_f|^2) - '
            '0.5(|S_U|^2 - sum_{f in U}|E_f|^2), with gradient S - E_f for the '
            'varying fields and S_I for the user-static ones, so it costs one '
            'extra (B,k) sum and no wall time. Correctness is asserted by '
            '--mode selftest (explicit pairwise reference + finite-difference '
            'gradients, and an empty mask reproducing baseline.FM bit-for-bit) '
            'and the user-static classification is asserted against the data by '
            '--mode audit (every masked field has exactly 1.000 levels per test '
            'user). No new columns are read, so iteration 3\'s legitimacy '
            'argument carries over unchanged. Variant selected on VALIDATION '
            'only.'),
        train_fn=lambda s_: run_masked(enc, dim, us_pos,
                                       drop_static_linear=(a.variant == 'mask_uu_lin'),
                                       k=a.k, seed=s_, verbose=False),
        seeds=3,
        config={'model': 'MaskedFM', 'variant': a.variant, 'k': a.k, 'lr': 0.001,
                'l2': 1e-6, 'epochs': 40, 'bs': 8192, 'patience': 4,
                'loss': 'pointwise BCE', 'fieldset': a.fieldset,
                'n_fields': F, 'dim': dim, 'n_user_static': n_us,
                'pairs_total': F * (F - 1) // 2, 'pairs_masked': dead,
                'pairs_kept': F * (F - 1) // 2 - dead,
                'variant_selected_on': 'validation, 3-seed mean'})