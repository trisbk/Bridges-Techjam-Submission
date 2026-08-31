"""Complete R33's pre-committed 5-seed promotion check.

The autonomous R33 session reached the committee-check stage, but a driver
failure left no recorded committee output. This operator-run script repeats
only that final pre-committed step and logs all five seeds through the harness
for complete provenance. It introduces no new hypothesis or decision rule.

Promotion occurs only if the 5-seed validation primary exceeds the banked
0.61906 score by more than the pre-committed 0.001 margin.
"""
import numpy as np
from evaluate import evaluate
from staleness_ablation import build_features, RICH
from tab_surface import add_tab_features, make_train_fn, BANKED_VALID
from harness import run_experiment, PROMOTION_MARGIN
from sequences import encode_rows

if __name__ == '__main__':
    print("building rich features (continuous, as shipped) + tab history ...")
    splits = add_tab_features(build_features('continuous'))
    fields = RICH + ['tab_n']

    va_store = []
    fn, dim, (uva, yva) = make_train_fn(splits, fields, va_store)

    enc, _ = encode_rows(splits, fields)
    Xte, yte, ute = enc['test']

    rec = run_experiment(
        name='R33c committee completion RICH + tab_n (5 seeds)',
        hypothesis=('Operator completion of R33\'s pre-committed rule: the '
                    'validated, control-passing tab_n win goes to the 5-seed '
                    'committee promotion check that the agent session\'s '
                    'driver fault left unrecorded.'),
        rationale=('tab_surface.py line 241 ran committee() outside the '
                   'harness and its stdout was lost; this re-runs the same '
                   'step with the singles logged.'),
        train_fn=fn, seeds=5,
        config={'fields': fields, 'k': 16, 'lr': 0.001, 'K': 4, 'dim': dim,
                'hypothesis_id': 'residual_tab_0',
                'completes': 'R33b pre-committed rule, final step'})

    from baseline import FM
    from pairwise import build_pair_index
    from listwise import sample_lists, infonce_step
    Xtr, ytr, utr = enc['train']; Xva, yva2, uva2 = enc['valid']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    te_preds = []
    for seed in range(5):
        m = FM(dim, k=16, lr=0.001, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 41):
            P, N = sample_lists(pairs_users, rng, 4)
            for i in range(0, len(P), 8192):
                infonce_step(m, Xtr[P[i:i + 8192]],
                             Xtr[N[i:i + 8192].reshape(-1)],
                             len(P[i:i + 8192]), 4)
            va = evaluate(uva2, yva2, m.predict(Xva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= 4:
                    break
        m.V, m.W, m.b = best_state
        pt = m.predict(Xte)
        te_preds.append((pt - pt.mean()) / pt.std())

    rv = evaluate(uva, yva, np.mean(va_store, 0))
    rt = evaluate(ute, yte, np.mean(te_preds, 0))
    print("\n=== R33 COMMITTEE (RICH + tab_n, 5 seeds, z-scored mean) ===")
    print(f"valid : GAUC {rv['GAUC']:.4f} | nDCG@5 {rv['nDCG@5']:.4f} "
          f"| primary {rv['primary']:.5f}")
    print(f"test  : GAUC {rt['GAUC']:.4f} | nDCG@5 {rt['nDCG@5']:.4f} "
          f"| primary {rt['primary']:.5f}")
    bar = BANKED_VALID + PROMOTION_MARGIN
    if rv['primary'] > bar:
        print(f"\nPRE-COMMITTED RULE SAYS PROMOTE: {rv['primary']:.5f} > {bar}")
    else:
        print(f"\nNO PROMOTION: {rv['primary']:.5f} does not clear {bar}; "
              f"incumbent 0.61906 kept.")