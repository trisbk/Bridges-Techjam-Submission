"""Completion of R33's pre-committed decision rule, run by the operator.

Provenance note, stated plainly: the autonomous session that ran R33
(tab_surface.py, 31 Aug) reached the last step of its own pre-committed
rule — "a win with a passing control goes on to the 5-seed committee
promotion check" — but that step calls committee() directly rather than
going through the harness, and the driver fault ate the session's stdout,
so no record of the committee's numbers exists. This script re-runs that
final step exactly as written, with the 5 singles logged through the
harness so the record is complete. It is operator-run bookkeeping of the
agent's rule, not a new hypothesis.

Decision rule (unchanged from tab_surface.py):
    promote iff 5-seed committee validation primary > 0.61906 + 0.001.

Run from code/:  python3 tab_committee_check.py     (~12 min)
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

    # make_train_fn's store captures z-scored validation predictions only;
    # training is deterministic per seed, so retraining the same 5 seeds
    # reproduces the identical models to also collect test predictions
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
