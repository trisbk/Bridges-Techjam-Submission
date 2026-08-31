"""Run 33: test same-surface causal history from a residual-identified weakness.

Residual analysis found tab=0 as the champion's weakest high-value validation
slice. The hypothesis is that global history features miss surface-specific
behavior, so this run adds:
- tab_prev1: previous outcome on the same surface
- tab_hist10: recent long_view history on the same surface
- tab_n: prior impression count on the same surface (label-free familiarity)

Two arms separate the mechanism:
- R33a: RICH + same-surface recency + familiarity
- R33b: RICH + tab_n familiarity only

This distinguishes whether any gain comes from recent same-surface outcomes or
simply knowing how familiar the user is with that surface.

Selection is validation-only across 3 seeds. An arm must beat the control by
>0.001 to trigger a falsification control; only a supported gain proceeds to
the 5-seed committee promotion check.
"""
import collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from sequences import encode_rows
from staleness_ablation import build_features, RICH
from harness import run_experiment, PROMOTION_MARGIN
from controls import time_shuffle

TABSEQ = ['tab_prev1', 'tab_hist10', 'tab_n']
BANKED_VALID = 0.61906          # 5-seed R24b committee, the incumbent


def add_tab_features(splits):
    """Per-(user, surface) causal history, self-exclusive: state updates
    only AFTER the row is featurized, and rows are visited in the user's
    chronological order, so nothing later-in-time can enter a feature."""
    rows = [x for rws in splits.values() for x in rws]
    rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    st = {}
    for x in rows:
        h = st.setdefault((x['user_id'], x['tab']),
                          {'prev1': None, 'last10': collections.deque(maxlen=10),
                           'n': 0})
        x['tab_prev1'] = 'none' if h['prev1'] is None else str(h['prev1'])
        x['tab_hist10'] = 'none' if not h['last10'] else str(sum(h['last10']))
        n = h['n']
        x['tab_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
                      else '11-30' if n <= 30 else '31-100' if n <= 100
                      else '100+')
        h['prev1'] = x['y']; h['last10'].append(x['y']); h['n'] += 1
    return splits


def split_of(row):
    if row['date'] <= 20220421:
        return 'train'
    if row['date'] <= 20220428:
        return 'valid'
    return 'test'


def make_train_fn(splits, fields, store=None):
    """One FM per seed: k=16, listwise InfoNCE K=4, lr 1e-3, patience 4 on
    validation primary — the banked recipe, single model."""
    enc, dim = encode_rows(splits, fields)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    def train_fn(seed):
        m = FM(dim, k=16, lr=0.001, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 41):
            P, N = sample_lists(pairs_users, rng, 4)
            for i in range(0, len(P), 8192):
                p = P[i:i + 8192]; n = N[i:i + 8192]
                infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
            va = evaluate(uva, yva, m.predict(Xva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= 4:
                    break
        m.V, m.W, m.b = best_state
        if store is not None:
            pv = m.predict(Xva)
            store.append((pv - pv.mean()) / pv.std())
        return {'valid': evaluate(uva, yva, m.predict(Xva)),
                'test': evaluate(ute, yte, m.predict(Xte))}
    return train_fn, dim, (uva, yva)


def committee(splits, fields, seeds=5):
    """5-seed committee, z-scored mean, the banked ensembling recipe."""
    enc, dim = encode_rows(splits, fields)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    va_p, te_p = [], []
    for seed in range(seeds):
        m = FM(dim, k=16, lr=0.001, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 41):
            P, N = sample_lists(pairs_users, rng, 4)
            for i in range(0, len(P), 8192):
                p = P[i:i + 8192]; n = N[i:i + 8192]
                infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
            va = evaluate(uva, yva, m.predict(Xva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= 4:
                    break
        m.V, m.W, m.b = best_state
        pv, pt = m.predict(Xva), m.predict(Xte)
        va_p.append((pv - pv.mean()) / pv.std())
        te_p.append((pt - pt.mean()) / pt.std())
    return (evaluate(uva, yva, np.mean(va_p, 0)),
            evaluate(ute, yte, np.mean(te_p, 0)))


def slice_scores(uva, yva, preds, tabs):
    """Diagnostic: where the arm's validation predictions land on the slice
    the hypothesis was aimed at. Reported, never selected on."""
    s = np.mean(preds, 0)
    idx = [i for i, t in enumerate(tabs) if t == '0']
    r = evaluate([uva[i] for i in idx], [yva[i] for i in idx],
                 [s[i] for i in idx])
    return float(r['primary'])


if __name__ == '__main__':
    print("building rich features (continuous, as shipped) + tab history ...")
    splits = add_tab_features(build_features('continuous'))
    tabs_va = [x['tab'] for x in splits['valid']]

    arms = [
        ('R33-ctrl RICH (control)', RICH),
        ('R33a RICH + tab recency + tab_n', RICH + TABSEQ),
        ('R33b RICH + tab_n only', RICH + ['tab_n']),
    ]
    results, preds_by_arm = {}, {}
    for name, fields in arms:
        store = []
        fn, dim, (uva, yva) = make_train_fn(splits, fields, store)
        rec = run_experiment(
            name=name,
            hypothesis=('The relevant "what did the user just do" for a row '
                        'on surface T is the previous event on surface T; '
                        'the shipped stream-wide history reports a different '
                        'surface with a 10x different positive rate.'),
            rationale=('Residual analyzer picked tab=0 as the champion\'s '
                       'worst slice by expected value; tab base rates are '
                       'already in the model, per-surface recency is not.'),
            train_fn=fn, seeds=3,
            config={'fields': fields, 'k': 16, 'lr': 0.001, 'K': 4,
                    'dim': dim, 'hypothesis_id': 'residual_tab_0'})
        results[name] = rec
        preds_by_arm[name] = store
        print(f"    tab=0 slice primary (valid, diagnostic): "
              f"{slice_scores(uva, yva, store, tabs_va):.5f}")

    ctrl = results['R33-ctrl RICH (control)']['valid_mean']
    challengers = {n: r['valid_mean'] for n, r in results.items()
                   if n != 'R33-ctrl RICH (control)'}
    win_name = max(challengers, key=challengers.get)
    win_val = challengers[win_name]
    print(f"\ncontrol valid {ctrl:.5f}; best challenger {win_name} "
          f"{win_val:.5f} (delta {win_val - ctrl:+.5f}, "
          f"margin {PROMOTION_MARGIN})")

    if win_val <= ctrl + PROMOTION_MARGIN:
        print("NO WIN on validation: no falsification control needed, no "
              "committee run, nothing banked (pre-committed rule).")
        raise SystemExit(0)

    print("\nWIN on validation -> falsification control REQUIRED before any "
          "promotion. Synthesizing the time_shuffle placebo over the new "
          "per-surface keys (per-user, per-split marginals preserved, "
          "alignment destroyed).")
    keys = TABSEQ if win_name.startswith('R33a') else ['tab_n']
    all_rows = [x for rws in splits.values() for x in rws]
    shuffled = time_shuffle(all_rows, keys, split_of, seed=0)
    spl_C = {s: [r for r in shuffled if split_of(r) == s]
             for s in ('train', 'valid', 'test')}
    fields = RICH + keys
    fn, _, _ = make_train_fn(spl_C, fields)
    ctl = run_experiment(
        name=f'R33-placebo time-shuffled {"+".join(keys)}',
        hypothesis='If the gain is same-surface RECENCY, destroying which '
                   'impression each per-surface feature vector is attached '
                   'to must collapse it.',
        rationale='Mechanism tag temporal -> controls.time_shuffle, the same '
                  'placebo that proved the headline sequence claim.',
        train_fn=fn, seeds=3,
        config={'fields': fields, 'control_of': win_name})
    survived = ctl['valid_mean'] - ctrl
    gain = win_val - ctrl
    print(f"\ncontrol: gain {gain:+.5f}, survives shuffle {survived:+.5f} "
          f"({survived / gain:.0%} of it)")
    if survived > 0.5 * gain:
        print("CONTROL FAILED: most of the gain survives destroyed timing, "
              "so this is not a recency effect. Refuting the mechanism; no "
              "promotion.")
        raise SystemExit(0)

    print("\ncontrol PASSED -> 5-seed committee promotion check vs banked "
          f"validation {BANKED_VALID}")
    rv, rt = committee(splits, RICH + keys)
    print(f"committee valid {rv['primary']:.5f} test {rt['primary']:.5f} "
          f"(GAUC {rt['GAUC']:.4f} nDCG@5 {rt['nDCG@5']:.4f})")
    if rv['primary'] > BANKED_VALID + PROMOTION_MARGIN:
        print(f"PROMOTE: {rv['primary']:.5f} > {BANKED_VALID} + "
              f"{PROMOTION_MARGIN}")
    else:
        print(f"NO PROMOTION: {rv['primary']:.5f} does not clear "
              f"{BANKED_VALID} + {PROMOTION_MARGIN}; incumbent kept.")