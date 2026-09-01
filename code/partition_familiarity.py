"""Run 39: test whether tab_n's partitioned-familiarity idea generalizes.

The autonomous queue selected the hist_n=31-100 slice. Rather than treating
the slice itself as a defect, the agent identified a missing signal inside it:
the champion tracks prior long_view outcomes by tag and author, but not how
often the user has previously been exposed to those partitions.

This run adds label-free, causal exposure counts:
- tag_n: prior impressions carrying the current video's tag
- auth_n: prior impressions from the current author

A train-only diagnostic showed that exposure count still strongly separates
long_view rates even after conditioning on the existing outcome-history
features, motivating the denominator/familiarity hypothesis.

Three arms test the generalization:
- R39-ctrl: RICH + tab_n
- R39a: + tag_n
- R39b: + tag_n + auth_n

Selection is validation-only across 3 seeds. A >0.001 gain must first survive
a temporal-shuffle falsification control; only a supported gain proceeds to
the 5-seed committee promotion check against the banked champion.
"""
import collections
import numpy as np
from evaluate import evaluate
from staleness_ablation import build_features, RICH
from tab_surface import add_tab_features, make_train_fn, committee, split_of
from familiarity_diagnostics import add_familiarity_features
from harness import run_experiment, PROMOTION_MARGIN, BANKED_VALID
from controls import time_shuffle

CHAMPION = RICH + ['tab_n']            # R33c's feature set
HID = 'residual_hist_31-100'


def slice_primary(uva, yva, preds, hists, want='31-100'):
    """Diagnostic only: the arm's score restricted to the target slice.
    Reported for the record, never selected on."""
    s = np.mean(preds, 0)
    idx = [i for i, h in enumerate(hists) if h == want]
    return float(evaluate([uva[i] for i in idx], [yva[i] for i in idx],
                          [s[i] for i in idx])['primary'])


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join('..', 'agent'))
    import belief_state as BS

    print("building rich features + tab_n + partition exposure counts ...")
    splits = add_familiarity_features(
        add_tab_features(build_features('continuous')))
    hist_va = [x['hist_n'] for x in splits['valid']]

    for k in ('tag_n', 'auth_n'):
        d = collections.Counter(x[k] for x in splits['train'])
        print(f"train {k} distribution:", dict(d.most_common()))

    arms = [
        ('R39-ctrl RICH + tab_n (champion control)', CHAMPION),
        ('R39a + tag_n', CHAMPION + ['tag_n']),
        ('R39b + tag_n + auth_n', CHAMPION + ['tag_n', 'auth_n']),
    ]
    HYP = ('The champion holds per-partition OUTCOME counts (auth_hist, '
           'tag_hist = prior long_views, capped 3+) and no exposure counts, '
           'so it has the numerator of a per-partition hit rate without the '
           'denominator. On train rows, long_view rate falls 4.7-8.6x with '
           'tag_n inside every stratum of tag_hist, on 1.1M rows. Supplying '
           'the denominator should let the FM form the rate (both fields are '
           'narrow, so the bilinear form has rank to spare).')
    RAT = ('Residual analyzer returns residual_hist_31-100 (10,947 users, '
           'oracle headroom 0.13222). The slice scores ABOVE overall, so its '
           'headroom is slice size, not a defect; the actionable reading is '
           'that hist_n is a lifetime count that says nothing about how the '
           'history is distributed. tab_n — the project\'s one winning '
           'feature — is exactly this construction on the surface partition, '
           'and R37 revised its mechanism to partitioned familiarity.')

    results, preds_by_arm = {}, {}
    for name, fields in arms:
        store = []
        fn, dim, (uva, yva) = make_train_fn(splits, fields, store)
        rec = run_experiment(
            name=name, hypothesis=HYP, rationale=RAT, train_fn=fn, seeds=3,
            config={'fields': fields, 'k': 16, 'lr': 0.001, 'K': 4,
                    'dim': dim, 'hypothesis_id': HID})
        results[name] = rec
        preds_by_arm[name] = store
        print(f"    hist=31-100 slice primary (valid, diagnostic): "
              f"{slice_primary(uva, yva, store, hist_va):.5f}")

    ctrl_name = arms[0][0]
    ctrl = results[ctrl_name]['valid_mean']
    challengers = {n: r['valid_mean'] for n, r in results.items()
                   if n != ctrl_name}
    win_name = max(challengers, key=challengers.get)
    win_val = challengers[win_name]
    print(f"\ncontrol valid {ctrl:.5f}; best challenger {win_name} "
          f"{win_val:.5f} (delta {win_val - ctrl:+.5f}, "
          f"margin {PROMOTION_MARGIN})")

    st = BS.load()
    BS.attach_evidence(st, HID, {
        'run': f'{win_name} (3 seeds)', 'valid_mean': win_val,
        'test_mean': results[win_name]['test_mean'],
        'control_run': ctrl_name, 'control_valid': ctrl,
        'delta_vs_control': round(win_val - ctrl, 5)})
    BS.save(st)

    if win_val <= ctrl + PROMOTION_MARGIN:
        print("NO WIN on validation: no falsification control needed, no "
              "committee run, nothing banked (pre-committed rule).")
        BS.refute(st, HID)
        BS.save(st)
        raise SystemExit(0)

    print("\nWIN on validation -> falsification control REQUIRED before any "
          "promotion. Synthesizing the time_shuffle placebo over the new "
          "partition-count keys (per-user, per-split marginals preserved, "
          "alignment to this row's tag/author destroyed).")
    keys = ['tag_n'] if win_name.startswith('R39a') else ['tag_n', 'auth_n']
    all_rows = [x for rws in splits.values() for x in rws]
    shuffled = time_shuffle(all_rows, keys, split_of, seed=0)
    spl_C = {s: [r for r in shuffled if split_of(r) == s]
             for s in ('train', 'valid', 'test')}
    fn, _, _ = make_train_fn(spl_C, CHAMPION + keys)
    ctl = run_experiment(
        name=f'R39-placebo time-shuffled {"+".join(keys)}',
        hypothesis='If the gain is per-partition familiarity, destroying '
                   'which impression each count vector is attached to must '
                   'collapse it; if it survives, the count is a coarse '
                   'per-user fingerprint (heavy users carry big counts '
                   'everywhere) and the partition story is wrong.',
        rationale='Mechanism tag temporal -> controls.time_shuffle.',
        train_fn=fn, seeds=3,
        config={'fields': CHAMPION + keys, 'control_of': win_name,
                'hypothesis_id': HID})
    gain = win_val - ctrl
    survived = ctl['valid_mean'] - ctrl
    print(f"\ncontrol: gain {gain:+.5f}, survives shuffle {survived:+.5f} "
          f"({survived / gain:.0%} of it)")
    st = BS.load()
    if survived > 0.5 * gain:
        BS.attach_control(st, HID, 'time_shuffle', 'failed',
                          f'placebo valid {ctl["valid_mean"]:.5f} vs control '
                          f'{ctrl:.5f}: {survived / gain:.0%} of the gain '
                          f'survives destroyed alignment')
        BS.refute(st, HID, by_control=True)
        BS.save(st)
        print("CONTROL FAILED: most of the gain survives destroyed alignment, "
              "so this is not per-partition familiarity. Mechanism refuted; "
              "no promotion.")
        raise SystemExit(0)
    BS.attach_control(st, HID, 'time_shuffle', 'passed',
                      f'placebo valid {ctl["valid_mean"]:.5f} vs control '
                      f'{ctrl:.5f}: only {max(0.0, survived) / gain:.0%} of '
                      f'the gain survives destroyed alignment')
    BS.save(st)

    print("\ncontrol PASSED -> 5-seed committee promotion check vs banked "
          f"validation {BANKED_VALID}")
    rv, rt = committee(splits, CHAMPION + keys)
    from harness import log_committee_result
    rec = log_committee_result(
        f'R39c {"+".join(["RICH", "tab_n"] + keys)} 5-seed committee',
        rv, rt, config={'fields': CHAMPION + keys, 'k': 16, 'lr': 0.001,
                        'K': 4, 'seeds': 5, 'hypothesis_id': HID},
        note='Campaign 6 iteration 2 promotion check.')
    if rv['primary'] > BANKED_VALID + PROMOTION_MARGIN:
        print(f"PROMOTE: {rv['primary']:.5f} > {BANKED_VALID} + "
              f"{PROMOTION_MARGIN}")
        st = BS.load(); BS.promote(st, HID); BS.save(st)
    else:
        print(f"NO PROMOTION: {rv['primary']:.5f} does not clear "
              f"{BANKED_VALID} + {PROMOTION_MARGIN}; incumbent kept.")