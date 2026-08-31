"""Run 39 — campaign 6, autonomous iteration 2: per-partition EXPOSURE counts.

Hypothesis source (queue, not brainstorm). `agent/residual_analysis.py`
against the champion (R33c, RICH + tab_n, validation 0.62059) now walks past
the resolved `tab=0` (confirmed) and `tab=1` (refuted) and returns
`residual_hist_31-100`: 10,947 users, oracle headroom 0.13222, mechanism tag
`temporal`. `priority.py --recompute` and `belief_state.py --next` both
return it, so it is what this iteration took.

**Restating the hypothesis before spending compute.** This slice is not a
defect. It scores 0.62154 against 0.62059 overall — *above* the model's
average — and its EV under the old measure is exactly 0.0. All of its
headroom is the arithmetic of slice size. So, as with `tab=1` last
iteration, the only actionable reading is structure INSIDE the slice that
the feature set cannot express, and here the slice definition names it:
`hist_n` is a lifetime impression count in six log buckets, the 31-100
bucket spans a 3.2x range, and nothing in it says how those 31-100
impressions were DISTRIBUTED across what the user was shown.

The champion carries per-author and per-tag *outcome* counts (`auth_hist`,
`tag_hist` — prior long_views, capped at 3+) and no exposure counts for
either partition. So "this user has seen twelve videos with this tag and
finished three" and "this user has seen three and finished three" are the
same row to the model: it has the numerator of a per-partition hit rate and
not the denominator. That missing denominator is precisely what `tab_n`
supplied for the surface partition — the one feature that has won in this
project (+0.0024, R33b) — and R37's discriminating control revised its
mechanism from surface familiarity to *partitioned familiarity*, of which
only about half was surface-specific. The generalisation this run tests is
that the winning axis was partitioned exposure counting, and that the tag
and author partitions have been left without it.

Grounding probe, TRAIN rows only (`familiarity_diagnostics.py`, output in
`logs/familiarity_diagnostic.out`; valid/test labels never read). long_view
rate by `tag_n` INSIDE each stratum of the shipped `tag_hist`:

    tag_hist \\ tag_n     0      1-3    4-10   11-30  31-100   100+       n
    0                  0.342   0.248   0.156   0.082   0.040     —    555,498
    1                    —     0.418   0.260   0.136   0.074     —    197,238
    2                    —     0.507   0.351   0.188   0.090     —    109,537
    3+                   —     0.586   0.491   0.370   0.242   0.125  278,839

A clean monotone decline of 4.7x to 8.6x inside every stratum of the feature
that is supposed to make it redundant, on 1.1M rows. This is the shape of a
rate: at a fixed number of prior successes, more prior exposures means a
lower hit rate. `tag_n` and `tag_hist` are both narrow fields (6 and 4
values), so their bilinear form has rank min(16,16)=16 over a 4x6 grid — the
FM has ample rank to represent the ratio exactly, if it is given the
denominator. And the slice that raised the hypothesis is where the
denominator has accumulated without saturating: inside hist_n=31-100, 62% of
train rows sit at tag_n between 4 and 30.

`auth_n` is the same construction on the author partition and is included as
a separate arm because its support is thin — 94% of train rows are at
auth_n=0 — so it may add cardinality without information (the failure mode
Run 32 measured: a narrow field's rank bounds every interaction it has with
`user_id` and `video_id`, so an uninformative one is not free).

Features (causal, self-exclusive, label-free by construction — they count
impressions, never outcomes; state updates only AFTER the row is featurised
and rows are visited in each user's chronological order):

  tag_n  : log-bucketed count of the user's prior impressions carrying this
           video's first tag  (0 / 1-3 / 4-10 / 11-30 / 31-100 / 100+)
  auth_n : bucketed count of the user's prior impressions from this author
           (0 / 1 / 2 / 3-5 / 6-10 / 11+)

Three arms:

  R39-ctrl  RICH + tab_n                     (fresh in-session champion control)
  R39a      RICH + tab_n + tag_n             (the tag denominator)
  R39b      RICH + tab_n + tag_n + auth_n    (both partitions)

Mechanism tag: TEMPORAL, inherited from the analyzer's `hist` hint and
correct for the control it selects. What `time_shuffle` falsifies for a
per-partition count is ALIGNMENT: it permutes which impression carries which
count vector within each (user, split), so every per-user marginal survives
and only the pairing with this row's actual tag/author dies. If the gain
survives that, the count was acting as a coarse per-user fingerprint —
heavy users carry big counts everywhere — and the partitioned-familiarity
story is wrong. That is the same test that ruled out the fingerprint reading
of `tab_n`.

Pre-committed decision rule (written before any arm ran):
  * selection on VALIDATION only, 3 seeds per arm through the harness;
  * an arm counts as a win only if it beats R39-ctrl by more than
    PROMOTION_MARGIN (0.001) on validation;
  * a win triggers the falsification control FIRST; more than half the gain
    surviving the shuffle refutes the mechanism and forbids promotion;
  * only a control-passing win reaches the 5-seed committee check against
    the banked committee's validation 0.62059, and only a committee clearing
    0.62059 + 0.001 promotes.

Run from code/:  python3 partition_familiarity.py     (~20 min)
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
