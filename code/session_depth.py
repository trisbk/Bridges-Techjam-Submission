"""Run 38: test within-session position as a missing ranking signal.

Residual analysis on the new champion identified tab=1 as the highest-priority
open slice. A train-only probe then found a strong monotonic drop in long_view
as users move deeper into a viewing session, suggesting session fatigue that
the existing gap and lifetime-history features do not capture.

Two causal features are tested:
- sess_pos: bucketed position within the current session (label-free)
- sess_hit: long_views observed earlier in the same session

Two arms separate the mechanisms:
- R38a: RICH + tab_n + sess_pos
- R38b: RICH + tab_n + sess_pos + sess_hit

Features use only information available before the current impression.
Selection is validation-only across 3 seeds. A >0.001 gain over the fresh
control must first survive a temporal-shuffle falsification test before any
5-seed promotion check against the banked champion.
"""
import collections
import numpy as np
from evaluate import evaluate
from staleness_ablation import build_features, RICH
from tab_surface import (add_tab_features, make_train_fn, committee,
                         split_of)
from harness import run_experiment, PROMOTION_MARGIN, BANKED_VALID
from controls import time_shuffle

SESSION_GAP_MS = 30 * 60 * 1000        # 30 minutes ends a visit
CHAMPION = RICH + ['tab_n']            # R33c's feature set
SESSKEYS = ['sess_pos', 'sess_hit']


def _pos_bucket(n):
    return ('0' if n == 0 else '1-2' if n <= 2 else '3-5' if n <= 5
            else '6-10' if n <= 10 else '11-20' if n <= 20
            else '21-50' if n <= 50 else '50+')


def add_session_features(splits):
    """Within-visit position and outcome momentum, per user, causal."""
    rows = [x for rws in splits.values() for x in rws]
    rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    st = {}
    for x in rows:
        h = st.get(x['user_id'])
        if h is None or h['last_t'] is None:
            pos, hits = 0, 0
        else:
            d = ((x['date'] - h['last_t'][0]) * 86400_000
                 + (x['t'] - h['last_t'][1]))
            if d >= SESSION_GAP_MS:
                pos, hits = 0, 0            # new visit
            else:
                pos, hits = h['pos'] + 1, h['hits']
        x['sess_pos'] = _pos_bucket(pos)
        x['sess_hit'] = str(hits) if hits < 3 else '3+'
        # update AFTER featurizing — own label never in own features
        st[x['user_id']] = {'last_t': (x['date'], x['t']), 'pos': pos,
                            'hits': hits + int(x['y'])}
    return splits


def slice_primary(uva, yva, preds, tabs, want='1'):
    """Diagnostic only: the arm's score restricted to the target slice.
    Reported for the record, never selected on."""
    s = np.mean(preds, 0)
    idx = [i for i, t in enumerate(tabs) if t == want]
    return float(evaluate([uva[i] for i in idx], [yva[i] for i in idx],
                          [s[i] for i in idx])['primary'])


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.join('..', 'agent'))
    import belief_state as BS

    print("building rich features + tab_n + session structure ...")
    splits = add_session_features(add_tab_features(build_features('continuous')))
    tabs_va = [x['tab'] for x in splits['valid']]

    # sanity: the features must actually vary and must be self-exclusive
    dist = collections.Counter(x['sess_pos'] for x in splits['train'])
    print("train sess_pos distribution:",
          {k: dist[k] for k in ('0', '1-2', '3-5', '6-10', '11-20', '21-50',
                                '50+')})

    arms = [
        ('R38-ctrl RICH + tab_n (champion control)', CHAMPION),
        ('R38a + sess_pos', CHAMPION + ['sess_pos']),
        ('R38b + sess_pos + sess_hit', CHAMPION + SESSKEYS),
    ]
    HYP = ('Within a visit, long_view rate falls 2.9x with depth on tab=1 '
           '(0.418 at position 0 to 0.143 at 50+), and the evaluation ranks '
           'rows from many visits against each other inside one user. The '
           'shipped features carry the last inter-impression gap and lifetime '
           'counts but not depth-within-visit, so the model cannot order an '
           'early-visit row against a deep one.')
    RAT = ('Residual analyzer against the new champion returns residual_tab_1 '
           '(20,119 users, oracle headroom 0.2113). tab=1 is 73% of rows, so '
           'the actionable move is structure inside the slice, not a slice '
           'prior. Grounding probe on TRAIN rows only.')

    results, preds_by_arm = {}, {}
    for name, fields in arms:
        store = []
        fn, dim, (uva, yva) = make_train_fn(splits, fields, store)
        rec = run_experiment(
            name=name, hypothesis=HYP, rationale=RAT, train_fn=fn, seeds=3,
            config={'fields': fields, 'k': 16, 'lr': 0.001, 'K': 4,
                    'dim': dim, 'session_gap_ms': SESSION_GAP_MS,
                    'hypothesis_id': 'residual_tab_1'})
        results[name] = rec
        preds_by_arm[name] = store
        print(f"    tab=1 slice primary (valid, diagnostic): "
              f"{slice_primary(uva, yva, store, tabs_va):.5f}")

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
    h = BS.get(st, 'residual_tab_1')
    h['mechanism'] = 'temporal'         
    BS.attach_evidence(st, 'residual_tab_1', {
        'run': f'{win_name} (3 seeds)', 'valid_mean': win_val,
        'test_mean': results[win_name]['test_mean'],
        'control_run': ctrl_name, 'control_valid': ctrl,
        'delta_vs_control': round(win_val - ctrl, 5)})
    BS.save(st)

    if win_val <= ctrl + PROMOTION_MARGIN:
        print("NO WIN on validation: no falsification control needed, no "
              "committee run, nothing banked (pre-committed rule).")
        BS.refute(st, 'residual_tab_1')
        BS.save(st)
        raise SystemExit(0)

    print("\nWIN on validation -> falsification control REQUIRED before any "
          "promotion. Synthesizing the time_shuffle placebo over the new "
          "session keys (per-user, per-split marginals preserved, alignment "
          "destroyed).")
    keys = ['sess_pos'] if win_name.startswith('R38a') else SESSKEYS
    all_rows = [x for rws in splits.values() for x in rws]
    shuffled = time_shuffle(all_rows, keys, split_of, seed=0)
    spl_C = {s: [r for r in shuffled if split_of(r) == s]
             for s in ('train', 'valid', 'test')}
    fn, _, _ = make_train_fn(spl_C, CHAMPION + keys)
    ctl = run_experiment(
        name=f'R38-placebo time-shuffled {"+".join(keys)}',
        hypothesis='If the gain is depth-within-visit, destroying which '
                   'impression each session vector is attached to must '
                   'collapse it; if it survives, the feature is acting as a '
                   'per-user fingerprint (capacity), not as timing.',
        rationale='Mechanism tag temporal -> controls.time_shuffle.',
        train_fn=fn, seeds=3,
        config={'fields': CHAMPION + keys, 'control_of': win_name,
                'hypothesis_id': 'residual_tab_1'})
    gain = win_val - ctrl
    survived = ctl['valid_mean'] - ctrl
    print(f"\ncontrol: gain {gain:+.5f}, survives shuffle {survived:+.5f} "
          f"({survived / gain:.0%} of it)")
    st = BS.load()
    if survived > 0.5 * gain:
        BS.attach_control(st, 'residual_tab_1', 'time_shuffle', 'failed',
                          f'placebo valid {ctl["valid_mean"]:.5f} vs control '
                          f'{ctrl:.5f}: {survived / gain:.0%} of the gain '
                          f'survives destroyed alignment')
        BS.refute(st, 'residual_tab_1', by_control=True)
        BS.save(st)
        print("CONTROL FAILED: most of the gain survives destroyed timing, "
              "so this is not a within-visit position effect. Mechanism "
              "refuted; no promotion.")
        raise SystemExit(0)
    BS.attach_control(st, 'residual_tab_1', 'time_shuffle', 'passed',
                      f'placebo valid {ctl["valid_mean"]:.5f} vs control '
                      f'{ctrl:.5f}: only {max(0.0, survived) / gain:.0%} of '
                      f'the gain survives destroyed alignment')
    BS.save(st)

    print("\ncontrol PASSED -> 5-seed committee promotion check vs banked "
          f"validation {BANKED_VALID}")
    rv, rt = committee(splits, CHAMPION + keys)
    from harness import log_committee_result
    rec = log_committee_result(
        f'R38c {"+".join(["RICH", "tab_n"] + keys)} 5-seed committee',
        rv, rt, config={'fields': CHAMPION + keys, 'k': 16, 'lr': 0.001,
                        'K': 4, 'seeds': 5,
                        'hypothesis_id': 'residual_tab_1'},
        note='Campaign 6 iteration 1 promotion check.')
    if rv['primary'] > BANKED_VALID + PROMOTION_MARGIN:
        print(f"PROMOTE: {rv['primary']:.5f} > {BANKED_VALID} + "
              f"{PROMOTION_MARGIN}")
        st = BS.load(); BS.promote(st, 'residual_tab_1'); BS.save(st)
    else:
        print(f"NO PROMOTION: {rv['primary']:.5f} does not clear "
              f"{BANKED_VALID} + {PROMOTION_MARGIN}; incumbent kept.")