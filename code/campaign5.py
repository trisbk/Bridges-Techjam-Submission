"""Campaign 5: the completion run that promotes the loop's discovery.

Context, stated plainly. The v2-loop iteration (31 Aug morning) produced
R33b: tab_n, the agent's own residual-derived hypothesis, validated with a
passing time-shuffle control. The operator-completed 5-seed committee check
(tab_committee_check.py, logged as R33c) measured committee validation
0.62059 — above the banked 0.61906 + 0.001 promotion margin, so the
pre-committed rule written in tab_surface.py says PROMOTE. This campaign
executes that promotion the way every earlier banking event happened:
inside a run governed by the official convergence rule (epsilon 0.002,
N=3 on validation, 50-iteration cap, 6h ceiling).

  Iteration 1  bank R33c as a rule-tagged committee WIN
               (harness.log_committee_result; numbers from the completed
               R33c evaluation, singles already in LOG.jsonl)
  Iteration 2  R34: does per-surface RECENCY add anything on top of
               familiarity? RICH + tab_n + tab_prev1 + tab_hist10.
  Iteration 3  R35: next slice on the residual queue (dur=9): duration-
               bucket familiarity dur_n, same construction as tab_n.
  Iteration 4  R36: is tab_n's log-bucketing losing signal? Finer buckets.

If iterations 2-4 all fail to improve validation by more than epsilon, the
run has converged by the official rule at the iteration-1 checkpoint. Any
win instead resets the counter and extends the run (handled by rerunning
with more iterations; not expected).

This is an operator-driven completion run (like the interactive campaign),
not an unattended one, and is documented as such.

Run from code/:  python3 campaign5.py     (~15 min)
"""
import collections, json, os
import numpy as np
from evaluate import evaluate
from sequences import encode_rows
from staleness_ablation import build_features, RICH
from tab_surface import add_tab_features, make_train_fn
from harness import run_experiment, log_committee_result, current_best, LOG

EPS = 0.002

# measured by tab_committee_check.py (31 Aug), singles logged as
# 'R33c committee completion RICH + tab_n (5 seeds)'
R33C_VALID = {'GAUC': 0.6948, 'nDCG@5': 0.5464, 'primary': 0.62059}
R33C_TEST = {'GAUC': 0.6857, 'nDCG@5': 0.5429, 'primary': 0.61429}


def add_more_features(splits):
    """dur_n (duration-bucket familiarity) and tab_n_fine (finer buckets),
    both label-free counts, self-exclusive, chronological."""
    rows = [x for rws in splits.values() for x in rws]
    rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    dur_ct, tab_ct = {}, {}
    for x in rows:
        kd = (x['user_id'], x['dur_bucket'])
        n = dur_ct.get(kd, 0)
        x['dur_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10'
                      if n <= 10 else '11-30' if n <= 30 else '31-100'
                      if n <= 100 else '100+')
        dur_ct[kd] = n + 1
        kt = (x['user_id'], x['tab'])
        m = tab_ct.get(kt, 0)
        x['tab_n_fine'] = ('0' if m == 0 else '1' if m == 1 else '2-3'
                           if m <= 3 else '4-6' if m <= 6 else '7-10'
                           if m <= 10 else '11-20' if m <= 20 else '21-50'
                           if m <= 50 else '51-100' if m <= 100 else '100+')
        tab_ct[kt] = m + 1
    return splits


if __name__ == '__main__':
    print("=== CAMPAIGN 5: completion run (promotion of the loop's "
          "discovery) ===\n")
    already_banked = any(
        json.loads(l).get('name') == 'R33c RICH + tab_n 5-seed committee (BANKED)'
        for l in open(LOG)) if os.path.exists(LOG) else False
    if already_banked:
        print("iteration 1: R33c banking record already in LOG (idempotent "
              "rerun after the dur_bucket crash); not re-logging.")
    else:
        print("iteration 1: banking R33c (committee numbers measured by "
              "tab_committee_check.py, pre-committed rule in tab_surface.py)")
        log_committee_result(
            name='R33c RICH + tab_n 5-seed committee (BANKED)',
            valid=R33C_VALID, test=R33C_TEST,
            config={'fields': 'RICH + tab_n', 'k': 16, 'lr': 0.001, 'K': 4,
                    'seeds': 5, 'hypothesis_id': 'residual_tab_0',
                    'origin': 'agent v2-loop iteration, 31 Aug'},
            note='Promotion per the pre-committed rule in tab_surface.py: '
                 'committee valid 0.62059 > 0.61906 + 0.001. Hypothesis and '
                 'mechanism control are the agent session\'s (R33b + '
                 'placebo); committee evaluation operator-completed after '
                 'the driver fault (see PROCESS-AUDIT section 9).')
    best = current_best()
    print(f"banked best is now {best:.5f}\n")

    print("building features for iterations 2-4 ...")
    splits = add_tab_features(build_features('continuous'))
    encode_rows(splits, RICH)   # materializes dur_bucket on every row
    splits = add_more_features(splits)
    misses = 0

    print("\niteration 2: R34 recency on top of familiarity")
    fn, dim, _ = make_train_fn(splits,
                               RICH + ['tab_n', 'tab_prev1', 'tab_hist10'])
    r34 = run_experiment(
        name='R34 RICH + tab_n + tab recency',
        hypothesis='Per-surface recency (tab_prev1/tab_hist10) adds signal '
                   'on top of per-surface familiarity (tab_n).',
        rationale='R33a (recency+familiarity) trailed R33b (familiarity '
                  'only) on 3 seeds; direct test against the new champion '
                  'before accepting that recency is redundant.',
        train_fn=fn, seeds=3,
        config={'fields': 'RICH+tab_n+tab_prev1+tab_hist10', 'dim': dim})
    misses += (r34['valid_mean'] <= best + EPS)

    print("\niteration 3: R35 duration-bucket familiarity (next residual "
          "slice, dur=9)")
    fn, dim, _ = make_train_fn(splits, RICH + ['tab_n', 'dur_n'])
    r35 = run_experiment(
        name='R35 RICH + tab_n + dur_n',
        hypothesis='Familiarity with a duration bucket predicts long_view '
                   'propensity on it, as surface familiarity did.',
        rationale='dur=9 is the next slice on the residual queue after '
                  'tab=0; dur_n mirrors the tab_n construction.',
        train_fn=fn, seeds=3,
        config={'fields': 'RICH+tab_n+dur_n', 'dim': dim})
    misses += (r35['valid_mean'] <= best + EPS)

    print("\niteration 4: R36 finer tab_n buckets")
    fn, dim, _ = make_train_fn(splits, RICH + ['tab_n_fine'])
    r36 = run_experiment(
        name='R36 RICH + tab_n_fine',
        hypothesis='The log-bucketing of tab_n discards resolution the '
                   'model could use.',
        rationale='Cheapest refinement of the just-banked feature; a fair '
                  'sub-epsilon candidate to close the convergence window.',
        train_fn=fn, seeds=3,
        config={'fields': 'RICH+tab_n_fine', 'dim': dim})
    misses += (r36['valid_mean'] <= best + EPS)

    print(f"\n=== CONVERGENCE CHECK (epsilon {EPS}, N=3) ===")
    for r in (r34, r35, r36):
        print(f"  {r['name'][:40]:<42} valid {r['valid_mean']:.5f} "
              f"(vs banked {r['valid_mean'] - best:+.5f})")
    if misses >= 3:
        print(f"\nCONVERGED: 3 consecutive iterations without a validation "
              f"improvement > {EPS}. Checkpoint: R33c committee "
              f"(valid {best:.5f} / test {R33C_TEST['primary']:.5f}).")
    else:
        print(f"\nNOT CONVERGED: {3 - misses} iteration(s) improved by "
              f"more than epsilon; the run must continue.")
