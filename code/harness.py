"""Research-agent harness: the single entry point for running an experiment.

Every hypothesis goes through run_experiment(), which mechanically enforces
the lab discipline — no iteration can skip it:

  1. multi-seed training (default 3) with mean/std,
  2. significance gating against the frozen baseline AND the current best
     (gate = 0.002, ~2.5x measured seed noise sigma ~= 0.0008),
  3. self-documenting append to experiments/LOG.jsonl (machine-readable) —
     hypothesis, rationale, config, per-seed results, verdict, wall time —
     BEFORE and AFTER the run, so a killed run still leaves its intent.

The test split is evaluated and recorded but the VERDICT, the banked-best
tracking, and the convergence signal are all decided on VALIDATION; test
numbers are reported for the log's honesty and audited at final scoring.
(Process note: before 30 Aug the printed verdict label was derived from the
test delta while banking decisions were made on validation by the operator
and documented in RESULTS.md; see logs/PROCESS-AUDIT.md. This file now
computes everything selection-relevant from validation.) Training code can
never see valid/test labels (only evaluate() reads them).
"""
import json, os, time
import numpy as np

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'logs', 'LOG.jsonl')
BASELINE = 0.5950          # reproduced FM baseline, TEST primary (reporting only)
BASELINE_VALID = 0.6014    # reproduced FM baseline, VALIDATION primary (selection)
GATE = 0.002               # significance bar vs baseline; seed noise sigma ~= 0.0008
PROMOTION_MARGIN = 0.001   # a challenger must beat the incumbent's validation
                           # by more than noise; ties and sub-noise diffs keep
                           # the incumbent (the documented banking rule)
RULE_TAG = 'valid-v2'      # stamped on records written under the corrected rule
BANKED_VALID = 0.62059     # the frozen champion's validation primary (R33c,
                           # campaign 5; R24b's 0.61906 until 31 Aug). The
                           # driver reads this for its convergence baseline;
                           # current_best() uses it as the floor.


def _append(rec):
    with open(LOG, 'a') as fh:
        fh.write(json.dumps(rec) + '\n')


def current_best():
    """Best banked VALIDATION primary recorded so far. Only records written
    under the corrected rule (RULE_TAG) count; the floor is BANKED_VALID,
    the validation of the frozen champion. Legacy records keep their
    original labels as display markers (see logs/PROCESS-AUDIT.md and
    replay_verdicts.py) and do not feed selection."""
    best = BANKED_VALID  # the frozen champion (R33c committee, campaign 5;
                         # the prior floor was R24b's 0.61906)
    if os.path.exists(LOG):
        with open(LOG) as fh:
            for line in fh:
                r = json.loads(line)
                if (r.get('phase') == 'result' and r.get('verdict') == 'WIN'
                        and r.get('rule') == RULE_TAG
                        and r.get('valid_mean') is not None):
                    best = max(best, r['valid_mean'])
    return best


def log_committee_result(name, valid, test, config=None, note=''):
    """Bank a committee-level evaluation as a first-class, rule-tagged
    record. Committee results (z-scored prediction averages over already
    logged singles) were previously written in a legacy format without the
    rule tag, so they could not feed current_best(); this closes that gap.
    The verdict is computed exactly as in run_experiment, from VALIDATION.
    `valid`/`test` are evaluate() dicts for the committee predictions."""
    vm, tm = float(valid['primary']), float(test['primary'])
    best = current_best()
    d_base = vm - BASELINE_VALID
    if d_base > GATE and vm > best + PROMOTION_MARGIN:
        verdict = 'WIN'
    elif d_base > GATE:
        verdict = 'SIGNIFICANT_BUT_NOT_BEST'
    elif d_base < -GATE:
        verdict = 'WORSE'
    else:
        verdict = 'NOISE'
    rec = {'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
           'name': name, 'valid_mean': round(vm, 5), 'test_mean': round(tm, 5),
           'test_std': None, 'committee': True,
           'valid_GAUC': round(float(valid['GAUC']), 5),
           'valid_nDCG@5': round(float(valid['nDCG@5']), 5),
           'test_GAUC': round(float(test['GAUC']), 5),
           'test_nDCG@5': round(float(test['nDCG@5']), 5),
           'd_baseline_valid': round(d_base, 5),
           'd_best_valid': round(vm - best, 5),
           'verdict': verdict, 'rule': RULE_TAG,
           'config': config or {}, 'note': note}
    _append(rec)
    print(f"[{verdict}] {name}: committee valid {vm:.5f} "
          f"(vs banked best {vm - best:+.5f}) | test {tm:.5f} recorded")
    return rec


def run_experiment(name, hypothesis, rationale, train_fn, seeds=3, config=None):
    """train_fn(seed) -> {'valid': {...}, 'test': {...}} from evaluate().

    Returns the result record. Appends intent before training and the result
    after, so an interrupted run still documents what it was trying.
    """
    t0 = time.time()
    _append({'phase': 'intent', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': name, 'hypothesis': hypothesis, 'rationale': rationale,
             'config': config or {}, 'seeds': seeds})
    vs, ts = [], []
    comps = {'valid_GAUC': [], 'valid_nDCG@5': [], 'test_GAUC': [], 'test_nDCG@5': []}
    for s in range(seeds):
        r = train_fn(s)
        vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
        comps['valid_GAUC'].append(r['valid'].get('GAUC'))
        comps['valid_nDCG@5'].append(r['valid'].get('nDCG@5'))
        comps['test_GAUC'].append(r['test'].get('GAUC'))
        comps['test_nDCG@5'].append(r['test'].get('nDCG@5'))
    vm, tm, tsd = float(np.mean(vs)), float(np.mean(ts)), float(np.std(ts))
    best = current_best()
    d_base = vm - BASELINE_VALID          # VALIDATION delta decides the verdict
    d_best = vm - best
    d_base_test = tm - BASELINE           # test delta, recorded for audit only
    if d_base > GATE and vm > best + PROMOTION_MARGIN:
        verdict = 'WIN'
    elif d_base > GATE:
        verdict = 'SIGNIFICANT_BUT_NOT_BEST'
    elif d_base < -GATE:
        verdict = 'WORSE'
    else:
        verdict = 'NOISE'
    def _m(key):
        xs = [x for x in comps[key] if x is not None]
        return round(float(np.mean(xs)), 5) if xs else None

    rec = {'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
           'name': name, 'valid_mean': round(float(vm), 5),
           'test_mean': round(float(tm), 5), 'test_std': round(float(tsd), 5),
           'test_per_seed': [round(float(x), 5) for x in ts],
           'valid_GAUC': _m('valid_GAUC'), 'valid_nDCG@5': _m('valid_nDCG@5'),
           'test_GAUC': _m('test_GAUC'), 'test_nDCG@5': _m('test_nDCG@5'),
           'd_baseline_valid': round(float(d_base), 5),
           'd_baseline_test': round(float(d_base_test), 5),
           'd_best_valid': round(float(d_best), 5),
           'verdict': verdict, 'rule': RULE_TAG,
           'wall_s': round(time.time() - t0, 1)}
    _append(rec)
    mark = '✅ BETTER than banked' if vm > best else '❌ not better'
    print(f"[{verdict}] {name}: valid {vm:.4f} (vs banked {vm - best:+.4f} {mark}) "
          f"| test {tm:.4f} ± {tsd:.4f} recorded | {rec['wall_s']}s)")
    return rec
