"""Research-agent harness: the common entry point for every experiment.

Each run goes through the same process: train across multiple seeds, evaluate
against the validation baseline/current best, and log the hypothesis, config,
results, verdict, and runtime to experiments/LOG.jsonl.

Test scores are recorded for transparency, but all model selection, banking,
and convergence decisions are based only on validation. See
logs/PROCESS-AUDIT.md for the historical verdict-label correction.
"""

import json, os, time
import numpy as np

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', 'logs', 'LOG.jsonl')
BASELINE = 0.5950          # reproduced FM baseline, TEST primary (reporting only)
BASELINE_VALID = 0.6014    # reproduced FM baseline, VALIDATION primary (selection)
GATE = 0.002               # significance bar vs baseline; seed noise sigma ~ 0.0008
PROMOTION_MARGIN = 0.001   # a challenger must beat the incumbent's validation
                           # by more than noise
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
    the validation of the frozen champion."""
    best = BANKED_VALID  
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