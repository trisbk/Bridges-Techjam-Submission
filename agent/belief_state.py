"""Pillar 3a: persistent structured memory for the research agent.

Stores hypotheses, evidence, mechanism tags, controls, expected value, and
status across independent agent sessions instead of relying on markdown logs.

A key rule is enforced directly in code: mechanism-based hypotheses cannot be
confirmed unless they have a passing falsification control. The API supports
proposing hypotheses, attaching evidence and controls, and promoting or
refuting beliefs.
"""
import json, os, sys

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'belief_state.json')


class ControlRequired(Exception):
    pass


def load(path=PATH):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {'hypotheses': []}


def save(state, path=PATH):
    with open(path, 'w') as fh:
        json.dump(state, fh, indent=1, default=float)


def propose(state, hid, claim, mechanism='none', expected_value=None,
            cost_estimate_s=None):
    for h in state['hypotheses']:
        if h['id'] == hid:
            return h                     # no duplicates
    h = {'id': hid, 'claim': claim, 'mechanism': mechanism,
         'status': 'proposed', 'evidence': [], 'control': None,
         'expected_value': expected_value, 'cost_estimate_s': cost_estimate_s}
    state['hypotheses'].append(h)
    return h


def get(state, hid):
    for h in state['hypotheses']:
        if h['id'] == hid:
            return h
    raise KeyError(hid)


def attach_evidence(state, hid, result):
    h = get(state, hid)
    h['evidence'].append(result)
    h['status'] = 'testing'


def attach_control(state, hid, control_type, outcome, detail=''):
    """outcome: 'passed' (behaved as the mechanism predicts) or 'failed'."""
    h = get(state, hid)
    h['control'] = {'type': control_type, 'outcome': outcome, 'detail': detail}


def promote(state, hid):
    h = get(state, hid)
    if h['mechanism'] != 'none':
        c = h['control']
        if c is None:
            raise ControlRequired(
                f"{hid}: mechanism '{h['mechanism']}' claimed but no "
                f"falsification control recorded — run controls.py first")
        if c['outcome'] != 'passed':
            raise ControlRequired(
                f"{hid}: control ran and FAILED — the mechanism story is "
                f"wrong; refute or retag instead of promoting")
    h['status'] = 'confirmed'


def refute(state, hid, by_control=False):
    h = get(state, hid)
    h['status'] = 'refuted_by_control' if by_control else 'refuted'


def next_open(state):
    """Highest expected-value-per-second open hypothesis."""
    opens = [h for h in state['hypotheses']
             if h['status'] in ('proposed', 'testing')]
    def score(h):
        ev = h['expected_value'] if h['expected_value'] is not None else 1e-3
        cost = h['cost_estimate_s'] if h['cost_estimate_s'] else 300.0
        bonus = 1.2 if (h['mechanism'] != 'none' and h['control'] is None) else 1.0
        return (ev / cost) * bonus
    return max(opens, key=score) if opens else None


def _selftest():
    st = {'hypotheses': []}
    propose(st, 'h1', 'cold users need a prior', 'temporal', 0.004, 120)
    propose(st, 'h1', 'duplicate attempt', 'temporal')             # ignored
    assert len(st['hypotheses']) == 1
    attach_evidence(st, 'h1', {'test': 0.6120, 'valid': 0.6195})
    try:
        promote(st, 'h1'); raise AssertionError('promoted without control')
    except ControlRequired:
        pass
    attach_control(st, 'h1', 'time_shuffle', 'failed', 'gain survived shuffle')
    try:
        promote(st, 'h1'); raise AssertionError('promoted on failed control')
    except ControlRequired:
        pass
    attach_control(st, 'h1', 'time_shuffle', 'passed', 'gain collapsed 82%')
    promote(st, 'h1')
    assert get(st, 'h1')['status'] == 'confirmed'
    propose(st, 'h2', 'mechanism-free bookkeeping change', 'none')
    promote(st, 'h2')                                              # allowed
    propose(st, 'h3', 'evidence-backed idea', 'capacity', 0.005, 100)
    propose(st, 'h4', 'hand-written idea', 'none')                 # flat prior
    assert next_open(st)['id'] == 'h3'
    save(st, '/tmp/bs_test.json'); st2 = load('/tmp/bs_test.json')
    assert st2 == st
    os.remove('/tmp/bs_test.json')
    print("selftest OK: dedup, control-gated promotion (missing AND failed), "
          "mechanism-free promotion, EV/cost ordering, save/load round-trip")


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    elif '--next' in sys.argv:
        h = next_open(load())
        print(json.dumps(h, indent=1) if h else 'no open hypotheses')
    else:
        print(__doc__)