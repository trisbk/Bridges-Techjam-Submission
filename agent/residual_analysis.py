"""Pillar 3b: generate hypotheses from the model's validation errors.

Slices the frozen model's validation predictions by existing attributes such
as history depth, surface, and duration, then uses the official evaluator to
identify where meaningful ranking headroom remains. The resulting slices are
written into the belief state as structured hypotheses.

Three headroom measures are tracked:
- EV: slice performance gap weighted by user share; retained for history but
  biased by slice-specific user/label degeneracy.
- EVo: oracle gain from perfectly ranking the slice inside the full validation
  set, avoiding that degeneracy but tending to favor large slices.
- EVx: excess oracle headroom after subtracting a per-user shape-matched random
  slice, isolating unusually poor model performance from slice size alone.

Campaign 6 showed that EVo could prioritize large but already well-performing
slices, so the autonomous loop revised its research instrument and now ranks
open hypotheses by EVx. Earlier resolved records retain their original EVo
values for provenance.
"""
import collections, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'code'))
import numpy as np
from evaluate import evaluate


def oracle_headroom(users, labels, scores, idx, overall=None):
    """Overall-primary gain from ranking the rows in `idx` with oracle
    knowledge of their labels, everything else untouched. Upper bound on
    what any intervention targeted at this slice can be worth."""
    if overall is None:
        overall = float(evaluate(users, labels, scores)['primary'])
    sc = np.asarray(scores, dtype=float)
    big = float(np.abs(sc).max()) + 1.0
    patched = sc.copy()
    for i in idx:
        # oracle placement, with the model's own score as a tie-break so the
        # patch never reorders rows the oracle is indifferent about
        patched[i] = (big if labels[i] else -big) + 1e-6 * sc[i]
    return float(evaluate(users, labels, list(patched))['primary']) - overall


def matched_null_headroom(users, labels, scores, idx, overall, seeds=2):
    """Estimate oracle headroom using a shape-matched null slice.

    For each user, the null samples the same number of rows as the real slice,
    preserving which users appear and their contribution sizes while randomizing
    which of their rows are selected. This isolates whether the slice's specific
    row assignment carries meaningful signal beyond its shape.

    Returns the mean null headroom and `locked_share`, the fraction of slice rows
    from users whose rows cannot be reshuffled. High locked_share makes the
    estimate less reliable and should be interpreted cautiously.
    """
    by_user = collections.defaultdict(list)
    for i, u in enumerate(users):
        by_user[u].append(i)
    want = collections.Counter(users[i] for i in idx)
    locked = sum(c for u, c in want.items() if c == len(by_user[u]))
    vals = []
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        null = []
        for u, c in want.items():
            rows = by_user[u]
            null.extend(int(i) for i in
                        rng.choice(rows, size=c, replace=False))
        vals.append(oracle_headroom(users, labels, scores, null, overall))
    return float(np.mean(vals)), float(locked / max(1, len(idx)))


def slice_report(users, labels, scores, tags, min_users=50, null_seeds=2):
    overall = float(evaluate(users, labels, scores)['primary'])
    total_users = len(set(users))
    out = []
    keys = tags[0].keys()
    for k in keys:
        vals = sorted({t[k] for t in tags})
        for v in vals:
            idx = [i for i, t in enumerate(tags) if t[k] == v]
            su = [users[i] for i in idx]
            n_u = len(set(su))
            if n_u < min_users:
                continue
            r = evaluate(su, [labels[i] for i in idx],
                         [scores[i] for i in idx])['primary']
            ev = float(max(0.0, overall - r) * (n_u / total_users))
            evo = oracle_headroom(users, labels, scores, idx, overall)
            null, locked = matched_null_headroom(
                users, labels, scores, idx, overall, seeds=null_seeds)
            if locked > 0.25:
                print(f"  [warn] {k}={v}: {locked:.0%} of the slice's rows "
                      f"belong to users who are entirely inside it, so the "
                      f"null cannot reshuffle them and this EVx is biased "
                      f"toward zero")
            pos = {}
            for i in idx:
                u = users[i]
                pos[u] = pos.get(u, 0) + int(labels[i])
            degen = sum(1 for c in pos.values() if c == 0) / len(pos)
            out.append((f"{k}={v}", float(r), n_u, ev, float(evo),
                        float(degen), float(evo - null)))
    out.sort(key=lambda x: -x[6])
    return overall, out


RESOLVED = ('confirmed', 'refuted', 'refuted_by_control')


def first_unresolved(rep, state):
    resolved = {h['id'] for h in state.get('hypotheses', [])
                if h.get('status') in RESOLVED}
    for row in rep:
        if f"residual_{row[0].replace('=', '_')}" not in resolved:
            return row
    return None


MECHANISM_HINT = {
    'hist': 'temporal',    # history-depth slices point at sequence starvation
    'tab': 'none',         # surface slices
    'dur': 'capacity',     # duration slices
}


def to_hypothesis(slice_name, slice_primary, overall, n_users, ev,
                  ev_oracle=None, degen=None, ev_excess=None):
    key = slice_name.split('=')[0]
    extra = ''
    if ev_oracle is not None:
        extra = (f"; oracle headroom on the overall metric {ev_oracle:.4f}"
                 + (f", of which {ev_excess:.4f} is EXCESS over a matched "
                    f"null slice (same users, same per-user row counts, rows "
                    f"drawn at random) and therefore attributable to the "
                    f"model rather than to slice size"
                    if ev_excess is not None else '')
                 + (f"; {degen:.0%} of the slice's users are all-negative "
                    f"inside it, so the slice-restricted score understates "
                    f"the model" if degen is not None else ''))
    return {
        'id': f"residual_{slice_name.replace('=', '_')}",
        'claim': (f"slice {slice_name} scores {slice_primary:.4f} vs overall "
                  f"{overall:.4f} across {n_users} users; a feature or prior "
                  f"targeted at this slice should close part of the gap"
                  + extra),
        'mechanism': MECHANISM_HINT.get(key, 'none'),
        'expected_value': round(
            ev_excess if ev_excess is not None
            else ev_oracle if ev_oracle is not None else ev, 5),
    }


def _selftest():
    rng = np.random.default_rng(0)
    users, labels, scores, tags = [], [], [], []
    for u in range(200):
        for i in range(12):
            cold = i < 2                    
            y = int(rng.random() < 0.4)
            s = y * 2.0 + rng.normal()      
            if cold:
                s = rng.normal()            
            users.append(f"u{u}"); labels.append(y); scores.append(float(s))
            tags.append({'hist': 'cold' if cold else 'warm',
                         'tab': str(i % 2)})
    
    overall, rep = slice_report(users, labels, scores, tags, min_users=20,
                                null_seeds=8)
    worst = rep[0]
    assert worst[0] == 'hist=cold', f"expected hist=cold worst, got {worst[0]}"
    assert worst[4] > 0, "oracle headroom must be positive for a broken slice"
    h = to_hypothesis(worst[0], worst[1], overall, worst[2], worst[3],
                      worst[4], worst[5], worst[6])
    assert h['mechanism'] == 'temporal'
    assert abs(h['expected_value'] - round(worst[6], 5)) < 1e-9, \
        "hypothesis EV must be the EXCESS oracle headroom (v3)"

    
    by = {r[0]: r for r in rep}
    evo_rank = [r[0] for r in sorted(rep, key=lambda x: -x[4])]
    assert evo_rank[0] != 'hist=cold' and evo_rank[-1] == 'hist=cold', \
        f"fixture no longer exercises the size bias: EVo order {evo_rank}"
    assert rep[0][0] == 'hist=cold', \
        f"EVx must rank the broken slice first, got {[r[0] for r in rep]}"

    for name in ('tab=0', 'tab=1', 'hist=warm'):
        assert by[name][4] > by['hist=cold'][4], \
            f"{name} should out-size hist=cold on EVo"
        assert abs(by[name][6]) < 0.25 * by['hist=cold'][6], \
            f"{name} is a size artifact: its EVx must be near zero"

    assert by['hist=warm'][6] < 0 < by['hist=warm'][4], \
        "a slice the model is unusually good on must have negative EVx"
    _, locked = matched_null_headroom(
        users, labels, scores,
        [i for i, t in enumerate(tags) if t['hist'] == 'cold'], overall)
    assert locked == 0.0, \
        f"fixture slices must vary within user for EVx to be meaningful "\
        f"(locked share {locked})"

    du, dl, ds, dt = [], [], [], []
    for u in range(120):
        for i in range(6):
            y = int(i == 0 and u < 60)         
            thin = (i >= 4)
            du.append(f"v{u}"); dl.append(0 if thin else y)
            ds.append(float(rng.normal() + (0 if thin else 2.0 * y)))
            dt.append({'grp': 'thin' if thin else 'main'})
    _, rep2 = slice_report(du, dl, ds, dt, min_users=20)
    thin_row = [r for r in rep2 if r[0] == 'grp=thin'][0]
    assert thin_row[5] == 1.0, "all-negative slice should be flagged degenerate"
    assert thin_row[4] == 0.0 or thin_row[4] < thin_row[3], \
        "oracle headroom must not credit a slice with nothing to gain"
    st = {'hypotheses': [{'id': 'residual_hist_cold', 'status': 'refuted'}]}
    nxt = first_unresolved(rep, st)
    assert nxt is not None and nxt[0] != 'hist=cold', \
        "a resolved slice must not be re-proposed"
    assert first_unresolved(rep, {'hypotheses': []})[0] == 'hist=cold', \
        "with an empty belief state the worst slice is still the pick"
    allres = {'hypotheses': [{'id': f"residual_{r[0].replace('=', '_')}",
                              'status': 'confirmed'} for r in rep]}
    assert first_unresolved(rep, allres) is None, \
        "fully resolved report must yield no proposal"

    print(f"selftest OK: found broken slice '{worst[0]}' "
          f"(primary {worst[1]:.3f} vs overall {overall:.3f}, "
          f"EV {worst[3]:.4f}, oracle headroom {worst[4]:.4f} -- which "
          f"ranks it LAST of {len(rep)} -- of which excess over the matched "
          f"null {worst[6]:.4f}, ranking it first; the larger healthy slices "
          f"keep at most "
          f"{max(abs(by[n][6]) for n in by if n != 'hist=cold') / worst[6]:.0%}"
          f" of that), "
          f"suggested mechanism '{h['mechanism']}'; degenerate slice "
          f"(EV {thin_row[3]:.4f}) correctly given oracle headroom "
          f"{thin_row[4]:.4f}")


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
        sys.exit(0)
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code'))
    import collections, csv
    from baseline import FM
    from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA
    import belief_state as BS_import  # noqa
    sys.path.insert(0, os.path.join('..', 'agent'))
    import belief_state as BS

    print("building features + scoring frozen committee on validation ...")
    from staleness_ablation import build_features, RICH
    from tab_surface import add_tab_features
    splits = add_tab_features(build_features('continuous'))
    enc, dim = encode_rows(splits, RICH + ['tab_n'])
    Xva, yva, uva = enc['valid']
    frozen = None
    for c in ('frozen_model', os.path.join('..', 'final_output',
                                           'frozen_model')):
        if os.path.exists(os.path.join(c, 'fm_seed0.npz')):
            frozen = c
            break
    if frozen is None:
        raise FileNotFoundError('frozen_model (current champion) not found')
    preds = []
    for seed in range(5):
        z = np.load(os.path.join(frozen, f'fm_seed{seed}.npz'))
        m = FM(dim, k=16, seed=seed)
        m.V, m.W, m.b = z['V'], z['W'], np.float32(z['b'])
        p = m.predict(Xva)
        preds.append((p - p.mean()) / p.std())
    scores = list(np.mean(preds, 0))

    tags = [{'hist': x['hist_n'], 'tab': x['tab'], 'dur': x['dur_bucket']}
            for x in splits['valid']]
    overall, rep = slice_report(uva, list(yva), scores, tags,
                                null_seeds=3)
    print(f"\noverall validation primary {overall:.5f}; worst slices "
          f"(ordered by EVx, the headroom the MODEL is responsible for):")
    print(f"  {'slice':<14} {'primary':>9} {'users':>7} {'EV(old)':>9} "
          f"{'EVo':>8} {'EVx':>8} {'all-neg users':>14}")
    for name, r, n_u, ev, evo, degen, evx in rep[:8]:
        print(f"  {name:<14} {r:>9.5f} {n_u:>7,} {ev:>9.5f} {evo:>8.5f} "
              f"{evx:>8.5f} {degen:>13.0%}")

    import json
    with open(os.path.join('..', 'agent', 'residual_report.json'), 'w') as fh:
        json.dump({'overall': overall,
                   'slices': [{'slice': n, 'primary': r, 'users': nu,
                               'ev_old': ev, 'ev_oracle': evo,
                               'ev_excess': evx,
                               'allneg_user_share': dg}
                              for n, r, nu, ev, evo, dg, evx in rep]},
                  fh, indent=1)

    st = BS.load()
    worst = first_unresolved(rep, st)
    if worst is None:
        print("\nevery slice above the user floor is already resolved in the "
              "belief state; nothing new to propose.")
        raise SystemExit(0)
    if worst is not rep[0]:
        print(f"\n(top slice {rep[0][0]} is already resolved; taking the "
              f"next unresolved one)")
    h = to_hypothesis(worst[0], worst[1], overall, worst[2], worst[3],
                      worst[4], worst[5], worst[6])
    BS.propose(st, h['id'], h['claim'], h['mechanism'], h['expected_value'])
    BS.save(st)
    print(f"\nwritten to belief state: {h['id']} (mechanism {h['mechanism']}, "
          f"EV {h['expected_value']})")