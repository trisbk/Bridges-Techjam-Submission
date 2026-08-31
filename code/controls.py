"""Falsification engine: mechanical control synthesis for causal claims.

A search loop asks "did the number go up?" A researcher asks "did it go up
for the reason I claimed?" and builds the experiment that could prove the
claim wrong. This module gives the agent that capability as a mechanism,
not a judgment call: a hypothesis is tagged with WHY it is supposed to
work, and the matching placebo is synthesized automatically.

Mechanism taxonomy and the control each one synthesizes:

  temporal  : "the feature works because of WHEN it is attached" (recency,
              causality). Control: time-shuffle placebo — within each user
              and within each split, permute which impression each feature
              vector is attached to. Every per-user feature marginal is
              preserved (the user's values all survive), the row-to-feature
              alignment is destroyed, and no value crosses a split
              boundary. If the claim is true, the gain should largely
              collapse; if the gain survives, the features act as coarse
              per-user fingerprints and the temporal story is (partly)
              false.
  capacity  : "the feature works because of its information, not because it
              adds parameters." Control: replace values with seeded random
              tokens of matched cardinality per field and retrain — the
              same number of embedding slots with no information.
  objective : "the gain comes from the training objective." Control: the
              capacity/architecture-matched comparison already used in the
              main campaign (Runs 4/12), referenced here for a uniform
              interface.

The engine never touches the designated submission; placebo models are
diagnostic artifacts trained only to be compared.

Self-test (no dataset needed):  python3 controls.py --selftest
"""
import sys
import numpy as np


def time_shuffle(rows, feature_keys, split_of, seed=0):
    """Permute the feature vectors among each user's rows WITHIN a split.

    rows        : list of dicts (each has 'user_id' and the feature keys)
    feature_keys: which keys form the feature vector being reattached
    split_of    : function row -> split name (values never cross splits)
    Returns a new list of dicts (originals untouched).
    """
    rng = np.random.default_rng(seed)
    out = [dict(r) for r in rows]
    groups = {}
    for i, r in enumerate(out):
        groups.setdefault((r['user_id'], split_of(r)), []).append(i)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        perm = rng.permutation(len(idxs))
        vecs = [{k: out[i][k] for k in feature_keys} for i in idxs]
        for slot, j in enumerate(perm):
            for k in feature_keys:
                out[idxs[slot]][k] = vecs[j][k]
    return out


def capacity_noise(rows, feature_keys, seed=0):
    """Replace each feature's value with a random token drawn from that
    feature's own observed vocabulary (matched cardinality, no info)."""
    rng = np.random.default_rng(seed)
    out = [dict(r) for r in rows]
    for k in feature_keys:
        vocab = sorted({r[k] for r in out})
        draws = rng.integers(0, len(vocab), size=len(out))
        for r, d in zip(out, draws):
            r[k] = vocab[int(d)]
    return out


def _selftest():
    rows = []
    for u in ('a', 'b'):
        for i in range(6):
            rows.append({'user_id': u, 'split': 'train' if i < 4 else 'test',
                         'f1': f'{u}{i}', 'f2': f'x{i}', 'y': i % 2})
    shuffled = time_shuffle(rows, ['f1', 'f2'], lambda r: r['split'], seed=1)

    for u in ('a', 'b'):
        for sp in ('train', 'test'):
            orig = sorted(r['f1'] for r in rows
                          if r['user_id'] == u and r['split'] == sp)
            new = sorted(r['f1'] for r in shuffled
                         if r['user_id'] == u and r['split'] == sp)
            assert orig == new, "per-user per-split marginal not preserved"
    moved = sum(1 for a, b in zip(rows, shuffled) if a['f1'] != b['f1'])
    assert moved > 0, "alignment was not broken"
    assert all(a['y'] == b['y'] for a, b in zip(rows, shuffled)), \
        "labels must never move"
    assert all(a['user_id'] == b['user_id'] and a['split'] == b['split']
               for a, b in zip(rows, shuffled)), "rows must stay in place"
    pairs = {('f1', r['f1'], 'f2', r['f2']) for r in shuffled}
    orig_pairs = {('f1', r['f1'], 'f2', r['f2']) for r in rows}
    assert pairs == orig_pairs, "feature vectors must move as units"

    noised = capacity_noise(rows, ['f1'], seed=2)
    assert {r['f1'] for r in noised} <= {r['f1'] for r in rows}, \
        "noise tokens must come from the observed vocabulary"
    print("selftest OK: marginals preserved, alignment broken, no "
          "split-crossing, labels untouched, vectors move as units")


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        print(__doc__)
