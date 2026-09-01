"""R37: discriminating control for tab_n's mechanism.

The earlier time-shuffle control destroyed both surface identity and counting
structure, so it could not tell which one explained tab_n's gain. This control
isolates them by shuffling surface labels within each user and split before
recomputing tab_n.

This preserves chronological counting structure and per-user surface
frequencies while breaking the link between each count and the true surface.
The original `tab` feature remains unchanged.

If the gain collapses, true surface-specific familiarity matters; if it
survives, the benefit is mainly from the partitioned counting structure.
The interpretation is pre-committed against the R33 control and tab_n results.
"""
import collections
import numpy as np
from staleness_ablation import build_features, RICH
from tab_surface import add_tab_features, make_train_fn, split_of
from harness import run_experiment


def add_scrambled_tab_n(splits, seed=0):
    rng = np.random.default_rng(seed)
    rows = [x for rws in splits.values() for x in rws]
    rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))

    by_user = collections.defaultdict(list)
    for x in rows:
        by_user[x['user_id']].append(x)

    for u, rws in by_user.items():
        pseudo = {}
        for spl in ('train', 'valid', 'test'):
            idx = [i for i, x in enumerate(rws) if split_of(x) == spl]
            tabs = [rws[i]['tab'] for i in idx]
            perm = rng.permutation(len(tabs))
            for j, i in enumerate(idx):
                pseudo[i] = tabs[perm[j]]
        ct = {}
        for i, x in enumerate(rws):
            k = pseudo[i]
            n = ct.get(k, 0)
            x['tab_n_scr'] = ('0' if n == 0 else '1-3' if n <= 3
                              else '4-10' if n <= 10 else '11-30' if n <= 30
                              else '31-100' if n <= 100 else '100+')
            ct[k] = n + 1
    return splits


if __name__ == '__main__':
    print("building features + scrambled-surface tab_n ...")
    splits = add_scrambled_tab_n(add_tab_features(build_features('continuous')))

    fn, dim, _ = make_train_fn(splits, RICH + ['tab_n_scr'])
    rec = run_experiment(
        name='R37 RICH + surface-scrambled tab_n (mechanism control)',
        hypothesis='If tab_n pays through SURFACE FAMILIARITY, a count with '
                   'identical chronological structure over scrambled '
                   'surfaces must lose the gain; if it pays through '
                   'counting structure alone, the gain survives.',
        rationale='The R33 time-shuffle placebo destroyed familiarity and '
                  'counting structure together; this control destroys only '
                  'the surface-specificity. Requested by review round 9.',
        train_fn=fn, seeds=3,
        config={'fields': 'RICH+tab_n_scr', 'dim': dim,
                'control_of': 'R33b/R33c (tab_n)'})

    ctrl, tab = 0.61715, 0.61955      # R33-ctrl and R33b, logged 31 Aug
    gain = tab - ctrl
    survived = rec['valid_mean'] - ctrl
    print(f"\nreference gain (R33b - R33-ctrl): {gain:+.5f}")
    print(f"scrambled arm vs control: {survived:+.5f} "
          f"({survived / gain:.0%} of the gain survives)")
    if survived < 0.5 * gain:
        print("VERDICT: collapse. The count must track the row's actual "
              "surface to pay — surface familiarity is the mechanism.")
    else:
        print("VERDICT: survives. Counting structure, not surface identity, "
              "carries the gain; the mechanism story must be revised to "
              "position/frequency, and the docs updated accordingly.")