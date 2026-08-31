"""Post-mortem for Run 39 (campaign 6, iteration 3): why the partition
exposure counts lost, measured with the SAME instrument that iteration 1 used
on `sess_pos`.

Label-free with respect to the eval windows: every rate is estimated on TRAIN
rows and then mapped onto validation rows. Validation/test labels are never
read. No model is trained here.

Iteration 1 refuted the cheap pre-run screen "rank candidate features by novel
within-user prior variance" by showing the *loser* (`sess_pos`) carried 0.79x
the *winner*'s (`tab_n`) novel within-user signal. Run 39 supplies a second,
independent test of the same screen from a different construction, and this
module measures it on the same axis so the two are directly comparable:

  novel(target, given) = within-user variance of the target's train-estimated
  prior on valid rows, before and after least-squares regression on the
  within-user-centered priors of `given`.

Three numbers, one instrument:
  tag_n  vs the champion's shipped fields (the loser, -0.00133)
  auth_n vs the champion's shipped fields (the second arm's addition)
  tab_n  vs RICH                          (the winner, +0.0024)

Also prints how much of `tag_n`'s marginal rate variance the shipped
`tag_hist` absorbs, which is the redundancy reading the conditional table in
`logs/familiarity_diagnostic.out` already argued against.

Run from code/:  python3 partition_postmortem.py     (~2 min, no training)
"""
import collections
import numpy as np
from staleness_ablation import build_features
from tab_surface import add_tab_features
from familiarity_diagnostics import add_familiarity_features
# the identical instrument iteration 1 used, imported rather than rewritten
from session_depth_diagnostics import rate_table, absorbed, prior, SHIPPED


def main():
    splits = add_familiarity_features(
        add_tab_features(build_features('continuous')))
    tr, va = splits['train'], splits['valid']
    idx = collections.defaultdict(list)
    for i, x in enumerate(va):
        idx[x['user_id']].append(i)
    slice_rows = [i for i, x in enumerate(va) if x['hist_n'] == '31-100']
    print(f"train {len(tr):,} rows | valid {len(va):,} rows, "
          f"{len(idx):,} users | valid rows in hist_n=31-100 "
          f"{len(slice_rows):,}")

    def center(v):
        out = v.copy()
        for ii in idx.values():
            out[ii] -= out[ii].mean()
        return out

    def novel(target, given):
        y = center(prior(tr, va, target))
        X = np.column_stack([center(prior(tr, va, f)) for f in given]
                            + [np.ones(len(y))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        return float(np.mean(y ** 2)), float(np.mean(res ** 2))

    print("\n[1] within-user prior variance, and what survives regression on"
          " the champion's other fields (the instrument iteration 1"
          " falsified as a screen):")
    rows = [('tag_n', SHIPPED, 'the loser,  -0.00133 (R39a)'),
            ('auth_n', SHIPPED + ['tag_n'], 'the R39b addition, +0.00024 vs R39a'),
            ('tab_n', [f for f in SHIPPED if f != 'tab_n'],
             'the winner, +0.0024 (R33b)')]
    out = {}
    for feat, given, tag in rows:
        t, r = novel(feat, given)
        out[feat] = (t, r)
        print(f"    {feat:<7} {t:.3e} -> {r:.3e}  ({r / t:>3.0%} novel, "
              f"R^2 {1 - r / t:.2f})   {tag}")
    ratio = out['tag_n'][1] / out['tab_n'][1]
    print(f"\n    novel within-user signal, loser / winner = {ratio:.2f}x")
    print("    (iteration 1's loser sess_pos scored 0.79x on this same axis"
          " and also lost.)")

    print("\n[2] the redundancy reading, tested: how much of tag_n's marginal"
          " rate variance the shipped outcome count absorbs (train rows):")
    for within in ('tag_hist', 'hist_n', 'hist30', 'tab_n'):
        v, vw = absorbed(tr, 'tag_n', within)
        print(f"    tag_n rate variance {v:.5f} -> {vw:.5f} within "
              f"{within:<9} ({100 * (1 - vw / v):.0f}% absorbed)")
    for within in ('auth_hist', 'hist_n'):
        v, vw = absorbed(tr, 'auth_n', within)
        print(f"    auth_n rate variance {v:.5f} -> {vw:.5f} within "
              f"{within:<9} ({100 * (1 - vw / v):.0f}% absorbed)")

    print("\n[3] support: a feature can only act where it varies. Share of"
          " train rows away from the zero bucket, and share of VALID users"
          " for whom the feature is constant across their own rows (a"
          " within-user metric cannot score a constant):")
    for feat in ('tag_n', 'auth_n', 'tab_n'):
        nz = np.mean([x[feat] != '0' for x in tr])
        s = prior(tr, va, feat)
        const = float(np.mean([len(set(np.round(s[ii], 6))) == 1
                               for ii in idx.values()]))
        print(f"    {feat:<7} non-zero on {nz:>5.1%} of train rows | "
              f"constant within {const:>5.1%} of valid users")

    print("\n[4] the same three novelty numbers restricted to the slice that"
          " raised the hypothesis (hist_n=31-100 valid rows):")
    for feat, given, _ in rows:
        y = center(prior(tr, va, feat))
        X = np.column_stack([center(prior(tr, va, f)) for f in given]
                            + [np.ones(len(y))])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = (y - X @ beta)[slice_rows]
        ys = y[slice_rows]
        print(f"    {feat:<7} {np.mean(ys ** 2):.3e} -> "
              f"{np.mean(res ** 2):.3e} "
              f"({np.mean(res ** 2) / np.mean(ys ** 2):.0%} novel)")


if __name__ == '__main__':
    main()
