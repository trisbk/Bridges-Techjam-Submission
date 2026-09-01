"""Run 38 post-mortem: investigate why session-depth features hurt validation.

Uses train-derived statistics only to test whether the apparent session-fatigue
signal is already captured by the champion's existing features and whether
enough genuinely new within-user information remains for the ranking metrics.

The analysis measures:
- fatigue by within-session position,
- how much existing recency features absorb that signal,
- how much novel within-user variation sess_pos contributes after controlling
  for existing features.

tab_n is analyzed the same way as a reference because it previously produced
a successful improvement. No model training or validation-label analysis
occurs here.
"""
import collections
import numpy as np
from staleness_ablation import build_features
from tab_surface import add_tab_features
from session_depth import add_session_features

ORDER = ['0', '1-2', '3-5', '6-10', '11-20', '21-50', '50+']
SHIPPED = ['gap', 'hist10', 'hist30', 'prev1', 'hist_n', 'auth_hist',
           'tag_hist', 'tab_n']


def rate_table(rows, key, cond=None):
    d = collections.defaultdict(lambda: [0, 0])
    for x in rows:
        k = (x[cond], x[key]) if cond else x[key]
        d[k][0] += x['y']; d[k][1] += 1
    return d


def absorbed(rows, key, within):
    """Weighted variance of `key`'s bucket rates, marginally and within
    strata of `within`. The shrinkage is how much of key's discriminative
    power the shipped feature already holds."""
    d = rate_table(rows, key)
    tot = sum(n for _, n in d.values())
    mu = sum(p for p, _ in d.values()) / tot
    var = sum(n * (p / n - mu) ** 2 for p, n in d.values()) / tot
    dd = rate_table(rows, key, cond=within)
    strat = collections.defaultdict(lambda: [0, 0])
    for (c, _), (p, n) in dd.items():
        strat[c][0] += p; strat[c][1] += n
    v2 = sum(n * (p / n - strat[c][0] / strat[c][1]) ** 2
             for (c, _), (p, n) in dd.items() if n)
    return var, v2 / tot


def prior(tr, va, feat):
    """Each row's train-estimated long_view rate for its own bucket."""
    d = rate_table(tr, feat)
    g = sum(p for p, _ in d.values()) / sum(n for _, n in d.values())
    rate = {k: p / n for k, (p, n) in d.items()}
    return np.array([rate.get(x[feat], g) for x in va])


def main():
    splits = add_session_features(add_tab_features(
        build_features('continuous')))
    tr = [x for x in splits['train'] if x['tab'] == '1']
    va = [x for x in splits['valid'] if x['tab'] == '1']
    users = [x['user_id'] for x in va]
    idx = collections.defaultdict(list)
    for i, u in enumerate(users):
        idx[u].append(i)
    print(f"train tab=1 {len(tr):,} rows | valid tab=1 {len(va):,} rows, "
          f"{len(idx):,} users")

    print("\n[1] long_view rate by sess_pos (tab=1, train):")
    m = rate_table(tr, 'sess_pos')
    for b in ORDER:
        p, n = m[b]
        print(f"    {b:<6} {p / n:.3f}  n={n:,}")

    print("\n[2] rate by sess_pos WITHIN each shipped recency feature:")
    for cond in ('gap', 'hist30'):
        d = rate_table(tr, 'sess_pos', cond=cond)
        print(f"    {cond:<8} " + " ".join(f"{b:>12}" for b in ORDER))
        for c in sorted({x[cond] for x in tr}):
            cells = [(f"{d[(c, b)][0] / d[(c, b)][1]:.3f}"
                      f"({d[(c, b)][1] // 1000}k)" if d[(c, b)][1] >= 500
                      else "     .      ") for b in ORDER]
            print(f"    {str(c):<8} " + " ".join(f"{x:>12}" for x in cells))
    for within in ('gap', 'hist10', 'hist30', 'hist_n'):
        v, vw = absorbed(tr, 'sess_pos', within)
        print(f"    sess_pos rate variance {v:.5f} -> {vw:.5f} within "
              f"{within}  ({100 * (1 - vw / v):.0f}% absorbed)")

    print("\n[3] between-user vs within-user, on the validation tab=1 rows:")
    print(f"    {'feature':<10} {'var(total)':>11} {'var(within-u)':>14} "
          f"{'within share':>13} {'users const':>12}")
    for f in ('sess_pos', 'sess_hit') + tuple(SHIPPED):
        s = prior(tr, va, f)
        by = [s[ii] for ii in idx.values()]
        tot = float(s.var())
        wi = float(np.average([np.var(v) for v in by],
                              weights=[len(v) for v in by]))
        const = float(np.mean([len(set(np.round(v, 6))) == 1 for v in by]))
        print(f"    {f:<10} {tot:>11.6f} {wi:>14.6f} "
              f"{wi / tot if tot else 0:>12.0%} {const:>11.0%}")

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

    t1, r1 = novel('sess_pos', SHIPPED)
    t2, r2 = novel('tab_n', [f for f in SHIPPED if f != 'tab_n'])
    print("\n    within-user prior variance, and what survives regression on"
          " the rest:")
    print(f"    sess_pos vs the champion's features : {t1:.3e} -> {r1:.3e} "
          f"({r1 / t1:.0%} novel, R^2 {1 - r1 / t1:.2f})")
    print(f"    tab_n    vs RICH (the winner, +0.0024): {t2:.3e} -> "
          f"{r2:.3e} ({r2 / t2:.0%} novel, R^2 {1 - r2 / t2:.2f})")
    print(f"    novel within-user signal, loser / winner = {r1 / r2:.2f}x — "
          f"same order, opposite outcome.")
    print("\n    Reading: a 'novel within-user prior variance' screen would "
          "have rated sess_pos as promising as tab_n. It is falsified as a "
          "pre-run filter. A narrow field does not enter this model as a "
          "prior; it enters every interaction with the wide fields at rank "
          "min(k_j,k_l) (Run 32), and that cost is invisible to any marginal "
          "statistic.")


if __name__ == '__main__':
    main()