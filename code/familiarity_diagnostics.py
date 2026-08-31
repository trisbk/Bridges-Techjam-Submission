"""Grounding probe for campaign 6, iteration 2 (hypothesis residual_hist_31-100).

Label-free with respect to the eval windows: every rate below is estimated on
TRAIN rows only. Validation/test labels are never read here.

Question: the analyzer's next unresolved slice is `hist=31-100` (10,947 users,
oracle headroom 0.13222). That slice scores 0.62154 vs 0.62059 overall -- i.e.
the model is not *worse* there; EV(old) is exactly 0.0 and the whole headroom
is an arithmetic fact about slice size. So the only actionable reading is
structure INSIDE the slice that the feature set cannot express. `hist_n` is a
lifetime impression count in six log buckets; the 31-100 bucket spans a 3.2x
range and says nothing about how those 31-100 impressions are DISTRIBUTED.

Candidate missing quantity: per-partition EXPOSURE counts. The champion carries
per-author and per-tag *outcome* counts (`auth_hist`, `tag_hist` = prior
long_views, capped 3+) but no exposure counts, so "seen this author 12 times,
never finished one" and "seen this author once, didn't finish it" are the same
row to the model. That is exactly the quantity `tab_n` supplied for the surface
partition -- the one feature that won (+0.0024, R33b) -- and R37's discriminating
control revised its mechanism to partitioned familiarity, half of which is not
surface-specific counting structure at all.

Prints, for auth_n and tag_n:
  1. bucket distribution and long_view rate (train rows), overall and inside
     the hist=31-100 slice;
  2. the same rate table CONDITIONED on the shipped outcome count for the same
     partition -- if exposure is only a proxy for outcomes, the decline dies
     inside each stratum;
  3. within-user share of the induced prior's variance on validation rows
     (train-estimated rates mapped onto valid rows; labels not read), the
     channel the metric scores.

Note on (3): campaign 6 iteration 1 FALSIFIED "novel within-user prior
variance" as a pre-run screen -- it rated the losing `sess_pos` as highly as
the winning `tab_n`. It is printed for continuity and as a necessary condition
(a feature constant within a user cannot move a within-user metric), never as
a go/no-go.

Run from code/:  python3 familiarity_diagnostics.py
"""
import collections, csv, os
import numpy as np
from staleness_ablation import build_features
from tab_surface import add_tab_features
from sequences import DATA

AUTH_BUCKETS = ('0', '1', '2', '3-5', '6-10', '11+')


def _auth_bucket(n):
    return ('0' if n == 0 else '1' if n == 1 else '2' if n == 2
            else '3-5' if n <= 5 else '6-10' if n <= 10 else '11+')


def _tag_bucket(n):
    return ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
            else '11-30' if n <= 30 else '31-100' if n <= 100 else '100+')

TAG_BUCKETS = ('0', '1-3', '4-10', '11-30', '31-100', '100+')


def add_familiarity_features(splits):
    """Per-(user, author) and per-(user, tag) EXPOSURE counts, causal and
    self-exclusive: state updates only AFTER the row is featurized and rows
    are visited in each user's chronological order. Neither feature reads a
    label -- they count impressions, not outcomes."""
    vid2tag = {}
    with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
    rows = [x for rws in splits.values() for x in rws]
    rows.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
    an, tn = collections.Counter(), collections.Counter()
    for x in rows:
        tg = vid2tag.get(x['video_id'], 'UNK')
        x['tag_id'] = tg
        ka, kt = (x['user_id'], x['author_id']), (x['user_id'], tg)
        x['auth_n'] = _auth_bucket(an[ka])
        x['tag_n'] = _tag_bucket(tn[kt])
        an[ka] += 1; tn[kt] += 1
    return splits


def rate_table(rows, key, buckets, label=''):
    c = collections.Counter(); s = collections.Counter()
    for x in rows:
        c[x[key]] += 1; s[x[key]] += x['y']
    print(f"  {label}")
    print(f"    {key:<10} {'rate':>7} {'n':>10}")
    for b in buckets:
        if c[b]:
            print(f"    {b:<10} {s[b] / c[b]:>7.3f} {c[b]:>10,}")


def conditioned(rows, key, buckets, cond, cond_vals):
    """rate(key) inside each stratum of the shipped outcome count `cond`."""
    print(f"    {key} conditioned on shipped {cond}:")
    hdr = ''.join(f"{b:>12}" for b in buckets)
    print(f"      {cond:<8}{hdr}")
    for cv in cond_vals:
        sub = [x for x in rows if x[cond] == cv]
        if len(sub) < 500:
            continue
        c = collections.Counter(); s = collections.Counter()
        for x in sub:
            c[x[key]] += 1; s[x[key]] += x['y']
        cells = ''.join(f"{(s[b] / c[b] if c[b] >= 200 else float('nan')):>12.3f}"
                        for b in buckets)
        print(f"      {cv:<8}{cells}   (n={len(sub):,})")


def within_user_share(train_rows, valid_rows, key):
    """Map each bucket to its TRAIN long_view rate, attach that prior to the
    validation rows, and split its variance into within- vs between-user."""
    c = collections.Counter(); s = collections.Counter()
    for x in train_rows:
        c[x[key]] += 1; s[x[key]] += x['y']
    rate = {b: s[b] / c[b] for b in c if c[b] >= 200}
    glob = sum(s.values()) / sum(c.values())
    p = np.array([rate.get(x[key], glob) for x in valid_rows])
    users = np.array([x['user_id'] for x in valid_rows])
    order = np.argsort(users, kind='stable')
    p, users = p[order], users[order]
    tot = p.var()
    if tot == 0:
        return 0.0, 0.0
    bnd = np.flatnonzero(users[1:] != users[:-1]) + 1
    means = np.concatenate([np.full(len(g), g.mean()) for g in np.split(p, bnd)])
    within = ((p - means) ** 2).mean()
    return within / tot, within


if __name__ == '__main__':
    print("building champion features + partition exposure counts ...")
    splits = add_familiarity_features(add_tab_features(build_features('continuous')))
    tr, va = splits['train'], splits['valid']
    print(f"train rows {len(tr):,}  valid rows {len(va):,}")

    slice_tr = [x for x in tr if x['hist_n'] == '31-100']
    print(f"\n--- the analyzer's slice on train rows: hist_n=31-100, "
          f"{len(slice_tr):,} rows, {len({x['user_id'] for x in slice_tr}):,} users, "
          f"base rate {np.mean([x['y'] for x in slice_tr]):.3f} "
          f"(all train rows {np.mean([x['y'] for x in tr]):.3f})")

    for key, bks in (('auth_n', AUTH_BUCKETS), ('tag_n', TAG_BUCKETS)):
        print(f"\n=== {key} (prior impressions of this "
              f"{'author' if key == 'auth_n' else 'tag'} by this user) ===")
        rate_table(tr, key, bks, 'all train rows')
        rate_table(slice_tr, key, bks, 'train rows inside hist_n=31-100')
        cond = 'auth_hist' if key == 'auth_n' else 'tag_hist'
        conditioned(tr, key, bks, cond, ('0', '1', '2', '3+'))
        sh, wv = within_user_share(tr, va, key)
        print(f"    within-user share of induced prior variance on valid: "
              f"{sh:.0%}  (within-user variance {wv:.3e})")

    for ref in ('tab_n', 'hist_n', 'auth_hist', 'tag_hist'):
        sh, wv = within_user_share(tr, va, ref)
        print(f"  reference {ref:<10} within-user share {sh:>4.0%}  "
              f"variance {wv:.3e}")
