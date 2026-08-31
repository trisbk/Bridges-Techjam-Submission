"""Feature-staleness ablation for the causal sequence features.

The shipped model's history features update continuously: a test row's
features include the realized labels of the same user's earlier test-window
impressions. That matches a streaming feature store. This script measures
how much of the headline gain depends on that serving assumption, using the
shipped frozen weights (the model is bit identical; only test-row
featurization changes).

Regimes:
  continuous : as submitted. History updates after every impression.
  daily      : daily batch refresh. A test row sees history from days
               strictly before its own date (plus everything pre-test).
               Within-day feedback is withheld.
  frozen     : history state frozen at the end of the validation window.
               No test-window feedback at all. Note: gap and hist_n go out
               of distribution here (train/serve skew), so this is a lower
               bound, not a realistic serving estimate.

Run from code/:  python3 staleness_ablation.py
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA


def build_features(regime):
    """Rebuild rows with the rich causal features under a feature-refresh
    regime. Train and validation rows always use the continuous stream (the
    shipped weights were trained that way and validation is pre-test);
    the regime governs test-window updates only."""
    splits = load_sequenced()  # continuous prev1/hist10/hist_n/auth_hist
    vid2tag = {}
    with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'

    rows_flat = [x for rws in splits.values() for x in rws]
    rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))

    TEST_START = 20220429

    hist = {}
    for x in rows_flat:
        u = x['user_id']
        if u not in hist:
            hist[u] = {'last30': collections.deque(maxlen=30),
                       'tag': collections.Counter(), 'last_t': None,
                       'pend': [], 'pend_day': None,
                       'last10': collections.deque(maxlen=10), 'n': 0,
                       'prev1': None, 'auth': collections.Counter()}
        h = hist[u]
        in_test = x['date'] >= TEST_START

        if in_test and regime == 'daily':
            # flush pending completed days (strictly before this row's date)
            if h['pend_day'] is not None and x['date'] > h['pend_day']:
                for p in h['pend']:
                    _apply(h, p, vid2tag)
                h['pend'], h['pend_day'] = [], None

        # ---- featurize from current applied state ----
        x['prev1'] = 'none' if h['prev1'] is None else str(h['prev1'])
        x['hist10'] = 'none' if not h['last10'] else str(sum(h['last10']))
        n = h['n']
        x['hist_n'] = ('0' if n == 0 else '1-3' if n <= 3 else '4-10' if n <= 10
                       else '11-30' if n <= 30 else '31-100' if n <= 100 else '100+')
        a = h['auth'][x['author_id']]
        x['auth_hist'] = str(a) if a < 3 else '3+'
        x['hist30'] = ('none' if not h['last30']
                       else str(int(10 * sum(h['last30']) / len(h['last30']))))
        tg = vid2tag.get(x['video_id'], 'UNK')
        tc = h['tag'][tg]
        x['tag_hist'] = str(tc) if tc < 3 else '3+'
        if h['last_t'] is None:
            x['gap'] = 'none'
        else:
            d = (x['date'] - h['last_t'][0]) * 86400_000 + (x['t'] - h['last_t'][1])
            x['gap'] = ('<1m' if d < 60_000 else '<1h' if d < 3_600_000
                        else '<1d' if d < 86_400_000 else '1d+')

        # ---- update state per regime ----
        if not in_test or regime == 'continuous':
            _apply(h, x, vid2tag)
        elif regime == 'daily':
            h['pend'].append(x); h['pend_day'] = x['date']
        # frozen: test rows never update state

    return splits


def _apply(h, x, vid2tag):
    h['prev1'] = x['y']; h['last10'].append(x['y']); h['n'] += 1
    h['auth'][x['author_id']] += x['y']
    h['last30'].append(x['y'])
    h['tag'][vid2tag.get(x['video_id'], 'UNK')] += x['y']
    h['last_t'] = (x['date'], x['t'])


RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']

def _frozen_dir():
    """Locate the R24b-era weights this ablation was measured on. After the
    31 Aug promotion (R33c, adds tab_n) the shipped frozen_model/ holds the
    new champion, whose feature dimension differs; this analysis and the
    numbers documented from it belong to the pre-promotion champion, kept
    at final_output/frozen_model_r24b/."""
    for c in (os.path.join('..', 'final_output', 'frozen_model_r24b'),
              'frozen_model_r24b'):
        if os.path.exists(os.path.join(c, 'fm_seed0.npz')):
            return c
    raise FileNotFoundError(
        'frozen_model_r24b not found; expected '
        '../final_output/frozen_model_r24b or ./frozen_model_r24b')


if __name__ == '__main__':
    FROZEN = _frozen_dir()
    print(f"{'regime':<12} {'GAUC':>8} {'nDCG@5':>8} {'primary':>9} {'vs baseline':>12}")
    for regime in ('continuous', 'daily', 'frozen'):
        splits = build_features(regime)
        enc, dim = encode_rows(splits, RICH)
        Xte, yte, ute = enc['test']
        te_preds = []
        for seed in range(5):
            z = np.load(os.path.join(FROZEN, f'fm_seed{seed}.npz'))
            m = FM(dim, k=16, seed=seed)
            assert m.V.shape == z['V'].shape, (m.V.shape, z['V'].shape)
            m.V, m.W, m.b = z['V'], z['W'], np.float32(z['b'])
            pt = m.predict(Xte)
            te_preds.append((pt - pt.mean()) / pt.std())
        r = evaluate(ute, yte, np.mean(te_preds, 0))
        print(f"{regime:<12} {r['GAUC']:>8.4f} {r['nDCG@5']:>8.4f} "
              f"{r['primary']:>9.5f} {r['primary']-0.5946:>+12.4f}")
