"""Feature-staleness ablation for the causal sequence features.

Measures how sensitive the frozen model is to feature freshness without
retraining; only test-time history construction changes.

Regimes:
- continuous: history updates after every impression, matching the submitted
  streaming setup and allowing strictly earlier test-window outcomes.
- daily: history refreshes once per day; within-day feedback is withheld.
- frozen: history stops at the validation boundary, so no test-window feedback
  is used.

Because the same continuously-trained weights are used in every regime, the
frozen setting introduces train/serve skew and should be interpreted as a
lower bound rather than a realistic frozen-serving estimate.
"""
import csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA


def build_features(regime):
    splits = load_sequenced() 
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
            if h['pend_day'] is not None and x['date'] > h['pend_day']:
                for p in h['pend']:
                    _apply(h, p, vid2tag)
                h['pend'], h['pend_day'] = [], None

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

        if not in_test or regime == 'continuous':
            _apply(h, x, vid2tag)
        elif regime == 'daily':
            h['pend'].append(x); h['pend_day'] = x['date']

    return splits


def _apply(h, x, vid2tag):
    h['prev1'] = x['y']; h['last10'].append(x['y']); h['n'] += 1
    h['auth'][x['author_id']] += x['y']
    h['last30'].append(x['y'])
    h['tag'][vid2tag.get(x['video_id'], 'UNK')] += x['y']
    h['last_t'] = (x['date'], x['t'])


RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']

def _frozen_dir():
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
