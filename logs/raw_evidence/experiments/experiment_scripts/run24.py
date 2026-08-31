""" Run 24: Improving the FM-Rich Model

Run 24 built on the strong 0.6101 average score from FM models using richer sequence features.

R24a: Increased the FM embedding size to k=32 to test whether richer features benefit from more capacity.

R24b: Tested a five-seed FM-only ensemble to see if mixing model types was still necessary.

R24c: Added deeper user history to capture longer-term behavior.
"""
import time, csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from harness import _append, BASELINE, run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

BANKED = 0.6104

print("loading + sequencing ...")
splits = load_sequenced()

vid2tag = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
for x in rows_flat:
    u = x['user_id']
    if u not in hist:
        hist[u] = {'last30': collections.deque(maxlen=30),
                   'last100': collections.deque(maxlen=100),
                   'tag': collections.Counter(),
                   'auth': collections.Counter(), 'last_t': None}
    h = hist[u]
    x['hist30'] = 'none' if not h['last30'] else str(int(10 * sum(h['last30']) / len(h['last30'])))
    x['hist100'] = 'none' if not h['last100'] else str(int(10 * sum(h['last100']) / len(h['last100'])))
    tg = vid2tag.get(x['video_id'], 'UNK')
    tc = h['tag'][tg]
    x['tag_hist'] = str(tc) if tc < 3 else '3+'
    a = h['auth'][x['author_id']]
    x['auth_hist9'] = str(a) if a < 9 else '9+'
    if h['last_t'] is None:
        x['gap'] = 'none'
    else:
        d = (x['date'] - h['last_t'][0]) * 86400_000 + (x['t'] - h['last_t'][1])
        x['gap'] = ('<1m' if d < 60_000 else '<1h' if d < 3_600_000
                    else '<1d' if d < 86_400_000 else '1d+')
    h['last30'].append(x['y']); h['last100'].append(x['y'])
    h['tag'][tg] += x['y']; h['auth'][x['author_id']] += x['y']
    h['last_t'] = (x['date'], x['t'])

FIELDS_B = BASE + SEQ + ['hist30', 'tag_hist', 'gap']
FIELDS_C = BASE + SEQ + ['hist30', 'tag_hist', 'gap', 'hist100', 'auth_hist9']


def make_fm_fn(enc, dim, k):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    def train_fn(seed, K=4, bs=8192, patience=4):
        m = FM(dim, k=k, lr=0.001, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 41):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                infonce_step(m, Xtr[P[i:i + bs]], Xtr[N[i:i + bs].reshape(-1)],
                             len(P[i:i + bs]), K)
            va = evaluate(uva, yva, m.predict(Xva))
            if va['primary'] > best + 1e-5:
                best, bad = va['primary'], 0
                best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
            else:
                bad += 1
                if bad >= patience:
                    break
        m.V, m.W, m.b = best_state
        return {'valid': evaluate(uva, yva, m.predict(Xva)),
                'test': evaluate(ute, yte, m.predict(Xte))}, m
    return train_fn, (Xva, yva, uva, Xte, yte, ute)


enc_b, dim_b = encode_rows(splits, FIELDS_B)
fn_b32, _ = make_fm_fn(enc_b, dim_b, 32)
run_experiment(
    name='R24a FM-rich k=32',
    hypothesis='Capacity finally pays once features are rich: earlier k=32 '
               'flops (Runs 4, 12) were on information-starved fields.',
    rationale='FM-rich at k=16 is the new best single class (0.6101 avg); '
              'k is the cheapest knob on it.',
    train_fn=lambda s: fn_b32(s)[0], seeds=3, config={'k': 32, 'fields': 'rich'})

# R24b: FM-rich-only committee (5 seeds, k=16)
fn_b16, (Xva, yva, uva, Xte, yte, ute) = make_fm_fn(enc_b, dim_b, 16)
va_p, te_p = [], []
for s in range(5):
    _, m = fn_b16(s)
    pv, pt = m.predict(Xva), m.predict(Xte)
    va_p.append((pv - pv.mean()) / pv.std()); te_p.append((pt - pt.mean()) / pt.std())
    print(f"fm-rich k16 seed {s} done")
v = evaluate(uva, yva, np.mean(va_p, 0))['primary']
t = evaluate(ute, yte, np.mean(te_p, 0))['primary']
mark = '✅ BETTER than banked' if t > BANKED else '❌ not better'
_append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
         'name': 'R24b FM-rich-only 5-seed committee',
         'valid_mean': round(float(v), 5), 'test_mean': round(float(t), 5),
         'test_std': None, 'd_baseline': round(float(t - BASELINE), 5),
         'verdict': 'WIN' if t - BASELINE > 0.002 else 'NOISE',
         'note': 'k=16 FM on rich fields only'})
print(f"R24b FM-rich committee: valid {v:.5f}  test {t:.4f}  vs banked {t - BANKED:+.4f} {mark}")

# R24c: deeper history fields, FM k=16, via harness
enc_c, dim_c = encode_rows(splits, FIELDS_C)
fn_c16, _ = make_fm_fn(enc_c, dim_c, 16)
run_experiment(
    name='R24c FM deeper history (+hist100, auth_hist 9+ cap)',
    hypothesis='A 100-impression window and a higher per-author cap carry '
               'longer-term taste the 30-window truncates.',
    rationale='The sequence family keeps paying; depth of history is its '
              'cheapest unexplored axis.',
    train_fn=lambda s: fn_c16(s)[0], seeds=3, config={'fields': 'rich+deep'})