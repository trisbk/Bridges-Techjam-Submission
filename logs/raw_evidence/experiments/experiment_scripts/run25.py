""" Run 25: Final Feature and Ensemble Tests

Run 25 tested two final improvements.

R25a – Author History: Isolated the expanded author-history feature to test its 
impact without the weaker hist100 feature.

R25b – Cross-View Ensemble: Combined the FM-rich and interest-MLP ensembles, using 
validation data to find the best weighting between their different views of user history.
"""
import time, csv, os, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from harness import _append, BASELINE, run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA
from interest import InterestMLP, infonce_interest_step, L

BANKED = 0.6116

print("loading + sequencing ...")
splits = load_sequenced()

vid2tag = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
watched = {}
for x in rows_flat:
    u = x['user_id']
    if u not in hist:
        hist[u] = {'last30': collections.deque(maxlen=30),
                   'tag': collections.Counter(),
                   'auth': collections.Counter(), 'last_t': None}
        watched[u] = collections.deque(maxlen=L)
    h = hist[u]
    x['hist30'] = 'none' if not h['last30'] else str(int(10 * sum(h['last30']) / len(h['last30'])))
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
    x['hvids'] = list(watched[u])
    h['last30'].append(x['y']); h['tag'][tg] += x['y']
    h['auth'][x['author_id']] += x['y']; h['last_t'] = (x['date'], x['t'])
    if x['y'] == 1:
        watched[u].append(x['video_id'])

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']
RICH_A9 = BASE + SEQ + ['hist30', 'tag_hist', 'gap', 'auth_hist9']


def train_fm(enc, dim, seed):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    m = FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        P, N = sample_lists(pairs_users, rng, 4)
        for i in range(0, len(P), 8192):
            infonce_step(m, Xtr[P[i:i + 8192]], Xtr[N[i:i + 8192].reshape(-1)],
                         len(P[i:i + 8192]), 4)
        va = evaluate(uva, yva, m.predict(Xva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= 4:
                break
    m.V, m.W, m.b = best_state
    return m


# ---- R25a: auth9 isolate via harness ----
enc_a9, dim_a9 = encode_rows(splits, RICH_A9)
def fn_a9(seed):
    m = train_fm(enc_a9, dim_a9, seed)
    _, _, uva = enc_a9['valid']; _, _, ute = enc_a9['test']
    Xva, yva, _ = enc_a9['valid']; Xte, yte, _ = enc_a9['test']
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test': evaluate(ute, yte, m.predict(Xte))}
run_experiment(
    name='R25a FM rich + auth_hist 9+ cap (isolated)',
    hypothesis='R24c conflated a bad change (hist100) with an untested one; '
               'the raised author-history cap alone may still help.',
    rationale='One-variable-at-a-time discipline; cheap.',
    train_fn=fn_a9, seeds=3, config={'fields': 'rich+auth9'})

# ---- R25b: cross-view blend, alpha on validation ----
enc_r, dim_r = encode_rows(splits, RICH)
Xva, yva, uva = enc_r['valid']; Xte, yte, ute = enc_r['test']
fm_va, fm_te = [], []
for s in range(5):
    m = train_fm(enc_r, dim_r, s)
    pv, pt = m.predict(enc_r['valid'][0]), m.predict(enc_r['test'][0])
    fm_va.append((pv - pv.mean()) / pv.std()); fm_te.append((pt - pt.mean()) / pt.std())
    print(f"fm-rich seed {s} done")

# interest models on base+seq fields with history arrays
fields_i = BASE + SEQ
enc_i, dim_i = encode_rows(splits, fields_i)
vocabs = [dict() for _ in fields_i]
for x in splits['train']:
    for i, f in enumerate(fields_i):
        if x[f] not in vocabs[i]:
            vocabs[i][x[f]] = len(vocabs[i])
dims_i = [len(v) + 1 for v in vocabs]
offsets = np.cumsum([0] + dims_i[:-1]).astype(np.int32)
vvocab = vocabs[1]; vunk = len(vvocab); voffset = int(offsets[1])

def hist_arrays(rws):
    Hm = np.zeros((len(rws), L), dtype=np.int32)
    Wt = np.zeros((len(rws), L), dtype=np.float32)
    for n, x in enumerate(rws):
        hv = x['hvids']; c = len(hv)
        for j, v in enumerate(hv):
            Hm[n, j] = vvocab.get(v, vunk) + voffset
        if c:
            Wt[n, :c] = 1.0 / c
    return Hm, Wt

H_ = {name: hist_arrays(rws) for name, rws in splits.items()}
Xtr_i, ytr_i, utr_i = enc_i['train']
Htr, Wtr = H_['train']; Hva, Wva = H_['valid']; Hte, Wte = H_['test']
pairs_i, _, _ = build_pair_index(utr_i, ytr_i)

int_va, int_te = [], []
for s in range(5):
    m = InterestMLP(dim_i, len(fields_i), k=16, H=64, lr=0.0003, seed=s)
    rng = np.random.default_rng(s)
    best, best_state, bad = -1, None, 0
    for ep in range(1, 41):
        P, N = sample_lists(pairs_i, rng, 4)
        for i in range(0, len(P), 8192):
            p = P[i:i + 8192]; n = N[i:i + 8192].reshape(-1)
            infonce_interest_step(m, Xtr_i[p], Htr[p], Wtr[p],
                                  Xtr_i[n], Htr[n], Wtr[n], len(p), 4)
        va = evaluate(enc_i['valid'][2], enc_i['valid'][1],
                      m.predict(enc_i['valid'][0], Hva, Wva))
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= 4:
                break
    m.load_state(best_state)
    pv = m.predict(enc_i['valid'][0], Hva, Wva)
    pt = m.predict(enc_i['test'][0], Hte, Wte)
    int_va.append((pv - pv.mean()) / pv.std()); int_te.append((pt - pt.mean()) / pt.std())
    print(f"interest seed {s} done")

Fva, Fte = np.mean(fm_va, 0), np.mean(fm_te, 0)
Iva, Ite = np.mean(int_va, 0), np.mean(int_te, 0)
best_a, best_v = 1.0, -1
for a in np.arange(0.0, 1.01, 0.05):
    v = evaluate(uva, yva, a * Fva + (1 - a) * Iva)['primary']
    if v > best_v:
        best_v, best_a = v, a
t = evaluate(ute, yte, best_a * Fte + (1 - best_a) * Ite)['primary']
mark = '✅ BETTER than banked' if t > BANKED else '❌ not better'
_append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
         'name': f'R25b cross-view blend alpha={best_a:.2f}',
         'valid_mean': round(float(best_v), 5), 'test_mean': round(float(t), 5),
         'test_std': None, 'd_baseline': round(float(t - BASELINE), 5),
         'verdict': 'WIN' if t - BASELINE > 0.002 else 'NOISE',
         'note': 'FM-rich committee x interest committee, alpha on validation'})
print(f"R25b blend: alpha {best_a:.2f}  valid {best_v:.5f}  test {t:.4f}  "
      f"vs banked {t - BANKED:+.4f} {mark}")