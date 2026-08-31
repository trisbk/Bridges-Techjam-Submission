""" Run 30: Side and Content Features for FM-Rich

Run 30 revisited features that previously showed little improvement, now testing 
them with the stronger FM-rich model.

The experiment tested hour, video/content attributes (video_type, music_type, and tag), 
and a combination of both. FM may benefit more from these features because it can directly 
learn interactions such as user × tag and tab × hour.

All comparisons and promotion decisions were based on validation performance, with no 
future information or test labels used during training.
"""
import csv, os, time, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step
from harness import _append, BASELINE, GATE, run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

BANKED = 0.6116
BANKED_VALID = 0.61906

print("loading + sequencing ...")
splits = load_sequenced()

vattr = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        tag = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
        vattr[r['video_id']] = (r['video_type'] or 'UNK',
                                r['music_type'] or 'UNK', tag)

hour_by_key = {}
for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
    with open(os.path.join(DATA, f)) as fh:
        for r in csv.DictReader(fh):
            h = str(int(r['hourmin']) // 100) if r['hourmin'] else 'UNK'
            hour_by_key[(r['user_id'], r['video_id'], int(r['time_ms']))] = h

rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
missing_hour = 0
for x in rows_flat:
    u = x['user_id']
    if u not in hist:
        hist[u] = {'last30': collections.deque(maxlen=30),
                   'tag': collections.Counter(), 'last_t': None}
    h = hist[u]
    x['hist30'] = 'none' if not h['last30'] else str(int(10 * sum(h['last30']) / len(h['last30'])))
    vt, mt, tg = vattr.get(x['video_id'], ('UNK', 'UNK', 'UNK'))
    x['video_type'], x['music_type'], x['tag'] = vt, mt, tg
    key = (u, x['video_id'], x['t'])
    if key in hour_by_key:
        x['hour'] = hour_by_key[key]
    else:
        x['hour'] = 'UNK'; missing_hour += 1
    tc = h['tag'][tg]
    x['tag_hist'] = str(tc) if tc < 3 else '3+'
    if h['last_t'] is None:
        x['gap'] = 'none'
    else:
        d = (x['date'] - h['last_t'][0]) * 86400_000 + (x['t'] - h['last_t'][1])
        x['gap'] = ('<1m' if d < 60_000 else '<1h' if d < 3_600_000
                    else '<1d' if d < 86_400_000 else '1d+')
    # update AFTER featurising - own label never enters its own features
    h['last30'].append(x['y']); h['tag'][tg] += x['y']; h['last_t'] = (x['date'], x['t'])

print(f"rows {len(rows_flat)}, hour join misses {missing_hour} "
      f"({100.0 * missing_hour / len(rows_flat):.3f}%)")

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']
HOUR = ['hour']
CONTENT = ['video_type', 'music_type', 'tag']

ARMS = [
    ('R30-ctrl FM-rich (control)',            RICH),
    ('R30a FM-rich + hour',                   RICH + HOUR),
    ('R30b FM-rich + content',                RICH + CONTENT),
    ('R30c FM-rich + hour + content',         RICH + HOUR + CONTENT),
]


def make_fm_fn(fields):
    enc, dim = encode_rows(splits, fields)
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)

    def train_fn(seed, K=4, bs=8192, patience=4):
        m = FM(dim, k=16, lr=0.001, seed=seed)
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
    return train_fn, (Xva, yva, uva, Xte, yte, ute), dim


HYP = ('An FM factorises every field pair, so hour / video_type / music_type / '
       'tag buy cross-field interactions (user x tag, tab x hour) that Runs 5-6 '
       'could not form with a concat-MLP on base fields; the rich-causal premise '
       'may also have changed what those side fields add on top.')
RAT = ('IDEAS #10 and #11 are explicit premise gaps: both side-field families '
       'were only ever measured under the MLP on the five base fields, two '
       'model-class and feature-set revolutions ago. Same logic that '
       'legitimately reopened capacity (R24a) and the recipe knobs (R28).')

results = {}
for name, fields in ARMS:
    fn, _, dim = make_fm_fn(fields)
    rec = run_experiment(
        name=name, hypothesis=HYP, rationale=RAT,
        train_fn=lambda s, _f=fn: _f(s)[0], seeds=3,
        config={'fields': fields, 'n_fields': len(fields), 'dim': dim,
                'k': 16, 'lr': 0.001, 'K': 4, 'patience': 4})
    results[name] = rec

ctrl = results['R30-ctrl FM-rich (control)']
print("\n--- validation-only decision table (test shown for audit) ---")
print(f"{'arm':<34} {'valid':>9} {'d/ctrl':>8}   {'test':>7}")
for name, _ in ARMS:
    r = results[name]
    print(f"{name:<34} {r['valid_mean']:>9.5f} {r['valid_mean'] - ctrl['valid_mean']:>+8.5f}"
          f"   {r['test_mean']:>7.4f}")

cands = [(results[n]['valid_mean'], n) for n, _ in ARMS[1:]]
cands.sort(reverse=True)
best_v, best_name = cands[0]
d_ctrl = best_v - ctrl['valid_mean']
print(f"\nbest feature arm on VALIDATION: {best_name} ({best_v:.5f}, "
      f"{d_ctrl:+.5f} vs control; gate {GATE})")

if d_ctrl > GATE:
    # Promotion: only a committee can legitimately challenge a committee.
    fields = dict(ARMS)[best_name]
    fn, (Xva, yva, uva, Xte, yte, ute), _ = make_fm_fn(fields)
    va_p, te_p = [], []
    for s in range(5):
        _, m = fn(s)
        pv, pt = m.predict(Xva), m.predict(Xte)
        va_p.append((pv - pv.mean()) / pv.std())
        te_p.append((pt - pt.mean()) / pt.std())
        print(f"committee seed {s} done")
    v = evaluate(uva, yva, np.mean(va_p, 0))['primary']
    t = evaluate(ute, yte, np.mean(te_p, 0))['primary']
    beats = v > BANKED_VALID + GATE
    _append({'phase': 'result', 'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
             'name': f'R30d 5-seed committee of {best_name}',
             'valid_mean': round(float(v), 5), 'test_mean': round(float(t), 5),
             'test_std': None, 'd_baseline': round(float(t - BASELINE), 5),
             'verdict': 'WIN' if t - BASELINE > GATE else 'NOISE',
             'note': f'promoted arm; valid {v:.5f} vs banked valid '
                     f'{BANKED_VALID} -> {"BANK" if beats else "no bank"}'})
    print(f"R30d committee: valid {v:.5f} (banked {BANKED_VALID}) test {t:.4f} "
          f"-> {'BANK' if beats else 'NO BANK (incumbent wins ties)'}")
else:
    print("No arm clears the gate over the control on validation -> "
          "IDEAS #10 and #11 die under the FM-rich premise too. No promotion.")