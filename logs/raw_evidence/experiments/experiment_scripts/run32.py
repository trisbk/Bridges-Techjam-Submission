""" Run 32: Per-Field Embedding Sizes

Run 32 tested whether different feature fields benefit from different embedding sizes, 
rather than using k=16 for every field.

High-cardinality fields such as user, video, and author were tested with larger embeddings, 
while smaller fields were tested with reduced sizes to limit overfitting. Several 
combinations were compared against the standard 16/16 configuration.

All models used the same FM-rich setup, with promotion decisions based on validation 
performance only.
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
WIDE_MIN_CARD = 1000     

def masked_infonce_step(m, Xp, Xn_flat, B, K, mask):
    """ The InfoNCE training step was modified to mask embedding gradients for each field, 
    allowing each field to use its assigned embedding size.

    Masked dimensions remain at zero throughout training, effectively limiting each field's 
    interaction capacity. When no dimensions are masked, the method behaves identically to 
    the original infonce_step.
    """
    zp, Ep, Sp = m.logits(Xp)
    zn, En, Sn = m.logits(Xn_flat)
    Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
    Z -= Z.max(axis=1, keepdims=True)
    Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
    e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    en = e[:, 1:].reshape(B * K)
    np.add.at(gW, Xp, e[:, 0][:, None])
    np.add.at(gW, Xn_flat, en[:, None])
    np.add.at(gV, Xp, e[:, 0][:, None, None] * (Sp[:, None, :] - Ep))
    np.add.at(gV, Xn_flat, en[:, None, None] * (Sn[:, None, :] - En))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    gV *= mask
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for Pm, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        Pm -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.V *= mask

print("loading + sequencing ...")
splits = load_sequenced()

vtag = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        vtag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'

rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
for x in rows_flat:
    u = x['user_id']
    if u not in hist:
        hist[u] = {'last30': collections.deque(maxlen=30),
                   'tag': collections.Counter(), 'last_t': None}
    h = hist[u]
    x['hist30'] = 'none' if not h['last30'] else str(int(10 * sum(h['last30']) / len(h['last30'])))
    tg = vtag.get(x['video_id'], 'UNK')
    tc = h['tag'][tg]
    x['tag_hist'] = str(tc) if tc < 3 else '3+'
    if h['last_t'] is None:
        x['gap'] = 'none'
    else:
        d = (x['date'] - h['last_t'][0]) * 86400_000 + (x['t'] - h['last_t'][1])
        x['gap'] = ('<1m' if d < 60_000 else '<1h' if d < 3_600_000
                    else '<1d' if d < 86_400_000 else '1d+')
    h['last30'].append(x['y']); h['tag'][tg] += x['y']; h['last_t'] = (x['date'], x['t'])

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']
enc, dim = encode_rows(splits, RICH)          # also stamps dur_bucket on every row
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)

tr = splits['train']
card = []
for f in RICH:
    seen = set()
    for x in tr:
        seen.add(x[f])
    card.append(len(seen))
dims = [c + 1 for c in card]
offsets = np.cumsum([0] + dims[:-1]).astype(np.int64)
assert int(sum(dims)) == dim, (sum(dims), dim)
WIDE = [c >= WIDE_MIN_CARD for c in card]

print(f"rows {len(rows_flat)}, dim {dim}, fields {len(RICH)}")
print(f"{'field':<12} {'card':>7}  {'group':<6}")
for f, c, w in zip(RICH, card, WIDE):
    print(f"{f:<12} {c:>7}  {'WIDE' if w else 'narrow':<6}")


def build_mask(k_wide, k_narrow, kmax):
    """(dim, kmax) float32 mask: field j is live in dims [0, k_j)."""
    M = np.zeros((dim, kmax), dtype=np.float32)
    for i, w in enumerate(WIDE):
        kj = k_wide if w else k_narrow
        M[offsets[i]:offsets[i] + dims[i], :kj] = 1.0
    return M

def preflight():
    rng = np.random.default_rng(0)
    a = FM(dim, k=16, lr=0.001, seed=0)
    b = FM(dim, k=16, lr=0.001, seed=0)
    ones = np.ones((dim, 16), dtype=np.float32)
    P, N = sample_lists(pairs_users, rng, 4)
    for i in range(5):
        p = P[i * 2048:(i + 1) * 2048]; n = N[i * 2048:(i + 1) * 2048]
        infonce_step(a, Xtr[p], Xtr[n.reshape(-1)], len(p), 4)
        masked_infonce_step(b, Xtr[p], Xtr[n.reshape(-1)], len(p), 4, ones)
    assert np.array_equal(a.V, b.V) and np.array_equal(a.W, b.W), "control drift!"
    print("preflight OK: masked_infonce_step == listwise.infonce_step at mask==1 "
          "(bit-identical V and W over 5 steps)")


preflight()

def make_fn(k_wide, k_narrow):
    kmax = max(k_wide, k_narrow)
    mask = build_mask(k_wide, k_narrow, kmax)
    live = float(mask.sum())

    def train_fn(seed, K=4, bs=8192, patience=4):
        m = FM(dim, k=kmax, lr=0.001, seed=seed)
        m.V *= mask
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 41):
            P, N = sample_lists(pairs_users, rng, K)
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs]
                masked_infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K, mask)
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
    return train_fn, live


ARMS = [
    ('R32-ctrl FM-rich wide16/narrow16', 16, 16),
    ('R32a FM-rich wide24/narrow8',      24, 8),
    ('R32b FM-rich wide16/narrow8',      16, 8),
    ('R32c FM-rich wide24/narrow16',     24, 16),
    ('R32d FM-rich wide24/narrow24',     24, 24),
]

HYP = ('k in an FM is the RANK of every interaction a field takes part in, so a '
       'single global k is a mis-specification: rank 16 lets 3-value fields like '
       'prev1 memorise their low-cardinality crosses, while user x video may be '
       'rank-limited. Setting rank per field - wide fields 24, narrow fields 8 - '
       'should regularise the narrow crosses and free the wide one.')
RAT = ('IDEAS #14, the only structural idea never tested under any premise. '
       'Run 4 killed UNIFORM k=32, which is a different question: it raised '
       'every interaction rank at once and so cannot separate "more rank on '
       'user x video" from "more rank on prev1 x tab". The 2x3 grid here '
       'separates them, with a uniform k=24 arm as the capacity control.')

results, livecnt = {}, {}
for name, kw, kn in ARMS:
    fn, live = make_fn(kw, kn)
    livecnt[name] = live
    rec = run_experiment(
        name=name, hypothesis=HYP, rationale=RAT,
        train_fn=lambda s, _f=fn: _f(s)[0], seeds=3,
        config={'fields': RICH, 'n_fields': len(RICH), 'dim': dim,
                'k_wide': kw, 'k_narrow': kn, 'wide_min_card': WIDE_MIN_CARD,
                'wide_fields': [f for f, w in zip(RICH, WIDE) if w],
                'live_V_params': int(live), 'lr': 0.001, 'K': 4, 'patience': 4})
    results[name] = rec

ctrl = results['R32-ctrl FM-rich wide16/narrow16']
print("\n--- validation-only decision table (test shown for audit) ---")
print(f"{'arm':<36} {'V params':>10} {'valid':>9} {'d/ctrl':>9}   {'test':>7}")
for name, kw, kn in ARMS:
    r = results[name]
    print(f"{name:<36} {livecnt[name]:>10.0f} {r['valid_mean']:>9.5f} "
          f"{r['valid_mean'] - ctrl['valid_mean']:>+9.5f}   {r['test_mean']:>7.4f}")

cands = sorted(((results[n]['valid_mean'], n) for n, _, _ in ARMS[1:]), reverse=True)
best_v, best_name = cands[0]
d_ctrl = best_v - ctrl['valid_mean']
print(f"\nbest arm on VALIDATION: {best_name} ({best_v:.5f}, "
      f"{d_ctrl:+.5f} vs control; gate {GATE})")

if d_ctrl > GATE:
    kw, kn = {n: (a, b) for n, a, b in ARMS}[best_name]
    fn, _ = make_fn(kw, kn)
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
             'name': f'R32e 5-seed committee of {best_name}',
             'valid_mean': round(float(v), 5), 'test_mean': round(float(t), 5),
             'test_std': None, 'd_baseline': round(float(t - BASELINE), 5),
             'verdict': 'WIN' if t - BASELINE > GATE else 'NOISE',
             'note': f'promoted arm; valid {v:.5f} vs banked valid '
                     f'{BANKED_VALID} -> {"BANK" if beats else "no bank"}'})
    print(f"R32e committee: valid {v:.5f} (banked {BANKED_VALID}) test {t:.4f} "
          f"-> {'BANK' if beats else 'NO BANK (incumbent wins ties)'}")
else:
    print("No per-field-rank arm clears the gate over the control on "
          "validation -> IDEAS #14 dies. No promotion.")