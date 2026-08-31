""" Run 31: Play-Ratio as an Auxiliary Signal

Run 31 revisited play-time ratio, but used it as an auxiliary training signal rather 
than replacing the main long_view target.

The play ratio was normalized within duration groups to reduce bias toward shorter videos. 
Different auxiliary weights and a non-normalized comparison were tested to measure whether 
this additional engagement signal could improve the FM-rich model.

The main InfoNCE ranking objective remained unchanged, and all auxiliary information came 
from training data only.
"""
import csv, os, time, collections
import numpy as np
from evaluate import evaluate
from baseline import FM, sigmoid
from pairwise import build_pair_index
from listwise import sample_lists
from harness import _append, BASELINE, GATE, run_experiment
from sequences import load_sequenced, encode_rows, BASE, SEQ, DATA

BANKED = 0.6116
BANKED_VALID = 0.61906

class AuxFM(FM):
    """FM whose ranking head is untouched, plus an aux head sharing V.

    aux score  za = ba + Wa[X].sum + sum_j ca_j * 0.5 * (S_j^2 - sum_f E_fj^2)

    Wa / ca / ba are the head's own parameters; V is shared with the ranking
    head, which is the entire point - the aux loss reshapes the embeddings.
    Both objectives route through ONE Adam state (shared moments, shared t),
    the Run 19 pattern: with shared moments, lambda genuinely controls the
    aux term's influence, whereas separate Adam states would normalise lambda
    away (Adam is scale-invariant per parameter).
    """

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        super().__init__(dim, k=k, lr=lr, l2=l2, seed=seed)
        self.Wa = np.zeros(dim, dtype=np.float32)
        self.ca = np.ones(k, dtype=np.float32)
        self.ba = np.float32(0.0)
        self.mWa = np.zeros_like(self.Wa); self.vWa = np.zeros_like(self.Wa)
        self.mca = np.zeros_like(self.ca); self.vca = np.zeros_like(self.ca)

    def _adam(self, gV, gW, gWa, gca):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV),
                            (self.W, gW, self.mW, self.vW),
                            (self.Wa, gWa, self.mWa, self.vWa),
                            (self.ca, gca, self.mca, self.vca)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)

    def rank_step(self, Xp, Xn_flat, B, K):
        """listwise.infonce_step, re-expressed over the shared Adam state."""
        zp, Ep, Sp = self.logits(Xp)
        zn, En, Sn = self.logits(Xn_flat)
        Z = np.concatenate([zp[:, None], zn.reshape(B, K)], axis=1)
        Z -= Z.max(axis=1, keepdims=True)
        Pr = np.exp(Z); Pr /= Pr.sum(axis=1, keepdims=True)
        e = Pr.copy(); e[:, 0] -= 1.0; e = (e / B).astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        en = e[:, 1:].reshape(B * K)
        np.add.at(gW, Xp, e[:, 0][:, None])
        np.add.at(gW, Xn_flat, en[:, None])
        np.add.at(gV, Xp, e[:, 0][:, None, None] * (Sp[:, None, :] - Ep))
        np.add.at(gV, Xn_flat, en[:, None, None] * (Sn[:, None, :] - En))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        # aux params get zero gradient here, but their moments must decay in
        # lockstep with the shared step counter
        self._adam(gV, gW, np.zeros_like(self.Wa), np.zeros_like(self.ca))
        # softmax rows sum to zero gradient for the bias: no b update

    def aux_step(self, X, target, lam):
        """Soft-label BCE on the duration-normalised play-ratio percentile."""
        B = len(X)
        E = self.V[X]; S = E.sum(1)
        q = (0.5 * ((S ** 2) - (E ** 2).sum(1))).astype(np.float32)   # (B,k)
        za = self.ba + self.Wa[X].sum(1) + q @ self.ca
        e = (lam * (sigmoid(za) - target) / B).astype(np.float32)     # (B,)
        gV = np.zeros_like(self.V); gWa = np.zeros_like(self.Wa)
        np.add.at(gWa, X, e[:, None])
        np.add.at(gV, X, (e[:, None] * self.ca[None, :])[:, None, :] * (S[:, None, :] - E))
        gca = (q * e[:, None]).sum(0).astype(np.float32)
        gV += self.l2 * self.V; gWa += self.l2 * self.Wa
        self._adam(gV, np.zeros_like(self.W), gWa, gca)
        self.ba -= np.float32(self.lr * e.sum())

print("loading + sequencing ...")
splits = load_sequenced()

vtag = {}
with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
    for r in csv.DictReader(fh):
        vtag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'

play_by_key = {}
for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
    with open(os.path.join(DATA, f)) as fh:
        for r in csv.DictReader(fh):
            play_by_key[(r['user_id'], r['video_id'], int(r['time_ms']))] = float(r['play_time_ms'])

rows_flat = [x for rws in splits.values() for x in rws]
rows_flat.sort(key=lambda x: (x['user_id'], x['date'], x['t']))
hist = {}
missing_play = 0
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
    key = (u, x['video_id'], x['t'])
    if key in play_by_key:
        x['play'] = play_by_key[key]
    else:
        x['play'] = float('nan'); missing_play += 1
    # update AFTER featurising - own label never enters its own features
    h['last30'].append(x['y']); h['tag'][tg] += x['y']; h['last_t'] = (x['date'], x['t'])

print(f"rows {len(rows_flat)}, play_time join misses {missing_play} "
      f"({100.0 * missing_play / len(rows_flat):.3f}%)")

RICH = BASE + SEQ + ['hist30', 'tag_hist', 'gap']
enc, dim = encode_rows(splits, RICH)          # also stamps dur_bucket on every row
Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
pairs_users, _, _ = build_pair_index(utr, ytr)

tr = splits['train']
ratio = np.array([min(max(x['play'] / max(x['duration'], 1.0), 0.0), 2.0)
                  if x['play'] == x['play'] else 0.0 for x in tr], dtype=np.float64)
bucket = np.array([int(x['dur_bucket']) for x in tr])


def _percentile(vals, mask=None):
    """Tie-aware percentile of each value within its reference population."""
    out = np.zeros(len(vals), dtype=np.float32)
    idx = np.arange(len(vals)) if mask is None else np.nonzero(mask)[0]
    ref = np.sort(vals[idx])
    lo = np.searchsorted(ref, vals[idx], side='left')
    hi = np.searchsorted(ref, vals[idx], side='right')
    out[idx] = (0.5 * (lo + hi) / max(len(ref), 1)).astype(np.float32)
    return out


t_norm = np.zeros(len(tr), dtype=np.float32)      # within-duration-bucket
for b in np.unique(bucket):
    t_norm += _percentile(ratio, bucket == b)
t_glob = _percentile(ratio)                       # duration-mediated contrast

print(f"aux targets: within-bucket mean {t_norm.mean():.3f}, "
      f"global mean {t_glob.mean():.3f}, "
      f"corr(t_norm, y) {np.corrcoef(t_norm, ytr)[0,1]:.3f}, "
      f"corr(t_glob, y) {np.corrcoef(t_glob, ytr)[0,1]:.3f}, "
      f"corr(t_glob, dur_bucket) {np.corrcoef(t_glob, bucket)[0,1]:+.3f}, "
      f"corr(t_norm, dur_bucket) {np.corrcoef(t_norm, bucket)[0,1]:+.3f}")

def make_fn(lam, target):
    def train_fn(seed, K=4, bs=8192, patience=4):
        m = AuxFM(dim, k=16, lr=0.001, seed=seed)
        rng = np.random.default_rng(seed)
        best, best_state, bad = -1, None, 0
        for ep in range(1, 41):
            P, N = sample_lists(pairs_users, rng, K)
            aux_idx = rng.permutation(len(ytr)) if lam > 0 else None
            na = 0
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; n = N[i:i + bs]
                m.rank_step(Xtr[p], Xtr[n.reshape(-1)], len(p), K)
                if lam > 0 and (i // bs) % 2 == 0 and na + bs <= len(aux_idx):
                    j = aux_idx[na:na + bs]; na += bs
                    m.aux_step(Xtr[j], target[j], lam)
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
    return train_fn


ARMS = [
    ('R31-ctrl FM-rich, no aux head',           0.0, None),
    ('R31a aux play-ratio (bucket) lam=0.3',    0.3, t_norm),
    ('R31b aux play-ratio (bucket) lam=1.0',    1.0, t_norm),
    ('R31c aux play-ratio (global) lam=0.3',    0.3, t_glob),
]

HYP = ('The duration-normalised play ratio is a dense graded engagement signal '
       'defined on every impression, including the negatives the binary '
       'long_view label says nothing about; as an AUXILIARY head sharing the '
       'FM embeddings (not as the target, which Run 1 killed) it should shape '
       'V with near-miss information the ranking loss cannot see.')
RAT = ('IDEAS #12: Run 1 killed the dense ratio TARGET as duration-mediated but '
       'explicitly left the duration-normalised variant open, and it was never '
       'revisited after the objective / model-class / feature revolutions. '
       'Unlike Run 19s sparse click+like heads, this signal is dense and graded. '
       'The global-percentile arm isolates whether the duration normalisation '
       'is what distinguishes it from Run 1s dead form.')

results = {}
for name, lam, tgt in ARMS:
    fn = make_fn(lam, tgt)
    rec = run_experiment(
        name=name, hypothesis=HYP, rationale=RAT,
        train_fn=lambda s, _f=fn: _f(s)[0], seeds=3,
        config={'fields': RICH, 'n_fields': len(RICH), 'dim': dim,
                'k': 16, 'lr': 0.001, 'K': 4, 'patience': 4, 'lam': lam,
                'aux_target': ('none' if tgt is None else
                               'within_dur_bucket_percentile' if tgt is t_norm
                               else 'global_percentile')})
    results[name] = rec

ctrl = results['R31-ctrl FM-rich, no aux head']
print("\n--- validation-only decision table (test shown for audit) ---")
print(f"{'arm':<40} {'valid':>9} {'d/ctrl':>9}   {'test':>7}")
for name, _, _ in ARMS:
    r = results[name]
    print(f"{name:<40} {r['valid_mean']:>9.5f} {r['valid_mean'] - ctrl['valid_mean']:>+9.5f}"
          f"   {r['test_mean']:>7.4f}")

cands = sorted(((results[n]['valid_mean'], n) for n, _, _ in ARMS[1:]), reverse=True)
best_v, best_name = cands[0]
d_ctrl = best_v - ctrl['valid_mean']
print(f"\nbest aux arm on VALIDATION: {best_name} ({best_v:.5f}, "
      f"{d_ctrl:+.5f} vs control; gate {GATE})")

if d_ctrl > GATE:
    lam, tgt = {n: (l, t) for n, l, t in ARMS}[best_name]
    fn = make_fn(lam, tgt)
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
             'name': f'R31d 5-seed committee of {best_name}',
             'valid_mean': round(float(v), 5), 'test_mean': round(float(t), 5),
             'test_std': None, 'd_baseline': round(float(t - BASELINE), 5),
             'verdict': 'WIN' if t - BASELINE > GATE else 'NOISE',
             'note': f'promoted arm; valid {v:.5f} vs banked valid '
                     f'{BANKED_VALID} -> {"BANK" if beats else "no bank"}'})
    print(f"R31d committee: valid {v:.5f} (banked {BANKED_VALID}) test {t:.4f} "
          f"-> {'BANK' if beats else 'NO BANK (incumbent wins ties)'}")
else:
    print("No aux arm clears the gate over the control on validation -> "
          "IDEAS #12 dies in auxiliary form too. No promotion.")