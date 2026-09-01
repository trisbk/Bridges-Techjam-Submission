"""KuaiRand-1K bonus: transfer the final Pure recipe without re-tuning.

Uses the same label, date-based splits, metrics, features, listwise objective,
FM configuration, and five-seed committee as the Pure model. A standard
pointwise FM provides the comparison baseline.

For the larger dataset, loading and embedding updates are made more
memory-efficient without changing the feature definitions. Model selection
remains validation-only, with test evaluated once per arm at the end.

Usage: python3 bonus_1k.py
"""
import csv, os, time
import numpy as np
from evaluate import evaluate

DATA = '../KuaiRand-1K/data'
TRAIN_END, VALID_END = 20220421, 20220428


def load_logs():
    t0 = time.time()
    vid2author, vid2tag = {}, {}
    with open(os.path.join(DATA, 'video_features_basic_1k.csv')) as fh:
        for r in csv.DictReader(fh):
            vid2author[r['video_id']] = r['author_id']
            vid2tag[r['video_id']] = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'

    users, vids, dates, ts, ys, tabs, durs = [], [], [], [], [], [], []
    uidx, vindex = {}, {}
    for f in ('log_standard_4_08_to_4_21_1k.csv',
              'log_standard_4_22_to_5_08_1k.csv'):
        with open(os.path.join(DATA, f)) as fh:
            rd = csv.reader(fh)
            head = next(rd)
            c = {k: i for i, k in enumerate(head)}
            iu, iv, idt, it = c['user_id'], c['video_id'], c['date'], c['time_ms']
            iy, itab, idur = c['long_view'], c['tab'], c['duration_ms']
            for row in rd:
                u = row[iu]
                users.append(uidx.setdefault(u, len(uidx)))
                v = row[iv]
                vids.append(vindex.setdefault(v, len(vindex)))
                dates.append(int(row[idt])); ts.append(int(row[it]))
                ys.append(0 if row[iy] == '0' else 1)
                tabs.append(int(row[itab]))
                durs.append(float(row[idur] or 0.0))

    n = len(users)
    user = np.array(users, dtype=np.int32); del users
    vid = np.array(vids, dtype=np.int32); del vids
    date = np.array(dates, dtype=np.int32); del dates
    t = np.array(ts, dtype=np.int64); del ts
    y = np.array(ys, dtype=np.int8); del ys
    tab = np.array(tabs, dtype=np.int8); del tabs
    dur = np.array(durs, dtype=np.float32); del durs

    # per-video author / tag as index arrays
    aidx, tgidx = {}, {}
    v_author = np.zeros(len(vindex), dtype=np.int32)
    v_tag = np.zeros(len(vindex), dtype=np.int32)
    for v, i in vindex.items():
        v_author[i] = aidx.setdefault(vid2author.get(v, 'UNK'), len(aidx))
        v_tag[i] = tgidx.setdefault(vid2tag.get(v, 'UNK'), len(tgidx))
    print(f"loaded {n:,} rows | {len(uidx)} users | {len(vindex):,} videos | "
          f"{len(aidx):,} authors | {len(tgidx)} tags | {time.time()-t0:.0f}s")
    return user, vid, date, t, y, tab, dur, v_author, v_tag


# ---------- featurization (one chronological pass, same definitions) ----------

def featurize(user, vid, date, t, y, tab, dur, v_author, v_tag):
    t0 = time.time()
    n = len(user)
    order = np.lexsort((t, date, user))
    author = v_author[vid]; tag = v_tag[vid]

    prev1 = np.zeros(n, np.int16); hist10 = np.zeros(n, np.int16)
    hist_n = np.zeros(n, np.int16); auth_h = np.zeros(n, np.int16)
    hist30 = np.zeros(n, np.int16); tag_h = np.zeros(n, np.int16)
    gap = np.zeros(n, np.int16); tab_n = np.zeros(n, np.int16)

    def nbucket(c):
        return (0 if c == 0 else 1 if c <= 3 else 2 if c <= 10
                else 3 if c <= 30 else 4 if c <= 100 else 5)

    st = {}          # user -> mutable state
    ua, ut, utab = {}, {}, {}     # (user,author)->pos count, (user,tag)->pos count, (user,tab)->count
    for i in order:
        u = int(user[i])
        s = st.get(u)
        if s is None:
            # prev, deque10(list), n, deque30(list), last_date, last_t
            s = st[u] = [None, [], 0, [], None, None]
        yv = int(y[i])
        prev1[i] = 0 if s[0] is None else 1 + s[0]
        d10 = s[1]
        hist10[i] = 0 if not d10 else 1 + sum(d10)
        hist_n[i] = nbucket(s[2])
        ka = (u, int(author[i]))
        a = ua.get(ka, 0)
        auth_h[i] = a if a < 3 else 3
        d30 = s[3]
        hist30[i] = 0 if not d30 else 1 + int(10 * sum(d30) / len(d30))
        kt = (u, int(tag[i]))
        tc = ut.get(kt, 0)
        tag_h[i] = tc if tc < 3 else 3
        if s[4] is None:
            gap[i] = 0
        else:
            dms = (int(date[i]) - s[4]) * 86400_000 + (int(t[i]) - s[5])
            gap[i] = (1 if dms < 60_000 else 2 if dms < 3_600_000
                      else 3 if dms < 86_400_000 else 4)
        kb = (u, int(tab[i]))
        nb = utab.get(kb, 0)
        tab_n[i] = nbucket(nb)
        utab[kb] = nb + 1
        # update AFTER featurizing (own label never in own features)
        s[0] = yv
        d10.append(yv)
        if len(d10) > 10: d10.pop(0)
        s[2] += 1
        d30.append(yv)
        if len(d30) > 30: d30.pop(0)
        ua[ka] = a + yv
        ut[kt] = tc + yv
        s[4] = int(date[i]); s[5] = int(t[i])
    print(f"featurized in {time.time()-t0:.0f}s")

    tr = date <= TRAIN_END
    edges = np.quantile(dur[tr], np.linspace(0, 1, 11)[1:-1])
    dur_b = np.searchsorted(edges, dur).astype(np.int16)
    return {'user': user, 'video': vid, 'author': author, 'tab': tab.astype(np.int16),
            'dur_bucket': dur_b, 'prev1': prev1, 'hist10': hist10,
            'hist_n': hist_n, 'auth_hist': auth_h, 'hist30': hist30,
            'tag_hist': tag_h, 'gap': gap, 'tab_n': tab_n}


def encode(fields, cols, date, min_count=2):
    """Train-vocab encoding with an UNK slot per field; ids below
    min_count train occurrences share UNK (vocab cap for the 4M-video
    tail; fields with small ranges keep every value)."""
    tr = date <= TRAIN_END
    offsets, dims, Xcols = [], [], []
    off = 0
    for name in cols:
        f = cols_data = fields[name]
        if name in ('video', 'author'):
            vals, counts = np.unique(f[tr], return_counts=True)
            keep = vals[counts >= min_count]
        else:
            keep = np.unique(f[tr])
        remap = np.full(int(f.max()) + 1, len(keep), dtype=np.int32)  # UNK
        remap[keep] = np.arange(len(keep), dtype=np.int32)
        Xcols.append(remap[f] + off)
        d = len(keep) + 1
        offsets.append(off); dims.append(d); off += d
    X = np.stack(Xcols, axis=1).astype(np.int32)
    return X, off, dims


# ---------- sparse-update FM ----------

class SparseFM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = (rng.normal(0, 0.05, (dim, k))).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.lr, self.l2, self.t = lr, l2, 0

    def logits(self, X):
        E = self.V[X]; S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def _apply(self, rows, gV_rows, gW_rows):
        b1, b2, eps = 0.9, 0.999, 1e-8
        t = self.t
        for P, G, M, Vv in ((self.V, gV_rows, self.mV, self.vV),
                            (self.W, gW_rows, self.mW, self.vW)):
            M[rows] = b1 * M[rows] + (1 - b1) * G
            Vv[rows] = b2 * Vv[rows] + (1 - b2) * (G * G)
            P[rows] -= self.lr * (M[rows] / (1 - b1 ** t)) / \
                (np.sqrt(Vv[rows] / (1 - b2 ** t)) + eps)

    def _accumulate(self, X, gper):
        """gper: (B,F) per-position gradient scale; returns unique rows and
        their summed V/W gradients."""
        flat = X.ravel()
        rows, inv = np.unique(flat, return_inverse=True)
        gW = np.zeros(len(rows), dtype=np.float32)
        np.add.at(gW, inv, gper.ravel())
        return rows, inv, gW

    def step_pointwise(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        g = ((1 / (1 + np.exp(-z)) - y) / B).astype(np.float32)
        gvec = (g[:, None, :] if False else None)
        per = np.broadcast_to(g[:, None], X.shape)
        rows, inv, gW = self._accumulate(X, per)
        gVfull = g[:, None, None] * (S[:, None, :] - E)      # (B,F,k)
        gV = np.zeros((len(rows), self.V.shape[1]), dtype=np.float32)
        np.add.at(gV, inv, gVfull.reshape(-1, self.V.shape[1]))
        gV += self.l2 * self.V[rows]; gW += self.l2 * self.W[rows]
        self.t += 1
        self._apply(rows, gV, gW)
        self.b -= self.lr * g.sum()

    def step_listwise(self, Xp, Xn_flat, B, K):
        """InfoNCE over 1 positive + K negatives, same maths as
        listwise.infonce_step, sparse application."""
        Xall = np.concatenate([Xp, Xn_flat.reshape(B * K, -1)], axis=0)
        z, E, S = self.logits(Xall)
        zp, zn = z[:B], z[B:].reshape(B, K)
        m = np.maximum(zp, zn.max(1))
        ep = np.exp(zp - m); en = np.exp(zn - m[:, None])
        denom = ep + en.sum(1)
        gp = ((ep / denom - 1.0) / B).astype(np.float32)     # dL/dzp
        gn = ((en / denom[:, None]) / B).astype(np.float32)  # dL/dzn
        g = np.concatenate([gp, gn.ravel()])
        per = np.broadcast_to(g[:, None], Xall.shape)
        rows, inv, gW = self._accumulate(Xall, per)
        gVfull = g[:, None, None] * (S[:, None, :] - E)
        gV = np.zeros((len(rows), self.V.shape[1]), dtype=np.float32)
        np.add.at(gV, inv, gVfull.reshape(-1, self.V.shape[1]))
        gV += self.l2 * self.V[rows]; gW += self.l2 * self.W[rows]
        self.t += 1
        self._apply(rows, gV, gW)
        self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0]
                               for i in range(0, len(X), bs)])


# ---------- list sampling (same as listwise.sample_lists, array version) ----------

def build_user_pools(utr, ytr):
    pools = {}
    for i in range(len(utr)):
        p = pools.setdefault(int(utr[i]), ([], []))
        p[ytr[i]].append(i)          # p[0]=negs, p[1]=poss
    return [ (np.array(pos, np.int64), np.array(neg, np.int64))
             for neg, pos in (pools[u] for u in pools)
             if len(pos) and len(neg) ]


def sample_lists_np(pools, rng, K):
    P, N = [], []
    for pos, neg in pools:
        P.append(pos)
        N.append(neg[rng.integers(0, len(neg), size=(len(pos), K))])
    return np.concatenate(P), np.concatenate(N, axis=0)


# ---------- training loops ----------

def train_eval(X, dim, y, user, date, mode, seed, k=16, lr=0.001,
               epochs=40, bs=8192, patience=4, K=4):
    tr = date <= TRAIN_END
    va = (date > TRAIN_END) & (date <= VALID_END)
    te = date > VALID_END
    Xtr, ytr = X[tr], y[tr].astype(np.float32)
    # the kit's evaluate() does python-int arithmetic; int8 overflows it
    Xva, yva, uva = X[va], y[va].astype(int).tolist(), user[va].tolist()
    Xte, yte, ute = X[te], y[te].astype(int).tolist(), user[te].tolist()
    utr = user[tr]
    m = SparseFM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    if mode == 'listwise':
        pools = build_user_pools(utr, y[tr])
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        if mode == 'pointwise':
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), bs):
                m.step_pointwise(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
        else:
            P, N = sample_lists_np(pools, rng, K)
            perm = rng.permutation(len(P))
            P, N = P[perm], N[perm]
            for i in range(0, len(P), bs):
                p = P[i:i + bs]; nn = N[i:i + bs]
                m.step_listwise(Xtr[p], Xtr[nn.reshape(-1)], len(p), K)
        r = evaluate(uva, yva, m.predict(Xva))
        print(f"    ep {ep:2d} valid {r['primary']:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        if r['primary'] > best + 1e-5:
            best, bad = r['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    pv, pt = m.predict(Xva), m.predict(Xte)
    return (evaluate(uva, yva, pv), evaluate(ute, yte, pt),
            (pv - pv.mean()) / pv.std(), (pt - pt.mean()) / pt.std(),
            (uva, yva, ute, yte))


BASE = ['user', 'video', 'author', 'tab', 'dur_bucket']
RICH_TAB = BASE + ['prev1', 'hist10', 'hist_n', 'auth_hist',
                   'hist30', 'tag_hist', 'gap', 'tab_n']

CACHE = '../bonus_cache.npz'

if __name__ == '__main__':
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        F = {k: z[k] for k in z.files if k not in ('date', 'y')}
        date, y = z['date'], z['y']
        user = F['user']
        print(f"loaded featurized cache ({len(y):,} rows)")
    else:
        user, vid, date, t, y, tab, dur, v_author, v_tag = load_logs()
        F = featurize(user, vid, date, t, y, tab, dur, v_author, v_tag)
        np.savez(CACHE, **F, date=date, y=y)
        print("featurized arrays cached")
    tr = date <= TRAIN_END
    va = (date > TRAIN_END) & (date <= VALID_END)
    te = date > VALID_END
    print(f"splits: train {tr.sum():,} | valid {va.sum():,} | test {te.sum():,} "
          f"| long_view rate train {y[tr].mean():.3f}")

    print("\n=== BASELINE ARM (kit recipe: pointwise FM, base fields) ===")
    Xb, dimb, _ = encode(F, BASE, date)
    print(f"dim {dimb:,}")
    b_tests = []
    for seed in range(3):
        rv, rt, _, _, _ = train_eval(Xb, dimb, y, user, date, 'pointwise', seed)
        b_tests.append(rt['primary'])
        print(f"  baseline seed {seed}: valid {rv['primary']:.4f} "
              f"test {rt['primary']:.4f}")
    print(f"BASELINE test mean over 3 seeds: {np.mean(b_tests):.4f} "
          f"± {np.std(b_tests):.4f}")

    print("\n=== OUR RECIPE (listwise, causal features + tab_n, 5-seed committee) ===")
    Xr, dimr, _ = encode(F, RICH_TAB, date)
    print(f"dim {dimr:,}")
    va_ps, te_ps, singles = [], [], []
    ref = None
    for seed in range(5):
        rv, rt, zva, zte, ref = train_eval(Xr, dimr, y, user, date,
                                           'listwise', seed)
        va_ps.append(zva); te_ps.append(zte); singles.append(rt['primary'])
        print(f"  ours seed {seed}: valid {rv['primary']:.4f} "
              f"test {rt['primary']:.4f}")
    uva, yva, ute, yte = ref
    cv = evaluate(uva, yva, np.mean(va_ps, 0))
    ct = evaluate(ute, yte, np.mean(te_ps, 0))
    print(f"\n=== KUAIRAND-1K RESULT (untuned transfer) ===")
    print(f"baseline (3-seed mean)  : test {np.mean(b_tests):.4f}")
    print(f"ours singles            : {[round(s,4) for s in singles]}")
    print(f"ours committee  valid   : GAUC {cv['GAUC']:.4f} nDCG@5 "
          f"{cv['nDCG@5']:.4f} primary {cv['primary']:.4f}")
    print(f"ours committee  test    : GAUC {ct['GAUC']:.4f} nDCG@5 "
          f"{ct['nDCG@5']:.4f} primary {ct['primary']:.4f}")
    print(f"delta vs our 1K baseline: {ct['primary'] - np.mean(b_tests):+.4f}")
