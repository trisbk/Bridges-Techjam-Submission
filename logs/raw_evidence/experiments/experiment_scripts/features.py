""" Run 5: Feature Engineering

Run 5 tested additional features under the winning listwise objective (L1), since 
the original model only used five fields.

The new features included time of day (F1), content attributes (F2), user activity 
level (F3), and video popularity (F4). F5 combined the features that performed best.

To prevent data leakage, all popularity statistics and vocabularies were built using 
training data only, while missing or unseen values were assigned to an UNK category.
""" 
import csv, os, time, collections
import numpy as np
from evaluate import evaluate
from baseline import FM
from pairwise import build_pair_index
from listwise import sample_lists, infonce_step

DATA = './KuaiRand-Pure/data'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}


def load_rows():
    """Rows as feature dicts + label, split by date. Base + candidate fields."""
    vfeat = {}
    with open(os.path.join(DATA, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            tag = (r['tag'] or 'UNK').split(',')[0].strip() or 'UNK'
            vfeat[r['video_id']] = (r['author_id'] or 'UNK',
                                    r['video_type'] or 'UNK',
                                    r['music_type'] or 'UNK', tag)
    ufeat = {}
    with open(os.path.join(DATA, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            ufeat[r['user_id']] = r['user_active_degree'] or 'UNK'

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(DATA, f)) as fh:
            for r in csv.DictReader(fh):
                author, vtype, mtype, tag = vfeat.get(r['video_id'], ('UNK',) * 4)
                hour = str(int(r['hourmin']) // 100) if r['hourmin'] else 'UNK'
                rows.append({
                    'date': int(r['date']),
                    'user_id': r['user_id'], 'video_id': r['video_id'],
                    'author_id': author, 'tab': r['tab'],
                    'duration': float(r['duration_ms']),
                    'hour': hour, 'video_type': vtype, 'music_type': mtype,
                    'tag': tag, 'uactive': ufeat.get(r['user_id'], 'UNK'),
                    'y': 1 if r['long_view'] != '0' else 0,
                })
    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x['date'] <= hi]
    return out


def add_derived(splits, prior=20.0):
    """Train-only derived features: dur_bucket edges + video long_view rate."""
    tr = splits['train']
    edges = np.quantile(np.array([x['duration'] for x in tr]),
                        np.linspace(0, 1, 11)[1:-1])
    pos, imp = collections.Counter(), collections.Counter()
    for x in tr:
        imp[x['video_id']] += 1; pos[x['video_id']] += x['y']
    gmean = sum(pos.values()) / sum(imp.values())
    rates_tr = [(pos[v] + prior * gmean) / (imp[v] + prior) for v in imp]
    redges = np.quantile(np.array(rates_tr), np.linspace(0, 1, 11)[1:-1])

    def vpop(v):
        r = (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
        return str(int(np.searchsorted(redges, r)))

    for rws in splits.values():
        for x in rws:
            x['dur_bucket'] = str(int(np.searchsorted(edges, x['duration'])))
            x['vpop'] = vpop(x['video_id'])
    return splits


BASE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']


def encode_fields(splits, fields):
    tr = splits['train']
    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, f in enumerate(fields):
            v = x[f]
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)
    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, f in enumerate(fields):
                X[n, i] = vocabs[i].get(x[f], unk[i]) + offsets[i]
            y[n] = x['y']
            users.append(x['user_id'])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


def run_l1(enc, dim, K=4, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0):
    Xtr, ytr, utr = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    pairs_users, _, _ = build_pair_index(utr, ytr)
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        P, N = sample_lists(pairs_users, rng, K)
        for i in range(0, len(P), bs):
            p = P[i:i + bs]; n = N[i:i + bs]
            infonce_step(m, Xtr[p], Xtr[n.reshape(-1)], len(p), K)
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
            'test': evaluate(ute, yte, m.predict(Xte))}


if __name__ == '__main__':
    print(f"loading {DATA} ...")
    splits = add_derived(load_rows())
    print({k_: len(v) for k_, v in splits.items()})

    groups = [
        ("L1 base (control)",      BASE_FIELDS),
        ("F1 +hour",               BASE_FIELDS + ['hour']),
        ("F2 +content",            BASE_FIELDS + ['video_type', 'music_type', 'tag']),
        ("F3 +uactive",            BASE_FIELDS + ['uactive']),
        ("F4 +vpop",               BASE_FIELDS + ['vpop']),
    ]
    SEEDS = 3
    BASE = 0.5950
    L1 = 0.5978
    print(f"\n{SEEDS} seeds each, objective = InfoNCE K=4."
          f" Baseline {BASE} | L1 base fields {L1} (+0.0028)\n")
    print(f"{'config':<24} {'valid':>10}   {'test primary':>20}   {'d/base':>8} {'d/L1':>8}")
    for name, fields in groups:
        enc, dim = encode_fields(splits, fields)
        vs, ts = [], []
        t0 = time.time()
        for s in range(SEEDS):
            r = run_l1(enc, dim, seed=s)
            vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
        vm, tm, tsd = np.mean(vs), np.mean(ts), np.std(ts)
        print(f"{name:<24} {vm:>10.4f}   {tm:>12.4f} ± {tsd:.4f}   {tm-BASE:+.4f} {tm-L1:+.4f}  ({time.time()-t0:.0f}s)")