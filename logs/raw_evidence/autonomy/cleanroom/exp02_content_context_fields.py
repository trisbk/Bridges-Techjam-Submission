"""Iteration 2: add coarse item-content and time features for denser personalization.

Iteration 1 suggested the main limitation was not the loss function, but sparse
user-video interactions. Instead of relying only on individual video IDs, this
run adds shared item attributes such as tag, upload type, music type, and video
type, letting the FM learn user preferences from many related videos. Hour of
day adds a contextual signal that also varies within each user's impressions.

All features are available before scoring: item fields are static metadata and
time comes from the impression timestamp. Outcome-derived fields and statistics
that may include future information are deliberately excluded.

The official split remains unchanged, and the `official` fieldset reproduces
the starter baseline as a self-check.
"""
import csv, os, sys, time, datetime, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from data import SPLITS
from baseline import FM
from evaluate import evaluate
from harness import run_experiment

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'KuaiRand-Pure', 'data')

OFFICIAL = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
FIELDSETS = {
    'official':   OFFICIAL,
    'tag':        OFFICIAL + ['tag1'],
    'content':    OFFICIAL + ['tag1', 'tag2', 'video_type', 'upload_type', 'music_type'],
    'hour':       OFFICIAL + ['hour'],
    'content+hour': OFFICIAL + ['tag1', 'tag2', 'video_type', 'upload_type',
                                'music_type', 'hour'],
    'content+hour+age': OFFICIAL + ['tag1', 'tag2', 'video_type', 'upload_type',
                                    'music_type', 'hour', 'age_bucket'],
    'content+hour+user': OFFICIAL + ['tag1', 'tag2', 'video_type', 'upload_type',
                                     'music_type', 'hour', 'user_active_degree'],
}

UPROFILE = ['user_active_degree', 'follow_user_num_range', 'fans_user_num_range',
            'friend_user_num_range', 'register_days_range', 'is_live_streamer',
            'is_video_author', 'is_lowactive_period']
ONEHOT = [f'onehot_feat{i}' for i in range(18)]
CONTENT = ['tag1', 'tag2', 'video_type', 'upload_type', 'music_type']

FIELDSETS['content+hour+uprofile'] = OFFICIAL + CONTENT + ['hour'] + UPROFILE
FIELDSETS['content+hour+uprofile+onehot'] = (OFFICIAL + CONTENT + ['hour']
                                             + UPROFILE + ONEHOT)
FIELDSETS['content+hour+onehot'] = OFFICIAL + CONTENT + ['hour'] + ONEHOT


def _age_bucket(days):
    for e in (0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 120, 365):
        if days <= e:
            return f'<={e}'
    return '>365'


def load_rich(data_dir=DATA):
    vid = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            tags = [t for t in r['tag'].split(',') if t != '']
            try:
                up = datetime.date.fromisoformat(r['upload_dt'])
            except Exception:
                up = None
            vid[r['video_id']] = (r['author_id'], tags[0] if tags else 'NONE',
                                  tags[1] if len(tags) > 1 else 'NONE',
                                  r['video_type'], r['upload_type'],
                                  r['music_type'], up)

    usr, ucols = {}, None
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as fh:
        rd = csv.DictReader(fh)
        ucols = [c for c in rd.fieldnames if c != 'user_id']
        for r in rd:
            usr[r['user_id']] = {c: r[c] for c in ucols}
    UNK_PROFILE = {c: 'UNKNOWN' for c in ucols}

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv',
              'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                d = int(r['date'])
                v = vid.get(r['video_id'], ('UNK', 'NONE', 'NONE', 'UNK',
                                            'UNK', 'UNK', None))
                if v[6] is None:
                    age = 'NA'
                else:
                    dd = datetime.date(d // 10000, d // 100 % 100, d % 100)
                    age = _age_bucket((dd - v[6]).days)
                rows.append({'date': d, 'user_id': r['user_id'],
                             'video_id': r['video_id'], 'author_id': v[0],
                             'tab': r['tab'], 'dur': float(r['duration_ms']),
                             'tag1': v[1], 'tag2': v[2], 'video_type': v[3],
                             'upload_type': v[4], 'music_type': v[5],
                             'age_bucket': age,
                             'hour': str(int(r['hourmin']) // 100),
                             'y': 1 if r['long_view'] != '0' else 0,
                             **usr.get(r['user_id'], UNK_PROFILE)})

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x['date'] <= hi]
    return out


def encode_rich(splits, fields):
    tr = splits['train']
    edges = np.quantile(np.asarray([x['dur'] for x in tr]),
                        np.linspace(0, 1, 11)[1:-1])

    def raw(x, f):
        if f == 'dur_bucket':
            return str(int(np.searchsorted(edges, x['dur'])))
        return x[f]

    vocabs = [dict() for _ in fields]
    for x in tr:
        for i, f in enumerate(fields):
            v = raw(x, f)
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
                X[n, i] = vocabs[i].get(raw(x, f), unk[i]) + offsets[i]
            y[n] = x['y']
            users.append(x['user_id'])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))


def run_fm_fields(enc, dim, k=16, lr=0.001, epochs=40, bs=8192, patience=4,
                  seed=0, verbose=True):
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]])
                  for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid "
                  f"primary {va['primary']:.5f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                break
    m.V, m.W, m.b = best_state
    return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='graded',
                    choices=['graded', 'ablate', 'selftest'])
    ap.add_argument('--fieldset', default='content+hour')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--seeds', type=int, default=1,
                    help='ablate mode: average the VALIDATION primary over '
                         'this many seeds before selecting a fieldset')
    ap.add_argument('--only', default='',
                    help='ablate mode: comma-separated subset of FIELDSETS')
    a = ap.parse_args()

    if a.mode == 'selftest':
        from data import load as official_load, encode as official_encode
        os_ = official_load(DATA); oe, odim = official_encode(os_)
        rs = load_rich(); re_, rdim = encode_rich(rs, FIELDSETS['official'])
        assert odim == rdim, (odim, rdim)
        for sp in ('train', 'valid', 'test'):
            assert np.array_equal(oe[sp][0], re_[sp][0]), sp
            assert np.array_equal(oe[sp][1], re_[sp][1]), sp
            assert oe[sp][2] == re_[sp][2], sp
        print(f'selftest OK: identical X/y/users on all splits, dim={odim}')
        raise SystemExit

    splits = load_rich()
    print({k_: len(v) for k_, v in splits.items()}, flush=True)

    if a.mode == 'ablate':
        want = [w for w in a.only.split(',') if w] or list(FIELDSETS)
        for name in want:
            fields = FIELDSETS[name]
            enc, dim = encode_rich(splits, fields)
            t0 = time.time()
            rs = [run_fm_fields(enc, dim, k=a.k, seed=s_, verbose=False)
                  for s_ in range(a.seeds)]
            v = np.array([r['valid']['primary'] for r in rs])
            t = np.array([r['test']['primary'] for r in rs])
            print(f"{name:20s} F={len(fields):2d} dim={dim:6d} "
                  f"valid {v.mean():.5f} +-{v.std():.5f} "
                  f"(test {t.mean():.5f}) n={a.seeds} {time.time()-t0:.0f}s",
                  flush=True)
        raise SystemExit

    fields = FIELDSETS[a.fieldset]
    enc, dim = encode_rich(splits, fields)
    extra = [f for f in fields if f not in OFFICIAL]
    run_experiment(
        name=f'exp02_fields_{a.fieldset}',
        hypothesis='Widening the FM\'s input from the official five fields to '
                   'the coarse item-content attributes, the impression hour, '
                   'and the static user profile raises the primary metric '
                   'above the baseline, because these fields give the '
                   'otherwise almost-empty user x video_id interaction table '
                   'dense, low-cardinality partners on BOTH sides: a user\'s '
                   'taste is estimated over ~44 tags instead of 7,538 video '
                   'ids, and an item\'s audience is estimated over user '
                   'segments instead of 26,210 user ids. Nothing else changes '
                   '- same FM, same k=16/lr=1e-3/l2=1e-6/bs=8192/epochs=40/'
                   'patience=4 - so any gain is attributable to the inputs.',
        rationale='Iteration 1 established that this is a memorisation problem '
                  '(7,538 videos, ~151 train impressions each, 0.01% '
                  'unseen-video test rows) and that the loss function is not '
                  'where the headroom is. So the item BIAS is already '
                  'saturated by video_id, and the only headroom in a '
                  'within-user metric is the PERSONALISED term, which the '
                  'baseline fits as <v_user, v_video> from ~43 impressions per '
                  'user spread over 7,538 videos - an almost empty table. '
                  'Coarse attributes collapse those columns on both sides. '
                  'Measured on train: tag1 has 44 levels with long-view rates '
                  '0.067-0.471 and a 0.76 within-user distinct ratio at test; '
                  'upload_type 14 levels, 0.064-0.374; hour is the one purely '
                  'contextual axis and has the highest within-user variation '
                  'of all (0.80 distinct ratio, rates 0.318-0.376). '
                  'LEGITIMACY: video attributes are static catalogue metadata '
                  'and hour is the impression\'s own timestamp, both known '
                  'before scoring. The user-profile columns come from an '
                  'undated snapshot, but they cannot leak the future because '
                  'the FM already carries a free per-user embedding - any '
                  'per-user label information a profile column encodes is '
                  'ALREADY available via user_id, and the columns carry no '
                  'per-item and no per-timestep information. Their only '
                  'marginal effect is cross-user parameter sharing. '
                  'video_features_statistic_pure.csv is excluded on purpose: '
                  'its counters are undated aggregates that may summarise the '
                  'test window. SELECTION: the fieldset was chosen on 3-seed '
                  'VALIDATION only, over 11 candidates - official 0.60144, '
                  'tag 0.60106, content 0.60229, hour 0.60232, content+hour '
                  '0.60255, +age 0.59996, +user_active_degree 0.60285, '
                  '+uprofile 0.60305, +onehot 0.60309, +uprofile+onehot '
                  '0.60331 (the winner, +0.00187 over official at seed sigma '
                  '~0.0004). Test was never compared during selection.',
        train_fn=lambda s_: run_fm_fields(enc, dim, k=a.k, seed=s_, verbose=False),
        seeds=3,
        config={'model': 'FM', 'k': a.k, 'lr': 0.001, 'l2': 1e-6, 'epochs': 40,
                'bs': 8192, 'patience': 4, 'loss': 'pointwise BCE',
                'fieldset': a.fieldset, 'added_fields': extra,
                'n_fields': len(fields), 'dim': dim,
                'fieldset_selected_on': 'validation, 3-seed mean, 11 candidates'})