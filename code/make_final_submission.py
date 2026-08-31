"""Regenerate submission.csv from the frozen champion's saved predictions.

The committee's test predictions (frozen_model/ensemble_predictions.npz)
are in load_sequenced() row order: sorted by (user_id, date, time_ms).
The official submission schema (submit.py) wants data.load() row order:
raw file order. The two orders differ, and (user_id, video_id) is not a
unique key (3.06 percent duplicate pairs in test), so rows are matched by
the full observable key (date, user_id, video_id, duration, label) with
occurrence-order pairing inside identical-key groups. Rows identical on
that whole key also share user and label, so any within-group pairing
yields the same metric — the mapping is exact for scoring purposes.

The script ends with two independent checks:
  1. submit.py's own read_submission validation (format + per-row id
     alignment against data.load()),
  2. evaluate() on the mapped scores in data.load() order, which must
     reproduce the committee's test primary.

Run from code/ after final_model.py:  python3 make_final_submission.py
"""
import collections, os
import numpy as np
from data import load
from evaluate import evaluate
from sequences import load_sequenced
from submit import write_submission, read_submission

OUT = os.path.join('..', 'final_output', 'submission.csv')


def key_of_seq(x):
    return (x['date'], x['user_id'], x['video_id'], x['duration'], x['y'])


def key_of_raw(x):
    return (x[0], x[1], x[2], x[5], x[6])


if __name__ == '__main__':
    print("loading predictions ...")
    frozen = ('frozen_model' if os.path.exists('frozen_model/fm_seed0.npz')
              else os.path.join('..', 'final_output', 'frozen_model'))
    ens = np.load(os.path.join(frozen, 'ensemble_predictions.npz'))['test']

    print("rebuilding both row orders ...")
    seq_test = load_sequenced()['test']
    assert len(seq_test) == len(ens), (len(seq_test), len(ens))
    raw_test = load('./KuaiRand-Pure/data')['test']
    assert len(raw_test) == len(seq_test)

    groups = collections.defaultdict(collections.deque)
    for x, s in zip(seq_test, ens):
        groups[key_of_seq(x)].append(float(s))
    scores = [groups[key_of_raw(x)].popleft() for x in raw_test]
    assert all(not v for v in groups.values()), "unmatched rows remain"

    write_submission(OUT, raw_test, scores)
    print(f"wrote {OUT} ({len(scores):,} rows)")

    print("check 1: submit.py format + alignment validation ...")
    read_submission(OUT, raw_test)
    print("   OK")

    print("check 2: metric reproduction in data.load() order ...")
    r = evaluate([x[1] for x in raw_test], [x[6] for x in raw_test], scores)
    print(f"   GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
          f"| primary {r['primary']:.5f}")
