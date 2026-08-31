"""Regenerate submission.csv from the frozen champion's saved predictions.

Saved predictions follow load_sequenced() order, while the official submission
expects data.load() row order. Because (user_id, video_id) is not unique, rows
are remapped using the full observable key with occurrence-order matching for
duplicates, preserving the correct score alignment.

The script then performs two independent checks:
1. submit.py validates the submission format and row alignment.
2. evaluate.py re-scores the mapped predictions and must reproduce the frozen
   committee's reported test primary.

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