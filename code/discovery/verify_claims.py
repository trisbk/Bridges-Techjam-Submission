"""Verification script: independently recheck the project's key claims.

Verifies the official evaluation/data files, dataset split sizes, oracle
ceiling, test-user composition, random-scoring baseline, and the final
committee's seed-to-seed variation using the repository's recorded results.

This provides a single reproducible check of the main numbers reported in the
project.
"""
import hashlib, json, os
import numpy as np
from data import load
from evaluate import evaluate

print("1. Kit integrity (compared against the official archive shipped in")
print("   third_party/kuairand-starter-kit-official.zip, not constants)")
import zipfile
zf = zipfile.ZipFile(os.path.join('..', 'third_party',
                                  'kuairand-starter-kit-official.zip'))
for fname in ('evaluate.py', 'data.py', 'baseline.py', 'submit.py'):
    member = [n for n in zf.namelist() if n.endswith('/' + fname) or n == fname]
    official = zf.read(member[0])
    ours = open(fname, 'rb').read()
    same = hashlib.sha256(official).hexdigest() == hashlib.sha256(ours).hexdigest()
    print(f"   {fname}: " + ("IDENTICAL to official archive" if same
                             else "*** DIFFERS from official archive ***"))

print("\n2. Split sizes")
splits = load('./KuaiRand-Pure/data')
for name, rws in splits.items():
    print(f"   {name}: {len(rws):,} rows")

print("\n3. Oracle ceiling on test (score = true label)")
rws = splits['test']
users = [x[1] for x in rws]
labels = [x[6] for x in rws]
r = evaluate(users, labels, [float(y) for y in labels])
print(f"   GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
from collections import defaultdict
byu = defaultdict(list)
for u, y in zip(users, labels):
    byu[u].append(y)
zero = sum(1 for v in byu.values() if sum(v) == 0)
allp = sum(1 for v in byu.values() if sum(v) == len(v))
print(f"   users: {len(byu):,} | zero-positive {zero/len(byu):.1%} | all-positive {allp/len(byu):.1%}")

print("\n4. Random-scoring floor (5 seeds; official statement quotes 0.4753)")
floors = []
for sd in range(5):
    rng = np.random.default_rng(sd)
    floors.append(evaluate(users, labels, rng.random(len(labels)))['primary'])
print(f"   per-seed: {[round(f,4) for f in floors]}")
print(f"   mean {np.mean(floors):.4f}")

print("\n5. Final-recipe seed noise")
import sys
if '--full' in sys.argv:
    print("   recomputing the 5 committee members from shipped weights ...")
    from staleness_ablation import build_features, RICH
    from tab_surface import add_tab_features
    from sequences import encode_rows
    from baseline import FM
    sp = add_tab_features(build_features('continuous'))
    enc, dim = encode_rows(sp, RICH + ['tab_n'])
    Xte, yte, ute = enc['test']
    frozen = None
    for c in ('frozen_model', os.path.join('..', 'final_output',
                                           'frozen_model')):
        if os.path.exists(os.path.join(c, 'fm_seed0.npz')):
            frozen = c
            break
    singles = []
    for seed in range(5):
        z = np.load(os.path.join(frozen, f'fm_seed{seed}.npz'))
        m = FM(dim, k=16, seed=seed)
        m.V, m.W, m.b = z['V'], z['W'], np.float32(z['b'])
        singles.append(float(evaluate(ute, yte, m.predict(Xte))['primary']))
    singles = [round(x, 4) for x in singles]
else:
    singles = [0.6115, 0.6130, 0.6127, 0.6121, 0.6120]
    print("   (as printed by final_model.py; pass --full to recompute from")
    print("    the shipped weights, ~2 min)")
print(f"   5 committee members (test primary): {singles}")
print(f"   mean {np.mean(singles):.5f} | std {np.std(singles):.5f}")
if '--full' in sys.argv:
    print("\n6. Official baseline components (retraining 1 seed, ~1 min)")
    from baseline import run_fm
    r6 = run_fm(splits, seed=0, verbose=False)
    for spl in ('valid', 'test'):
        q = r6[spl]
        print(f"   {spl}: GAUC {q['GAUC']:.4f} | nDCG@5 {q['nDCG@5']:.4f} "
              f"| primary {q['primary']:.4f}")
    print("   (official statement: test GAUC 0.6610 / nDCG@5 0.5282 / 0.5946)")
log = os.path.join('..', 'logs', 'LOG.jsonl')
if os.path.exists(log):
    stds = []
    for line in open(log):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get('phase') == 'result' and rec.get('test_std'):
            stds.append(rec['test_std'])
    if stds:
        print(f"   per-run 3-seed stds in LOG.jsonl: n={len(stds)}, "
              f"median {np.median(stds):.5f}, max {max(stds):.5f}")
