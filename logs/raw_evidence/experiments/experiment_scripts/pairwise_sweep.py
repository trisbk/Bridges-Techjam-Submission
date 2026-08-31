""" P1 Follow-Up: BPR Tuning

BPR improved over the baseline by +0.0017, but used settings originally tuned for 
pointwise training.

This follow-up tested different learning rates, pair counts, and patience values to 
find better settings for BPR. Each configuration was tested across three seeds for more 
reliable results.
"""
import time
import numpy as np
from data import load
from pairwise import run

if __name__ == '__main__':
    print("loading ./KuaiRand-Pure/data ...")
    splits = load('./KuaiRand-Pure/data')

    configs = [
        ("P1a lr=0.001 npp=8  pat=6", dict(lr=0.001, neg_per_pos=8, patience=6)),
        ("P1b lr=0.002 npp=4  pat=6", dict(lr=0.002, neg_per_pos=4, patience=6)),
        ("P1c lr=0.002 npp=8  pat=6", dict(lr=0.002, neg_per_pos=8, patience=6)),
        ("P1d lr=0.003 npp=8  pat=6", dict(lr=0.003, neg_per_pos=8, patience=6)),
    ]
    SEEDS = 3
    BASE = 0.5950   # our same-session baseline re-run (3 seeds)
    print(f"\n{SEEDS} seeds each. Same-session baseline = {BASE} | P1 (lr=0.001, npp=4, pat=4) = 0.5967\n")
    print(f"{'config':<28} {'valid':>10}   {'test primary':>20}   {'delta':>8}")
    for name, kw in configs:
        vs, ts = [], []
        t0 = time.time()
        for s in range(SEEDS):
            r = run(splits, 'bpr', seed=s, **kw)
            vs.append(r['valid']['primary']); ts.append(r['test']['primary'])
        vm, tm, tsd = np.mean(vs), np.mean(ts), np.std(ts)
        d = tm - BASE
        flag = "SIGNIFICANT" if d > 0.002 else ("noise" if abs(d) < 0.0016 else "marginal")
        print(f"{name:<28} {vm:>10.4f}   {tm:>12.4f} ± {tsd:.4f}   {d:+.4f}  {flag}  ({time.time()-t0:.0f}s)")