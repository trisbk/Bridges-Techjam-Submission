"""Replay historical experiment decisions using validation only.

Ignores the legacy verdict labels in LOG.jsonl and reconstructs the banking
trajectory from validation scores using the current promotion margin. The
original log is never changed; corrected verdicts are written separately to
LOG-replay.jsonl.

This provides an independent check that validation-only selection still leads
to the shipped R33c champion.
"""
import json, os

LOG = os.path.join('..', 'logs', 'LOG.jsonl')
OUT = os.path.join('..', 'logs', 'LOG-replay.jsonl')
BASELINE_VALID = 0.6014
GATE = 0.002
PROMOTION_MARGIN = 0.001

champion = ('reproduced baseline', BASELINE_VALID, 0.5950)
trajectory = [champion]
out = []
skipped = 0

with open(LOG) as fh:
    for line in fh:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get('phase') != 'result':
            continue
        vm, tm = r.get('valid_mean'), r.get('test_mean')
        if vm is None:
            skipped += 1
            out.append({'name': r.get('name'), 'replay_verdict': 'NO_VALID_RECORDED'})
            continue
        d_base = vm - BASELINE_VALID
        if d_base > GATE and vm > champion[1] + PROMOTION_MARGIN:
            verdict = 'PROMOTE'
            champion = (r.get('name'), vm, tm)
            trajectory.append(champion)
        elif d_base > GATE:
            verdict = 'SIGNIFICANT_NOT_PROMOTED'
        elif d_base < -GATE:
            verdict = 'WORSE_THAN_BASELINE'
        else:
            verdict = 'NOISE'
        out.append({'name': r.get('name'), 'valid_mean': vm, 'test_mean': tm,
                    'replay_verdict': verdict,
                    'legacy_label': r.get('verdict')})

with open(OUT, 'w') as fh:
    for rec in out:
        fh.write(json.dumps(rec) + '\n')

print(f"replayed {len(out)} result records ({skipped} lacked a validation mean)")
print("\nchampion trajectory under the validation-only rule:")
for name, vm, tm in trajectory:
    print(f"  valid {vm:.5f} | test {tm if tm is None else round(tm,5)} | {name}")
final = trajectory[-1]
print(f"\nfinal champion: {final[0]}  (valid {final[1]:.5f}, test {final[2]:.5f})")
ok = abs(final[1] - 0.62059) < 1e-6
print("matches the shipped checkpoint: " + ("YES" if ok else "NO — investigate"))
