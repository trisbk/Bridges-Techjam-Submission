"""Autonomy driver: runs the research loop without human steering.

Each iteration launches a fresh Claude session to execute one research cycle
from ITERATION_PROMPT.md. The driver only handles orchestration, convergence,
logging, and crash recovery — it never chooses experiments or changes results.

The run stops after the configured convergence condition or MAX_ITERS.
Crash restarts are mechanical recovery, not strategic intervention.
"""
import json, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, 'logs', 'LOG.jsonl')
DRIVER_LOG = os.path.join(REPO, 'agent', 'driver_log.jsonl')
PROMPT = ('Read agent/ITERATION_PROMPT.md and execute exactly one research '
          'iteration as it specifies. You are running unattended; do not ask '
          'questions — make the best call and document it.')
EPS, N_CONV = 0.002, 3
WALL_CLOCK_S = 6 * 3600   # 6h wall-clock ceiling 
MAX_ITERS = 50            # per-run cap 
TIMEOUT_S = 2400          # 40 min per iteration, then it counts as a crash
RETRIES_PER_ITER = 2


def log_event(**kw):
    kw['ts'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with open(DRIVER_LOG, 'a') as fh:
        fh.write(json.dumps(kw) + '\n')
    print(f"[driver] {kw}")


HARNESS = os.path.join(REPO, 'code', 'harness.py')


def banked_best():
    import re
    with open(HARNESS) as fh:
        m = re.search(r'BANKED_VALID\s*=\s*([0-9.]+)', fh.read())
    return float(m.group(1)) if m else 0.61906


def run_iteration(i):
    import signal
    for attempt in range(1, RETRIES_PER_ITER + 2):
        log_event(event='iteration_start', iteration=i, attempt=attempt)
        out_path = os.path.join(REPO, 'agent',
                                f'session_i{i}_a{attempt}.out')
        try:
            with open(out_path, 'w') as out:
                p = subprocess.Popen(
                    ['claude', '-p', PROMPT, '--dangerously-skip-permissions'],
                    cwd=REPO, stdout=out, stderr=subprocess.STDOUT,
                    text=True, start_new_session=True)
                try:
                    rc = p.wait(timeout=TIMEOUT_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                    log_event(event='iteration_timeout', iteration=i,
                              attempt=attempt)
                    log_event(event='iteration_restart', iteration=i,
                              attempt=attempt)
                    continue
            with open(out_path) as fh:
                tail = fh.read()[-2000:]
            log_event(event='iteration_end', iteration=i, attempt=attempt,
                      returncode=rc, tail=tail)
            if rc == 0:
                return True
        except FileNotFoundError:
            log_event(event='fatal', error='claude CLI not found on PATH')
            sys.exit(1)
        # crash -> restart (webinar ruling: not human intervention)
        log_event(event='iteration_restart', iteration=i, attempt=attempt)
    log_event(event='iteration_abandoned', iteration=i)
    return False


def main():
    log_event(event='driver_start', eps=EPS, n_conv=N_CONV,
              max_iters=MAX_ITERS)
    below_eps = 0
    prev_best = banked_best()
    run_t0 = time.time()
    for i in range(1, MAX_ITERS + 1):
        if time.time() - run_t0 > WALL_CLOCK_S:
            log_event(event='wall_clock_ceiling', final_best=round(prev_best, 5),
                      rule='6h ceiling per problem statement')
            return
        ok = run_iteration(i)
        best = banked_best()
        gain = best - prev_best
        below_eps = 0 if gain > EPS else below_eps + 1
        log_event(event='convergence_check', iteration=i, ok=ok,
                  best=round(best, 5), gain=round(gain, 5),
                  consecutive_below_eps=below_eps)
        prev_best = best
        if below_eps >= N_CONV:
            log_event(event='converged', final_best=round(best, 5),
                      rule=f'{N_CONV} consecutive iterations < {EPS}')
            return
    log_event(event='max_iterations', final_best=round(prev_best, 5))


if __name__ == '__main__':
    main()
