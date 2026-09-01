# Run and Iteration Logs

This document indexes every research iteration against the Track 2 run-log
requirements. Three layers of record back it:

- `LOG.jsonl` . Machine log: intent (hypothesis, rationale, config) is
 appended before training. Results (per-seed scores, GAUC/nDCG
 components for later runs, result, wall time) after. A crashed run
 therefore still documents what it was attempting.
- `RESULTS_CODE.md` . The narrative: full tables and mechanism analysis per run.
- Git history . The code diff applied per iteration. Runs 30-32 were
 committed by the agent itself in this repo (hashes below). The manual-phase
 history lives in the research workspace repo
 (github.com/SteveWilsonK/techjam-2026-workspace), whose commits are cited
 per run. Each run's experiment code is also preserved verbatim as a
 standalone script in `src/`.

 primary = mean(GAUC, nDCG@5). Published baseline 0.5946. our reproduced baseline 0.5950. Seed noise σ≈0.0008. Significance gate
0.002. Every entry ≥3 seeds. Final component scores: GAUC 0.6825,
nDCG@5 0.5408 (valid: 0.6926 / 0.5455).

## Manual interventions summary

**Autonomous demonstration run (Runs 30-32, ): zero
manual interventions, zero crash-restarts.** The driver
(`agent/driver.py`) looped fresh headless agent sessions to convergence
(ε=0.002, N=3) with no human input after launch. The launch itself was
human-initiated by design. Event log: `agent/driver_log.jsonl`.

Interactive research phase (Runs 1-29): the agent (the agent in an
interactive session) made all iteration-level research decisions . What to
try, how to implement, how to interpret, when an idea was dead.

*Loop-relevant interventions (counted): 3*
1. Deferring three idea families (sequence features, multi-task labels,
 random-exposure log) . Between Runs 13→14. Constrained the search space.
2. Permitting them . Between Runs 17→18. Directly enabled the Run 18
 breakthrough. The single most consequential human decision.
3. Setting an additional stopping rule (stop at 0.65 or after 5 unbanked
 runs) . Before Run 26. Governed when exploration ended.

*Administrative actions (not loop interventions):* competition-track
selection (before any agent loop existed) and a report-formatting request.
No human proposed, implemented, tuned, or interpreted any experiment.
Restart-after-interruption occurred once (Run 6, lid-close SIGTERM) . Per
the organizers' ruling, restarts are not interventions.

## Error and recovery events

| Run | Event | Handling |
|---|---|---|
| Run 6 | External SIGTERM (laptop lid closed) killed the process after config 1 of 3 | Completed config's result was already logged. remaining configs relaunched. nothing lost |
| Run 9, attempt 1 | Harness crash writing results (numpy float32 not JSON-serializable) after training completed | Intent record survived as designed. harness fixed (float casts). run repeated |
| Driver false start | All iterations failed: CLI not authenticated ("Not logged in") | Detected. log archived as `agent/driver_log.failed-start.jsonl` (kept for the record). authenticated. relaunched cleanly |
| run | . | No errors, no restarts |

## Per-iteration index

Phase 1 . Interactive research (agent-driven, human-strategic).
Full detail per run in RESULTS_CODE.md. Code in `src/`. Workspace-repo commit
in brackets.

| Run | Hypothesis (short) | Code | test primary | Δ base | Result |
|---|---|---|---|---|---|
| 1-B1 | Metric weights users equally → reweight rows 1/impressions | experiments.py [2a1056f] | 0.5888 ± 0.0008 | −0.0062 | GAUC is positive-count-weighted. helping nDCG hurts GAUC more |
| 1-B2 | Train on continuous play-ratio instead of binary label | experiments.py [2a1056f] | 0.5598 ± 0.0002 | −0.0352 | duration-mediated → learns "prefer short videos" |
| 2-P1 | Pairwise BPR matches the ranking metric | pairwise.py [900a9b9] | 0.5967 ± 0.0005 | +0.0017 | real direction, under gate |
| 2-P2 | BPR + pointwise blend | pairwise.py [900a9b9] | 0.5949 ± 0.0005 | −0.0001 | pointwise erases the gain |
| 3 | BPR gain scales with lr / pair count (4 configs) | pairwise_sweep.py [900a9b9] | best 0.5965 | +0.0015 | capped. lr ≥ 2e-3 destructive |
| 4-L1 | Listwise InfoNCE upweights hard negatives | listwise.py [06e2a95] | 0.5978 ± 0.0002 | +0.0028 | first banked win |
| 4-L2 | Bigger K (8) helps | listwise.py [06e2a95] | 0.5977 ± 0.0002 | +0.0027 | K saturated at 4 |
| 4-C0/C1 | Capacity k=32 (with pointwise control) | listwise.py [06e2a95] | 0.5965 / 0.5950 | . | capacity does nothing. gain attributed to objective |
| 5 | Side features stack with the objective (4 groups) | features.py [4c8e5ce] | ~0.5981 each | +0.0031 | +0.0003 each, noise-level |
| 6-F5 | The four groups stack combined | features2.py [4c8e5ce] | 0.5981 ± 0.0001 | +0.0031 | redundant, no stacking |
| 6-F6/F7 | User×author/tag affinity rates | features2.py [4c8e5ce] | 0.5858 ± 0.0010 | −0.0092 | self-inclusion leakage diagnosed |
| 7 | Leave-one-out fixes the affinity leakage | (inline) [5057b90] | 0.5970 ± 0.0003 | +0.0020 | leak confirmed as mechanism. honest version redundant with FM |
| 8 | Seed ensembling cancels noise | ensemble seeds [5057b90] | 0.5982 (5 seeds) | +0.0032 | banked. +0.0004 only . FM seeds converge |
| 9 | FwFM: field-pair weights matter | fwfm.py [e7ead3e] | 0.5978 ± 0.0002 | +0.0028 | = plain FM. interactions already balanced |
| 10 | Nonlinear MLP head over embeddings (2 lrs) | mlp.py [513a4f9] | 0.5984 ± 0.0002 | +0.0034 | best single model at the time |
| 11 | Cross-class committee beats same-class | ensemble.py [513a4f9] | 0.5986 | +0.0036 | banked. same-class adds ~nothing |
| 12 | MLP capacity (k=32, H=128) | (inline) [bfcb046] | ≤0.5983 | . | capacity flat for MLP too |
| 13a | Pointwise warm-start anneal | refinements.py [bfcb046] | 0.5982 ± 0.0003 | +0.0032 | pointwise adds nothing in any mixture |
| 13b | Hard-negative mining curriculum | refinements.py [bfcb046] | 0.5849 ± 0.0015 | −0.0101 | top-scored negatives are near-positives |
| 14 | Depth (MLP 64→32) and FFM field-aware embeddings | architectures.py [fc697fe] | 0.5982 / 0.5976 | . | same ceiling from both directions |
| 15 | 4-class committees | ensemble2.py [2d8997d] | best-test 0.5988 | +0.0038 | test-peek refusal #1: validation tied. incumbent kept |
| 16 | user×tab cross. InfoNCE temperature | run16.py [4545622] | ≤0.5982 | . | sparse cross overfits. τ=1 optimal |
| 17 | Validation-weighted / diverse-config / lr-decay | run17.py [86faf64] | 0.5988 / 0.5986 / 0.5981 | . | α-blend banked +0.0002 (validation-selected). others |
| 18 | Causal sequence features (strictly-prior behavior) | sequences.py [08cae0d] | 0.6016 ± 0.0004 | +0.0066 | breakthrough. largest single gain |
| 19 | Multi-task aux heads (click/like) | multitask.py [67f6c7e] | 0.6014 ± 0.0005 | +0.0064 | recency already carries the signal |
| 20 | Committee on seq features. richer sequence set | run20.py [fdfae5b] | 0.6043 / 0.6040 | +0.0093 | banked. richer set promising, high variance |
| 21 | Committee on richer sequences | run21.py [825f08d] | 0.6104 | +0.0154 | banked. FM-rich singles revelation (avg 0.6101) |
| 22 | Interest vector (pooled watched-video embeddings) | interest.py [825f08d] | 0.6036 ± 0.0003 | +0.0086 | best single on base+seq. superseded by rich line |
| 23 | Grand committee incl. interest models | run23.py [86faf64] | 0.6045 | +0.0095 | superseded by rich line. diversity lesson kept |
| 24 | FM-rich k=32. FM-only committee. deeper history | run24.py [86faf64] | 0.6099 / 0.6116 / 0.6079 | +0.0166 | 0.6116 banked . final. capacity dead 3rd time. long windows dilute |
| 25 | Auth-cap isolate. cross-view blend (α on valid) | run25.py [a777d1e] | 0.6105 / 0.6116 | . | flat. validation put zero weight on the interest view (α=1.0 ⇒ blend is identical to R24b by construction. a search that chose the incumbent, not an independent result) |
| 26 | Attention pooling over history (+ control) | run26.py [86faf64] | 0.6020 both | . | attention = mean pool exactly (clean null) |
| 27 | Hybrid FM (candidate·history dot term) | run27.py [86faf64] | 0.6097 ± 0.0010 | +0.0147 | FM interactions already encode it |
| 28 | Recipe retune on new premises (4 configs) | run28.py [86faf64] | best singles 0.6113 | . | vs committee. lr 5e-4 lifts singles → fed Run 29 |
| 29 | Committees on improved singles | run29.py [d53cdf1] | 0.6123 / 0.6117 | . | test-peek refusal #2: best-test arm has worst validation. incumbent kept. Convergence → freeze at 0.6116 |

**Phase 2 . Autonomous verification campaign as a non-regression check:
it banked nothing and demonstrates that the converged state survives
unattended re-challenge, not autonomous improvement)** This was a separate campaign, not a
resumption: Campaign 1 converged and froze at Run 29. This new run was
initialized from that frozen state to test it autonomously. Global run
numbers 30-32 are project-wide identifiers for traceability, not a claim of
continuation. Its result . Immediate re-convergence in 3 sub-ε iterations . is itself the finding: a converged state, re-challenged with zero human
input, stays converged. Driver-launched, zero interventions. Agent commits
in THIS repo. GAUC/nDCG components in LOG.jsonl):

| Run | Hypothesis | Commit | test primary | Result |
|---|---|---|---|---|
| 30 | hour/content fields under the FM-rich premise (with same-session control) | aaa5ef3 | best arm 0.6103 | +0.0003 . same margin as Run 5 across two model classes and feature regimes → the fields, not the architecture, are the limit. Test-peek refusal #3 recorded |
| 31 | Duration-normalised play-ratio as auxiliary head (3 arms. gradients finite-difference-verified. control bit-identical to banked step) | 49a2192 | 0.6095 | all arms below control. overturns Run 1's explanation . duration mediation was symptom, not cause |
| 32 | Per-field embedding sizes as interaction rank (2×3 grid) | b2f6007 | best arm 0.6098 (=control) | hypothesis inverted: narrow-field rank is the binding constraint (−0.005 at rank 8). wide rank buys nothing |

 3 consecutive iterations < ε=0.002 → stopped at
, banked best unchanged at 0.6116. Full event stream:
`agent/driver_log.jsonl`.

## Tally

53 runs across six campaigns (29 interactive · 3 verification · 6
clean-room · 4 in the v2-loop iteration · 4 in campaign 5, the
completion run · 1 post-promotion mechanism control, R37 · 6 in campaign 6,
the session-depth and partition-exposure refutations . See the addenda
below) · ~89 configurations · 4 banked structural wins (objective,
sequence features, committee, and the loop's own tab_n) · 6 refusals of a
test-favorable result on
validation grounds (5 test-peek refusals plus the unattended sub-margin
decline of R33b, later superseded by the pre-committed committee check) ·
2 diagnosed leakage traps · 1 legality retirement
(random-exposure log) · 4 error-recovery events, all with zero data loss ·
final: **0.6143 (+0.0197 vs published), converged under the official rule
in campaign 5. The pre-promotion champion 0.6116 was itself
twice-converged and independently reproduced.**

## clean-room autonomous run + spec compliance

Clean-room run with records in `cleanroom/`. On convergence
accounting, iterations 2 and 4 ended without a completed experiment . The
session was waiting on a still-training grid . And still counted toward the
below-ε streak. The official rule counts iterations without improvement, so
the convergence stands. Under a stricter result-bearing reading the run
would have needed one more iteration. Stated openly either way.): the same
agent relaunched with
ZERO prior knowledge . Empty logs, no backlog, bare official baseline. Six
iterations, , zero interventions: reproduced the baseline (0.59497),
banked a feature-engineering win at iteration 3 (**0.59744, +0.0028 over the
published baseline**), refused two better-looking test scores on validation
grounds entirely on its own (iterations 4-5), and converged by the official
rule. Its research path diverged from the main campaign's (feature fields
paid under its BCE regime. Pairwise loss refuted with a mechanism), which
makes it a genuine independent trajectory, not a replay.

Official-limits compliance, all campaigns under the Primary-metric
clause: validation-ε convergence, 50-iteration cap, ceiling . whichever first):

**Strict-rule segmentation of the interactive phase as a retrospective
mapping: the interactive session was supervised and not live-instrumented. this table applies the official rule to the validation record after the
fact).** Applying the
official convergence rule (validation improvement > ε resets a 3-miss
counter) to the validation record partitions Runs 1-29 into bounded runs,
each relaunched with accumulated memory following the same pattern as the
verification campaign:

| Segment | Iterations | Validation ε-wins | Converged at | Checkpoint |
|---|---|---|---|---|
| Exploration run 1 | Runs 1-7 (7) | R4 +0.0021 (listwise) | Run 7 | valid 0.6035 / test 0.5978 |
| Low-yield probes | Runs 8-16 (9, as three 3-miss probes) | none | each after 3 misses | unchanged |
| Culminating run | Runs 17-27 (11) | R18 +0.0026, R20 +0.0025, R21 +0.0072, R24 +0.0027 | Run 27 (25/26/27 sub-ε) | valid 0.61906 / test 0.6116 |
| Post-convergence probes | Runs 28-29 (2) | none (banked nothing. checkpoint unchanged) | project stopping rule | unchanged |

The culminating run's converged checkpoint (R24b) was the scored
submission until (superseded by campaign 5, below). That
run used 11 of 50 iterations in a measured ≤ of wall-clock
(the full R17-R29 log span, , which overstates it) . inside every official limit.

All campaigns, side by side . None resumed after converging:

| Campaign | Iterations (≤50) | Wall-clock (≤) | Terminated by |
|---|---|---|---|
| Interactive culminating run (17-27) | 11 | ≤ | validation-ε convergence → freeze |
| Demo A (Runs 30-32) | 3 | | validation-ε convergence |
| Clean-room (6 runs) | 6 | | validation-ε convergence |
| v2-loop iteration (R33) | 1 | ~ of session work | driver wrapper fault which was disclosed |
| Campaign 5 (R33c + R34-R36) | 4 | ~ measured training | validation-ε convergence → final freeze |
| Campaign 6 (R38-R39, unattended) | 3 | 11 s (driver-timed, ) | validation-ε convergence |

 The official clause ("no validation
improvement above ε for N consecutive iterations") admits two
formalizations: per-step (each of the last N banked gains ≤ ε . What
driver.py implements) and cumulative (best of the last N vs best before
that window ≤ ε). The two diverge only on a steady stream of sub-ε
improvements, which never occurred here: every improvement ever banked
exceeded ε, so every converged campaign's closing window contains zero
banked gain (verifiable in the driver logs: Demo A 0/0/0, clean-room
0/0/0 after its reset, campaigns 5 and 6 0/0/0), and both formalizations
return the identical result on every declared convergence. Disclosed
rather than retroactively reimplemented.

its iteration 2 ended without an adjudicated experiment (the session
launched arms and exited. Iteration 3 adjudicated them) and still counted
toward the below-ε streak. The official rule counts iterations without
improvement, so the convergence stands. Under a stricter result-bearing
reading the run would have needed one more iteration. Stated either way.

Scored checkpoint = validation-best at convergence: R33c (campaign 5. valid 0.62059 / test 0.61429), evaluated once on the test split. Until this was R24b (the R29 validation tie resolved to the incumbent).

## fourth campaign . One iteration of the v2 research loop

After the research-extension work (mechanism controls, belief state,
residual analyzer, priority queue. See `PROCESS-AUDIT.md` section 8), the
driver was relaunched at with the v2 iteration prompt. One iteration
ran, unattended, and executed the full loop (all steps verifiable in
`logs/LOG.jsonl` runs R33-ctrl/R33a/R33b/R33-placebo, `code/tab_surface.py`,
`agent/residual_analysis.py`, `agent/belief_state.json`):

1. Observed the frozen champion's residuals and found its worst
 validation slice: tab=0 (slice-restricted primary 0.30436, 5,579 users).
2. Repaired its own instrument. The session noticed the analyzer's
 expected-value measure was inflated by metric degeneracy (92% of tab=0's
 users have no positive label inside the slice, so the slice score is
 floor-dominated) and rewrote `residual_analysis.py` to rank slices by
 oracle headroom on the overall metric instead . A methodological fix the
 loop made to itself, with a self-test.
3. Hypothesized a mechanism (per-surface causal history: the shipped
 prev1/hist10 are stream-wide and 73% tab=1, so on tab=0 rows they report
 behavior from a surface with a 10x higher positive rate) and wrote a
 targeted experiment, `code/tab_surface.py`.
4. Tested: R33-ctrl (fresh RICH control) valid 0.61715. R33a (recency +
 familiarity) 0.61806. R33b (tab_n familiarity only) 0.61955 . A +0.0024
 gain over control.
5. Falsified before banking: ran the time-shuffle placebo on tab_n
 unprompted. Shuffled valid 0.61612, below the control . The gain
 collapses completely, so the signal is real per-surface history, not
 added capacity.
6. Declined to bank. R33b sits +0.00049 above the banked best
 (0.61906), below the 0.001 promotion margin. The result was logged as
 SIGNIFICANT_BUT_NOT_BEST. The frozen 0.6116 checkpoint is unchanged.

 the agent session completed and
exited cleanly at , but the driver wrapper process died without
writing its iteration-end event, so `agent/driver_log.jsonl` for this
campaign records only driver_start and iteration_start. The run is
therefore reported as 1 completed iteration, terminated by a wrapper
fault rather than by the convergence rule. Zero human interventions
occurred during the iteration. All six loop steps above are reconstructed
from the harness log and committed artifacts, not from the driver log.
This is the project's third error-recovery event, again with zero data
loss.

This campaign is included in the Tally section above. One bookkeeping gap,
disclosed: the session never
called `attach_evidence`/`attach_control` on the belief state, so
`agent/belief_state.json` (as the session left it. Preserved at commit
59fcafe) showed both hypotheses as `proposed` with
the pre-repair EV . The iteration's actual evidence, control, and decline
live in `logs/LOG.jsonl` (R33-ctrl/a/b/placebo). The gap was closed on by `agent/close_the_loop.py` (run after the promotion, after the promotion below):
evidence and the passing control were attached through the module's own
API and `promote` . The code-enforced gate . Was exercised live for the
first time, confirming `residual_tab_0`.

## Campaign 5: the completion run that promoted the loop's discovery

The v2 iteration's pre-committed rule (written in `code/tab_surface.py`
before any arm ran) had one step left: a validated, control-passing win
goes to a 5-seed committee promotion check. The driver fault ate that
step's output, so it was re-run, logged through the harness
(`code/tab_committee_check.py`, record R33c): committee validation
0.62059 . Above the banked 0.61906 + 0.001 margin. The rule says
promote. Campaign 5 (`code/campaign5.py`, guided) executed the
promotion inside a run governed by the official convergence rule:

| Iteration | Run | Valid | vs banked | Outcome |
|---|---|---|---|---|
| 1 | R33c committee banked (rule-tagged WIN) | 0.62059 | +0.00153 over 0.61906 | new champion |
| 2 | R34 + per-surface recency | 0.61704 | −0.00355 | miss 1 (recency confirmed redundant) |
| 3 | R35 + duration familiarity (next residual slice) | 0.61889 | −0.00170 | miss 2 |
| 4 | R36 finer tab_n buckets | 0.61989 | −0.00070 | miss 3 |

Converged by the official rule (3 consecutive iterations below
ε=0.002) at the iteration-1 checkpoint. The designated final submission
is therefore R33c: valid 0.62059 / test 0.61429 (+0.0197 over the
official baseline), replacing R24b (0.61906 / 0.61164).

 hypothesis generated by
the unattended v2 loop from the champion's residuals (tab=0 slice). mechanism claim written before the experiment. Time-shuffle placebo run
unprompted by the agent (gain collapses entirely). 3-seed decline at the
sub-margin stage (correct under the rule). 5-seed committee check cleared
the margin (completed after the driver fault). Banked and
converged under the official rule. Every step is in `logs/LOG.jsonl` and
committed code. The completion run is guided and labeled so . its role was bookkeeping of a promotion the loop had already earned, plus
three genuine convergence-window experiments.

 (R37, review-requested): a
discriminating control . The identical count over surface labels
scrambled within each user . Found roughly half of tab_n's gain survives
(56 percent as measured, one 3-seed run, both halves near the noise floor),
so the mechanism was revised from pure surface familiarity to partitioned
familiarity (about half surface-specific, half counting structure). The
promotion is unaffected (validation-based). See PROCESS-AUDIT section 11.

 46 runs across five campaigns, ~82
configurations (superseded by the campaign 6 addenda below. The Tally
section above carries the final figures). The sixth refusal (R33b's
3-seed decline)
stands as a correct rule-following decision at its stage. It was
superseded, not reversed, when the committee check cleared the bar.

## Campaign 6: the loop's first refutation of its own top hypothesis

Driver relaunched at with the v2 iteration prompt. Iteration 1 ran
unattended and executed the full loop (`logs/LOG.jsonl` runs R38-ctrl /
R38a / R38b, `code/session_depth.py`, `agent/belief_state.json`,
`logs/session_depth.out`, `logs/session_depth_diagnostic.out`).

1. Observed. Residual analyzer re-run against the *new* champion
 (R33c, validation 0.62059). Top open hypothesis: `residual_tab_1` . slice tab=1, 20,119 users, oracle headroom 0.2113. `priority.py
 --recompute` and `belief_state.py --next` agreed. That is what was taken.
2. Interrogated the hypothesis before spending on it. Tab=1 is 73% of
 impressions, so its headroom is largely arithmetic. A grounding probe on
 train rows only found real structure inside the slice: long_view
 rate falls 2.9× monotonically with within-visit position (0.418 → 0.143).
3. Hypothesised a mechanism and raised its own bar. Depth-within-visit
 is not `gap` (one interval) and not `hist_n`/`tab_n` (lifetime counts
 that never reset). Tagged temporal . An upgrade from the analyzer's
 default `none`, which makes `promote` demand a falsification control.
4. Tested two arms against a fresh in-session champion control:
 R38-ctrl 0.61955, R38a (+`sess_pos`) 0.61850, R38b (+`sess_pos`
 +`sess_hit`) 0.61706.
5. Refuted, by its own pre-committed rule. Both arms below control, so
 no win, so no control run and no committee . The rule stops there.
 R38b's −0.00347 clears the 0.002 gate and is claimed as a negative. R38a's −0.00105 is recorded as directional only.
6. Explained the failure with three label-free diagnostics and killed a
 plausible screen in the process: `sess_pos`'s prior is 77% within-user
 and 54% novel after the champion's features (8.19e-05), versus `tab_n`
 . The feature that won . At 86% novel and 1.04e-04. A "novel
 within-user prior variance" pre-filter would have rated the loser as
 highly as the winner. Feature value in this FM is decided in the shared
 interaction geometry (Run 32's min(k_j,k_l) result), not in any marginal
 statistic. The 3-seed run remains the only instrument that measures it.
7. Repaired the loop. Refuting the top slice exposed a liveness bug . the analyzer always proposed `rep[0]`, and `propose` never revives a
 resolved record, so the queue would have been permanently empty from
 iteration 2. `first_unresolved` now walks to the first unresolved
 slice (three new self-test assertions). Next iteration is served
 `hist=31-100`, EVo 0.13222.

Zero manual interventions during the iteration. Nothing banked. Champion
and submission unchanged at validation 0.62059 / test 0.61429. This is the
loop's second recorded refusal to bank under the rule and its first
outright refutation of an analyzer-generated hypothesis . The negative and
its post-mortem are the iteration's product.

 at that stage. The Tally section at the top carries the final 53/~89: 50 runs, ~86 configurations.

## Campaign 6, iterations 2-3: a second refutation, and the queue's ranking rebuilt

Two driver iterations, one experiment. Iteration 2 took the queued hypothesis,
wrote the script, ran the grounding probe and launched the arms. Its session
ended while they were still training. Iteration 3 found the experiment in
flight and adjudicated it rather than launching a duplicate . The arms were
already the queue's top hypothesis, and a second run would have burned the
iteration and broken the one-experiment rule. Artefacts: `logs/LOG.jsonl` (runs
R39-ctrl / R39a / R39b), `code/partition_familiarity.py`,
`code/familiarity_diagnostics.py`, `code/partition_postmortem.py`,
`agent/residual_analysis.py` (v3), `logs/partition_familiarity.out`,
`logs/familiarity_diagnostic.out`, `logs/partition_postmortem.out`,
`logs/residual_analysis_v3.out`, `agent/belief_state.json`.

1. Observed. With `tab=0` confirmed and `tab=1` refuted, the analyzer walked
 to `residual_hist_31-100` (10,947 users, oracle headroom 0.13222).
 `priority.py --recompute` and `belief_state.py --next` agreed.
2. Interrogated it before spending. The slice scores 0.62154 against 0.62059
 overall . *above* average, EV(old) exactly 0.0 . So its headroom is slice
 size. The actionable reading: `hist_n` is a lifetime count that says nothing
 about how a history was distributed.
3. Generalised the one feature that ever won. The champion holds per-partition
 *outcome* counts and no exposure counts . The numerator of a hit rate without
 the denominator. `tab_n` is exactly that denominator for the surface
 partition. Train-only probe: long_view rate falls 4.7-8.6× with `tag_n`
 inside every stratum of `tag_hist`, on 1.1M rows.
4. Tested against a fresh in-session champion control: R39-ctrl 0.61955,
 R39a (+`tag_n`) 0.61822, R39b (+`tag_n` +`auth_n`) 0.61846.
5. Refuted by the pre-committed rule. Both arms below control → no win, so
 no falsification control and no committee. Both deltas sit under the 0.002
 bar and are recorded as directional only, not claimed negatives. The
 control reproduces R33b to five decimals from a fourth independent code path.
6. Killed the same screen a second time, harder. On the identical instrument
 iteration 1 used, `tag_n` carries 4.871e-04 of novel within-user prior
 variance against the winning `tab_n`'s 3.656e-04 . The loser at 1.33× the
 winner. It also varies within more users (73.2% vs 43.7%). Iteration 1's
 loser scored 0.79×. The screen now fails in both directions. Redundancy is
 ruled out backwards: conditioning `tag_n` on `tag_hist` multiplies its rate
 variance by 12 rather than absorbing it.
7. Repaired the loop, again. The refutation exposed that the queue was
 ranking by *size*: EVo grows with any large slice. EVx subtracts a
 matched null (same per-user row counts, rows drawn at random within each
 user) and keeps only the headroom attributable to the model being
 differentially wrong there . The falsification-control construction turned on
 the loop's own hypothesis queue, and signed, so a slice the model handles
 unusually well scores negative. The self-test is now a regression test for
 the exact defect: a small broken slice EVo ranks last of four and EVx ranks
 first. A first null that dealt row counts to *other* users was built,
 measured, and discarded . The skewed rows-per-user distribution made it
 infeasible. Under v3 the refuted slice leaves the top eight and the duration
 slices (model at 0.52-0.54 vs 0.621) take the queue. Next iteration is served
 `residual_dur_4`, mechanism tag `capacity`.

Zero manual interventions. Nothing banked. Champion and submission unchanged at
validation 0.62059 / test 0.61429. Two campaign-6 iterations, two refutations of
analyzer-generated hypotheses, and two instrument repairs found by the
refutations rather than by inspection.

 53 runs across six campaigns, ~89 configurations.
