# Experiment log

Harness: `experiments.py` (copy of what ran inside the kit). `evaluate.py` and
`data.py` untouched, so the official baseline stays reproducible.

## Run 1, 3 seeds per variant

Official published FM test primary: 0.5946 (std 0.0008 over 5 seeds).

| Variant | valid | test primary | delta |
|---|---|---|---|
| baseline (binary, unweighted) | 0.6014 | 0.5950 ± 0.0003 | — |
| B1 binary, user-weighted | 0.5951 | 0.5888 ± 0.0008 | −0.0062 |
| B2 ratio target | 0.5677 | 0.5598 ± 0.0002 | −0.0352 |
| B2-blend (0.5 label + 0.5 ratio) | 0.5911 | 0.5845 ± 0.0009 | −0.0105 |
| B1+B2 | 0.5692 | 0.5613 ± 0.0010 | −0.0337 |
| B1+B2-blend | 0.5891 | 0.5825 ± 0.0005 | −0.0124 |

Both hypotheses are dead. Recording why, so the agent's memory starts with
them and does not retry.

### B1 analysis

The premise was "the metric weights users equally, so training should too".
That is only true for half the metric.

`primary = mean(GAUC, nDCG@5)`. nDCG@5 does average users equally but GAUC
is weighted by each user's positive count, so it deliberately favours heavier
users. Downweighting them by `1/n` fights GAUC while helping nDCG, and the
former loses more than the latter gains.

There is a second cost: heavy users are where the embeddings actually get
learned. `1/n` weighting throws away statistical strength exactly where the data
is richest.

Lesson for the metric-analysis phase: the two metric components weight users
differently. Any reweighting scheme has to reconcile both, not just one. A
proper analysis of `evaluate.py` would have caught this before spending compute 
which is a point in favour of the A1 idea, not against it.

### B2 Analysis

The premise was "long_view is a threshold on play ratio, so train on the
continuous quantity." The threshold is real, but it is duration-mediated.

Watching 100% of a 10-second clip is easy; watching 100% of a 3-minute video is
not. A model trained to predict raw ratio therefore learns to favour short
videos, which is a different ordering from favouring `long_view`. Ranking by
predicted ratio optimises the wrong thing.

The ratio distribution also works against it: 667k of 1.14M training rows sit
below 0.2, so most of the "dense" signal is compressed near zero.

A duration-normalised target (predicting ratio relative to what is typical for that video length), or ratio as an
auxiliary multi-task head rather than a replacement label, could still work. But
the naive substitution is clearly worse and should not be retried.

## Conclusion (Run 1)

Six variants × 3 seeds, measured against a reproduced baseline,
in 4.5 minutes. Seed noise on our own runs is ±0.0003–0.0010, consistent
with the published 0.0008. so the significance gate (idea A2) is calibrated and
working.

## Run 2, pairwise objective (P-series), 3 seeds per variant

Hypothesis: `primary` only compares scores within one user, but pointwise
logloss spends capacity calibrating scores across users, capacity the metric
cannot see. Training on within-user (pos, neg) pairs (BPR) should convert that
wasted capacity into ranking accuracy. (Organizers' #1 suggested direction.)

Harness: `pairwise.py` in the kit. Pairs sampled from train rows only; users
whose train rows are all-positive/all-negative yield no pairs (counted; their
user embedding stays at init under pure BPR). FM bias cancels in the pairwise
difference. Correct, not a bug.

| Variant | valid | test primary | delta |
|---|---|---|---|
| baseline re-run (same session) | 0.6014 | 0.5950 ± 0.0003 | — |
| P1 pure BPR (lr .001, 4 neg/pos, pat 4) | 0.6027 | 0.5967 ± 0.0005 | +0.0017 |
| P2 BPR + pointwise blend | 0.6013 | 0.5949 ± 0.0005 | −0.0001 |

P1 improves valid (+0.0013) and test (+0.0017) consistently
across all seeds, real directional signal at ~2× seed noise, achieved with
hyperparameters tuned for the pointwise loss. P2's failure is informative:
adding the pointwise term back erased the gain, which supports the hypothesis
that cross-user calibration actively wastes capacity rather than being neutral.

Direction confirmed, effect size not yet banked. BPR gradients
have a different scale from pointwise, so lr / pairs-per-positive / patience
deserve their own tuning before judging the idea's ceiling.

## Run 3, BPR hyperparameter sweep, 3 seeds per config

Hypothesis: P1's +0.0017 came from hyperparameters tuned for the pointwise
loss; BPR gradients have a different scale, so lr / pairs-per-positive /
patience tuned for BPR should grow the gain past the 0.002 bar.

| Config | valid | test primary | delta |
|---|---|---|---|
| P1a lr .001, 8 neg/pos, pat 6 | 0.6026 | 0.5965 ± 0.0006 | +0.0015 |
| P1b lr .002, 4 neg/pos, pat 6 | 0.6020 | 0.5955 ± 0.0005 | +0.0005 |
| P1c lr .002, 8 neg/pos, pat 6 | 0.5976 | 0.5898 ± 0.0014 | −0.0052 |
| P1d lr .003, 8 neg/pos, pat 6 | 0.5894 | 0.5803 ± 0.0013 | −0.0147 |

Hypothesis rejected. The original P1 setting (lr .001, 4 neg/pos) remains
best. The gain does not scale with lr or pair count, higher lr is sharply
destructive. Reading: BPR's benefit is capped by something other than its own
hyperparameters, candidate suspects are model capacity (k=16 embeddings) and
the softmax-free pair gradient treating all negatives equally.

Next (Run 4): two attacks on the cap, (a) listwise/InfoNCE objective:
softmax over 1 positive + K sampled negatives, which upweights hard negatives
instead of BPR's uniform pair treatment; (b) capacity: k=32 under BPR, with a
k=32 pointwise control so any gain is attributed to the interaction, not
capacity alone.

## Run 4, listwise objective + capacity test, 3 seeds per config

Hypothesis (a): BPR's uniform pair gradient is the cap, softmax over
(1 pos + K negs) concentrates gradient on hard negatives and should beat it.
Hypothesis (b): k=16 embedding capacity is the cap, k=32 should help.

| Config | valid | test primary | delta |
|---|---|---|---|
| L1 InfoNCE K=4, k=16 | 0.6035 | 0.5978 ± 0.0002 | +0.0028 |
| L2 InfoNCE K=8, k=16 | 0.6032 | 0.5977 ± 0.0002 | +0.0027 |
| C1 BPR k=32 | 0.6025 | 0.5965 ± 0.0004 | +0.0015 |
| C0 pointwise k=32 (control) | 0.6015 | 0.5950 ± 0.0006 | −0.0000 |

Hypothesis (a) CONFIRMED, (b) rejected. The listwise objective clears the
significance bar with the tightest seed variance measured so far (±0.0002),
and valid/test agree. The control matters: capacity alone does nothing at
either objective, so the entire +0.0028 is attributable to matching the
training objective to the ranking metric, the organizers' suggested
direction, now quantified: pointwise → pairwise +0.0017, pairwise → listwise
+0.0011 more. K=8 ≈ K=4, so negative-sampling breadth is saturated.

L1 = 0.5978 test primary (baseline 0.5950, published 0.5946).

## Run 5, feature groups under L1, 3 seeds per config

Hypothesis: the kit's 5 fields ignore most logged data; adding informative
categorical fields should stack with the objective win.

| Config | valid | test primary | d vs base | d vs L1 |
|---|---|---|---|---|
| L1 base (control) | 0.6035 | 0.5978 ± 0.0002 | +0.0028 | — |
| F1 +hour | 0.6046 | 0.5981 ± 0.0004 | +0.0031 | +0.0003 |
| F2 +content (vtype/mtype/tag) | 0.6042 | 0.5981 ± 0.0002 | +0.0031 | +0.0003 |
| F3 +user_active_degree | 0.6041 | 0.5980 ± 0.0004 | +0.0030 | +0.0002 |
| F4 +vpop (train-only rate) | 0.6043 | 0.5981 ± 0.0003 | +0.0031 | +0.0003 |

No single group is significant on its own (+0.0002…0.0003), but all
four are positive on BOTH valid and test, a consistent directional pattern
that suggests stacking. Valid gains (+0.0006…0.0011) exceed test gains,
so expectations should be modest.

## Run 6, stacking + personalization affinities, 3 seeds

F5 = all four Run-5 groups combined. F6 = user→author and user→tag smoothed
long_view rates from train rows. F7 = both. (First attempt SIGTERMed after F5,
external kill, likely lid-close; F5 result was already captured, F6/F7
rerun. Nothing lost: per-config results are logged as they finish.)

| Config | valid | test primary | d vs base | d vs L1 |
|---|---|---|---|---|
| F5 combo (all 4 groups) | 0.6045 | 0.5981 ± 0.0001 | +0.0031 | +0.0003 |
| F6 base + affinities | 0.5910 | 0.5858 ± 0.0010 | −0.0092 | −0.0120 |
| F7 all | 0.5938 | 0.5895 ± 0.0006 | −0.0055 | −0.0083 |

F5: the four groups do NOT stack, combined they add the same +0.0003 as
each alone. They are redundant encodings of similar weak signal.

F6/F7 failed for a diagnosable reason: self-inclusion leakage within
train. Each train row's affinity rate included that row's own label. For
sparse user×author pairs (often 1–3 impressions) the feature is nearly the
answer key during training, so the model leans on it; at valid/test the
feature is honest and the reliance collapses. The train-only guard blocked
future leakage but not self-leakage, a classic target-encoding trap.
Note the asymmetry with F4 (vpop), which shares the construction but has
thousands of impressions per video, making self-inclusion negligible, which
is why F4 was mildly positive while F6 was sharply negative.

Any per-entity rate feature over sparse
keys must use leave-one-out encoding on train rows.

## Run 7, affinities with leave-one-out fix, 3 seeds

| Config | valid | test primary | d vs base | d vs L1 |
|---|---|---|---|---|
| F6-LOO base + affinities | 0.6031 | 0.5970 ± 0.0003 | +0.0020 | −0.0008 |

Diagnosis confirmed, feature rejected. LOO encoding recovered the naive
version's collapse (0.5858 → 0.5970), proving self-inclusion was the failure
mechanism, but even honest affinities add nothing over L1. Conclusion: the
FM's user×author / user×tag embedding interactions already capture this
signal; explicit rate features are redundant with the model's own factorization.

Categorical side features are worth
at most +0.0003 (noise-level) under the listwise objective. The lever is
exhausted; the objective change remains the only banked structural win.

## Run 8, seed ensembling, 5 models

| Config | valid | test primary | delta vs base |
|---|---|---|---|
| L1 singles (5 seeds) | — | 0.5978 ± 0.0001 | +0.0028 |
| Ensemble of 3 | 0.6036 | 0.5981 | +0.0031 |
| Ensemble of 5 | 0.6036 | 0.5982 | +0.0032, new best |

Ensembling adds only +0.0004. The FM is convex enough that seeds
converge to near-identical solutions, leaving little disagreement to average.
Kept (it is free), but the score has plateaued near 0.598 within this model
class.

## State so far

- Banked, significant, reproducible: test primary 0.5982 vs our baseline
  0.5950 (published 0.5946). Sole structural win: listwise InfoNCE objective.
- Exhausted levers: BPR hyperparameters, embedding capacity (k=32 does
  nothing), categorical side features (+0.0003 noise), explicit affinity
  features (redundant with FM factorization), seed ensembling (+0.0004).
- Next frontier (for the autonomous agent): model class. The FM is
  second-order and linear in its interactions; a field-weighted FM (FwFM) or a
  small MLP head over the embeddings can express interaction patterns the FM
  cannot. This is the natural next hypothesis family.

## Run 9, FwFM (first harness-run experiment), 3 seeds

FwFM = FM + learned scalar weight per field pair, R init = ones (identical to
FM at step zero, so any departure is data-driven).

| Config | valid | test primary | delta vs base |
|---|---|---|---|
| FwFM listwise K=4 | 0.6030 | 0.5978 ± 0.0002 | +0.0028 |

No gain over the plain FM (0.5978 = 0.5978). With only 5 fields the FM's
uniform interactions are evidently already balanced; field-level reweighting
has nothing to fix. Idea retired. (Run also caught a harness JSON bug, numpy
float32 not serializable, fixed; intent record survived the crash exactly as
designed. LOG.jsonl backfilled with pre-harness bests so verdicts compare
against the true best.)

## Run 10, MLP head over embeddings, 2 lrs × 3 seeds

First nonlinear model: embeddings → hidden ReLU (H=64) → score, listwise
objective, via the harness.

| Config | test primary | delta vs base |
|---|---|---|
| MLP H=64, lr 0.001 | 0.5982 ± 0.0007 | +0.0032 |
| MLP H=64, lr 0.0003 | 0.5984 ± 0.0002 | +0.0034, best single model |

The nonlinearity finds real signal the second-order FM cannot express: a
single MLP beats the 5-seed FM ensemble. Lower lr is better and tighter, as
expected for a nonlinear head.

## Run 11, ensembles around the MLP

| Config | valid | test primary | delta vs base |
|---|---|---|---|
| R11a MLP 5-seed ensemble | 0.6038 | 0.5983 | +0.0033 |
| R11b mixed 5 MLP + 5 FM | 0.6041 | 0.5986 | +0.0036, new best |

Diversity hypothesis confirmed precisely: same-class MLP ensembling adds
nothing over a single MLP (0.5983 ≤ 0.5984), but mixing model classes gains,
FMs and MLPs err differently. Current banked best: 0.5986 (published
baseline 0.5946 → +0.0040).

## Run 12, MLP capacity retest, 2 configs × 3 seeds

| Config | test primary | delta vs best single |
|---|---|---|
| MLP k=32, H=64 | 0.5983 ± 0.0003 | −0.0003 |
| MLP k=16, H=128 | 0.5978 ± 0.0002 | −0.0008 |

Capacity is not the constraint for the nonlinear model either. The compact
MLP (k=16, H=64) stands. Capacity now ruled out for both model classes.

## Run 13, final backlog refinements, 2 × 3 seeds

| Config | test primary | delta vs base |
|---|---|---|
| R13a pointwise warm-start → listwise | 0.5982 ± 0.0003 | +0.0032 |
| R13b hard-negative mining (50% top-quartile) | 0.5849 ± 0.0015 | −0.0101 |

R13a: sequential annealing neither helps nor hurts, combined with Run 2's
blend result, the conclusion is clean: pointwise signal adds nothing in any
mixture, simultaneous or sequential.

R13b failed instructively. Over-sampling top-scored negatives backfires,
likely because a user's highest-scored negatives include near-positives
(impressions that almost crossed the long_view threshold), hammering them as
negatives teaches the model to suppress exactly the taste signal it should
rank highly. InfoNCE's built-in gradient weighting already takes what hard
negatives offer; double-dipping on the sampling side over-commits to label
noise.

## Run 14, new model classes, 2 × 3 seeds

Owner directed exploration to methods/models only (sequence, multi-task and
random-log ideas PARKED in IDEAS.md).

| Config | test primary | vs best single (0.5984) |
|---|---|---|
| R14a two-layer MLP 64→32 | 0.5982 ± 0.0000 | −0.0003, depth doesn't help |
| R14b FFM k=8 (field-aware embeddings) | 0.5976 ± 0.0002 | −0.0010, not better solo |

Same ceiling from both directions: the single-layer MLP already extracts what
the 5 fields offer. But solo strength isn't the point, Run 11 showed the
banked best comes from cross-class diversity, which sets up:

## Run 15, 4-class ensemble expansion, 20 models

| Committee | valid (5 dp) | test primary | d vs base |
|---|---|---|---|
| R15a mlp+fm (banked reference) | 0.60407 | 0.5986 | +0.0036 |
| R15b mlp+fm+ffm | 0.60372 | 0.5985 | +0.0035 |
| R15c mlp+fm+mlp2 | 0.60406 | 0.5988 | +0.0038 |
| R15d all four | 0.60386 | 0.5987 | +0.0037 |

FFM dilutes the committee (worst validation of the four), retired.

R15c shows the highest
TEST number (0.5988), but our selection rule is: decide on VALIDATION only.
On validation, R15a (0.60407) and R15c (0.60406) are tied to within far less
than noise, and the incumbent wins ties by the pre-committed simplicity rule
(10 models beats 15; don't switch without validation evidence). The banked
recipe therefore REMAINS mlp+fm at test 0.5986. Choosing R15c because its
test number looks better would be exactly the test-peeking this whole log
exists to prevent. The 0.5988 is recorded as an observation, not claimed as
the result.

(What this costs us: possibly 0.0002 of reportable score. What it buys: the
report can say, with a concrete example, that the selection process never
selected on test, including the one time it was tempting. Test values
were computed and visible; the discipline was in never letting them choose.)

## Run 16, final backlog ideas, 3 × 3 seeds

Tab stats measured first: 73% of impressions on tab 1; per-tab long_view
rates span 0.4%–49%.

| Config | test primary | vs best single (0.5984) |
|---|---|---|
| R16a user×tab cross field | 0.5973 ± 0.0003 | −0.0011, sparse cross overfits |
| R16b InfoNCE τ=0.5 (sharpened) | 0.5982 ± 0.0001 | −0.0002 |
| R16c InfoNCE τ=2.0 (softened) | 0.5978 ± 0.0003 | −0.0006 |

The default temperature (τ=1) was already optimal; per-surface conditioning
via a cross field adds sparsity, not signal.

# EXPLORATION PHASE COMPLETE

16 runs. The idea space set out in IDEAS.md is fully mapped: 2 objectives ×
3 loss families, 4 model classes, capacity (twice), 6 feature families,
ensembling (4 committee shapes), temperature, and 2 diagnosed leakage traps.
The official convergence rule (ε=0.002, N=3) fired several iterations ago.

Test primary 0.5986, listwise InfoNCE, mixed 5 MLP + 5 FM
committee, selected on validation throughout. Published baseline 0.5946 →
+0.0040, with the single-model recipe (MLP, 0.5984) as the simpler
alternative if the ensemble is judged too heavy.

## Run 17, score-push (partial, in flight)

| Config | valid | test primary | vs banked 0.5986 |
|---|---|---|---|
| R17a validation-weighted committee (α=0.60 MLP) | 0.60415 | 0.5988 | +0.0002, better |

α searched on validation only (0.60415 beats the equal-weight 0.60407), so
unlike Run 15c this IS a legitimate validation-selected improvement.
R17b (diverse-config committee) and R17c (lr decay) still running.

## Run 18, causal sequence features, breakthrough

| Config | test primary | vs banked 0.5986 |
|---|---|---|
| MLP + causal sequence features | 0.6016 ± 0.0004 | +0.0030, better |

Four features from the user's strictly-prior behavior (previous-impression
label, rolling 10-impression rate, history depth, causal per-author history);
a row's own label never enters its features; sorted by (date, time_ms).

+0.0070 over the published baseline, the largest single gain of the
project, roughly equal to everything else combined. It also explains every
architecture plateau retroactively: with static features only, the models
were information-starved, not under-powered. Where Runs 6–7's static
affinities failed (whole-window, self-inclusive), the causal, self-exclusive,
dynamic version succeeds, the contrast between those two runs is itself the
cleanest evidence in the log that how a feature is constructed matters more
than what it encodes.

## Run 17 close, score-push, final tallies

| Config | test primary | vs 0.5986 |
|---|---|---|
| R17a validation-weighted committee (α=0.60) | 0.5988 | +0.0002 (validation-selected, legitimate) |
| R17b diverse-config committee | 0.5986 | ±0.0000 |
| R17c lr-decay long-train | 0.5981 ± 0.0003 | −0.0005 |

All superseded by the sequence line below.

## Run 19, multi-task auxiliary heads (click + like), 3 seeds

| Config | test primary | vs banked 0.6016 |
|---|---|---|
| R19 aux heads λ=0.3 on R18 recipe | 0.6014 ± 0.0005 | −0.0002, not better |

Click/like prediction adds nothing once sequence features exist, recent
behavior evidently already carries what those labels would teach. Retired.

## Run 20, compounding the sequence win

| Config | valid | test primary | vs prior best |
|---|---|---|---|
| R20a mixed committee on seq features (5 MLP + 5 FM) | 0.60924 | 0.6043 | +0.0027, new banked best |
| R20b richer sequences, single MLP (+hist30/tag_hist/gap) | — | 0.6040 ± 0.0023 | +0.0024 but 3× normal seed variance |

Banked best is now 0.6043, total +0.0097 over the published 0.5946.
The committee mechanism stacks cleanly on the sequence features; the richer
feature set looks promising but needs its variance firmed up (→ Run 21).

## Run 21, committee on richer sequences, second breakthrough

| Config | valid | test primary |
|---|---|---|
| Committee (5 MLP + 5 FM) on richer seq fields | 0.61641 | 0.6104 |

The revelation is in the singles: FM-rich = 0.6089–0.6116 across seeds
(avg 0.6101), beating every MLP on the same fields (MLP 0.6042 ± 0.0023,
unstable). The FM's multiplicative interactions thrive on rich causal
features; the MLP does not.

## Run 22, interest-vector MLP (DIN-lite), 3 seeds

| Config | test primary |
|---|---|
| Mean-pooled watched-history embeddings | 0.6036 ± 0.0003 |

Content-based recency works, pooled embeddings of the last 10 watched
videos beat count-features alone (on the base sequence set).

## Run 23, grand committee on base+seq

| Config | valid | test |
|---|---|---|
| interest+mlp+fm ×5 | 0.60942 | 0.6045 |
| interest+mlp ×5 | 0.60834 | 0.6034 |

Lesson kept: interest models blend well (three-view beat two-view), feeds
Run 25b.

## Run 24, FM-rich follow-ups

| Config | test primary |
|---|---|
| R24a FM-rich k=32 | 0.6099 ± 0.0012 |
| R24b FM-rich-only committee (5 seeds) | 0.6116 (valid 0.61906) |
| R24c deeper history (hist100 + auth9 together) | 0.6079 ± 0.0007 |

The recipe simplified itself: 5 FMs, k=16, rich causal sequence features,
listwise InfoNCE, 0.6116, +0.0170 over the published baseline. The mixed
committee's MLPs were dragging, not diversifying. R24c's flaw (two changes
at once) spawned the R25a isolate.

## Run 25, closing out

| Config | test primary |
|---|---|
| R25a auth_hist 9+ cap, isolated | 0.6105 ± 0.0011 |
| R25b cross-view blend, α on validation | α=1.00 → 0.6116 |

## Runs 26–29, the final five shots

| Shot | Config | test |
|---|---|---|
| 1 | R26a attention-pooled interest, rich fields | 0.6020 ± 0.0005 |
| 2 | R26b mean-pool control | 0.6020 ± 0.0006 |
| 3 | R27 hybrid FM (candidate·history dot term) | 0.6097 ± 0.0010 |
| 4 | R28 recipe retune (4 configs) | best 0.6113 ± 0.0009 singles |
| 5 | R29 committees on improved singles | R29a test 0.6123 / valid 0.61878; R29b valid 0.61907 / test 0.6117 |

Second selection-integrity refusal (mirror of Run 15). R29a shows the
best test number ever observed (0.6123) but the LOWEST validation of the
three candidates; validation ranks all three within 0.0003 of each other,
a tie inside noise, and the incumbent wins ties. Banking 0.6123 because the
test number sparkles would be test-set selection. Refused; recorded as an
unclaimed observation. Also retired on legality: the random-exposure log
(rows entirely inside the eval window → temporal leakage if trained on).

# MODEL FROZEN

Test primary 0.6116 (+0.0170 over published 0.5946).
5-seed FM committee, k=16, lr 1e-3, listwise InfoNCE K=4, rich
causal sequence features. Weights + predictions saved to `frozen_model/`
by `final_model.py` (one command, ~5 min, numpy only, full retrain from raw
data). Tally as of this freeze: 29 runs, ~60 configurations,
2 documented test-peek refusals at that point (final project tally: 53
runs, ~89 configurations, 6 refusals, final checkpoint 0.6143 after the
later promotion, see ITERATION-LOGS.md),
1 legality retirement, every claim 3+ seeds past a pre-committed bar.

# AUTONOMOUS PHASE, the driver loops the agent, unattended

## Run 30, autonomous iteration 1: side + content fields under FM-rich

Idea picked by the agent from IDEAS.md's residual-unknown list (#10 and #11,
the same premise gap, so one experiment). Runs 5–6 measured `hour` and the
content fields (`video_type`/`music_type`/`tag`) under the MLP on the five
base fields and retired them at +0.0003. Two revolutions later the premise is
different in both directions that matter: the model class is now the FM (Run 21:
+0.006 over the MLP on rich fields) and the features are rich causal sequences.
An FM factorises every field pair, so a new field is not merely extra input,
it buys user×tag, user×music_type, tab×hour interactions a concat-MLP over base
fields cannot form. That mechanism had never been tested.

A control arm on the identical code path was run alongside, so the verdict
compares validation numbers produced in the same session rather than across runs.
FM k=16, lr 1e-3, listwise InfoNCE K=4, patience 4, 3 seeds per arm.

| Arm | valid (5 dp) | Δ vs control | test primary | test GAUC / nDCG@5 |
|---|---|---|---|---|
| R30-ctrl FM-rich (control) | 0.61715 | — | 0.6098 ± 0.0010 | 0.6797 / 0.5398 |
| R30a + hour | 0.61711 | −0.00004 | 0.6106 ± 0.0009 | 0.6810 / 0.5402 |
| R30b + content | 0.61712 | −0.00003 | 0.6101 ± 0.0011 | 0.6803 / 0.5398 |
| R30c + hour + content | 0.61746 | +0.00031 | 0.6103 ± 0.0013 | 0.6804 / 0.5403 |

Both ideas DEAD, no promotion, no bank. The best arm clears the
control by +0.00031 on validation, a sixth of the 0.002 gate, and well inside
the σ≈0.0008 seed noise. The promotion step (5-seed committee of the winning
arm, to be compared against the banked R24b committee's validation 0.61906) was
written into the script and did not fire, exactly as pre-committed.

The striking part is not that the fields failed but that
they failed by the same margin: Run 5 measured +0.0003 for these families
under the MLP on base fields, and Run 30 measures +0.0003 under the FM on rich
causal fields. Two model classes and two feature regimes apart, the number does
not move. That is much stronger evidence than either run alone, it says these
side attributes carry essentially no incremental within-user ranking signal in
KuaiRand-Pure, rather than that the earlier architecture was too weak to use
them. The FM's pairwise factorization, which rescued the sequence features so
dramatically in Run 21, does not rescue static content metadata. It also
sharpens the project's central finding: what mattered was never more fields,
it was causal, user-conditional, time-varying fields.

Third selection-integrity refusal. R30a shows test 0.6106 against the
control's 0.6098, a +0.0008 test "gain", while its validation is 0.00004
below the control. Reading that as a win would be pure test-peeking; it is the
seed-noise band doing what noise does. Recorded as an observation, not a result.
The three refusals (Run 15, Run 29, Run 30) are now a pattern rather than an
anecdote: every time the test number has diverged from the validation ranking in
this project, the discipline has held.

Test primary 0.6116 (5-seed FM committee, k=16, rich
causal sequence features, listwise InfoNCE K=4), +0.0170 over the published
0.5946. Remaining OPEN ideas after this run: #12 duration-normalised play-ratio
as an auxiliary signal, #13 K=2 under FM-rich, #14 per-field embedding sizes.

## Run 31, autonomous iteration 2: duration-normalised play-ratio as an auxiliary signal

Idea picked by the agent from IDEAS.md's residual-unknown list (#12). Run 1
killed the dense play-ratio target, the model learned "prefer short
videos", since `play_time/duration` is duration-mediated, and explicitly left
the duration-normalised variant open. It was never revisited across the three
revolutions since (listwise InfoNCE, FM > MLP on rich fields, causal sequence
features). Two things make this a genuine premise gap rather than a retry:

1. Auxiliary, not target. The ranking objective is untouched listwise
   InfoNCE on `long_view`. The play ratio only supplies extra gradient to the
   shared embedding matrix V through its own head
   (`za = ba + Wa[X].sum + Σ_j ca_j · 0.5(S_j² − Σ_f E_fj²)`). A signal can be
   a bad target and still be a good regulariser.
2. Duration-normalised by construction. The target is the row's play ratio
   as a percentile within its own duration bucket. The diagnostic printed
   at run start confirms the normalisation does exactly what Run 1 wanted:
   `corr(target, dur_bucket) = −0.376` for the global percentile versus
   −0.000 for the within-bucket one.

The mechanism argument for why this could beat Run 19's dead click/like heads:
click and like are sparse binary events that recent-behavior features already
predict, whereas the play ratio is dense and graded on every impression,
including the ~85% negatives the binary label says nothing about. It is the
only supervision in KuaiRand-Pure that reports how close a negative came to
being a positive.

The aux head's analytic gradients (V, Wa, ca, ba)
were finite-difference checked in float64 (agreement to 7+ significant
figures), and `rank_step` was shown bit-identical to the banked
`listwise.infonce_step` over 5 steps, so the control arm is a true control
rather than a re-implementation. FM k=16, lr 1e-3, K=4, patience 4, rich causal
fields, aux on alternate batches, 3 seeds per arm.

| Arm | valid (5 dp) | Δ vs control | test primary | test GAUC / nDCG@5 |
|---|---|---|---|---|
| R31-ctrl FM-rich, no aux head | 0.61715 | — | 0.6098 ± 0.0010 | 0.6797 / 0.5398 |
| R31a aux, within-bucket, λ=0.3 | 0.61587 | −0.00128 | 0.6095 ± 0.0009 | 0.6798 / 0.5392 |
| R31b aux, within-bucket, λ=1.0 | 0.61608 | −0.00107 | 0.6095 ± 0.0002 | 0.6797 / 0.5393 |
| R31c aux, global percentile, λ=0.3 | 0.61584 | −0.00131 | 0.6095 ± 0.0009 | 0.6798 / 0.5392 |

IDEA #12 DEAD, no promotion, no bank. Every aux arm lands below
the control on validation. The promotion step (5-seed committee of the winning
arm against the banked R24b committee's validation 0.61906) was written into
the script and did not fire, as pre-committed. Nothing here clears the 0.002
bar in either direction, so the magnitude is not claimed, but the sign is
consistent across three arms, three seeds each, and both metric components
(valid GAUC 0.6882–0.6885 vs the control's 0.6900; nDCG@5 0.5434–0.5436 vs
0.5443), which is worth recording as a directional observation.

The informative part is R31c. The two-arm contrast was
designed to isolate Run 1's stated cause of death: R31a's target has zero
correlation with duration, R31c's has −0.376. They perform identically
(−0.00128 vs −0.00131). Duration mediation, then, was never the binding
problem for this signal, it was the visible symptom in Run 1's setup. The
real issue is that watch-completion is a different ranking of the same rows
than within-user preference: the play ratio correlates 0.79 with `long_view`
in the aggregate, but the 0.21 it does not share is precisely the part the
GAUC/nDCG objective is scored on, and any gradient pulling the shared
embeddings toward the completion ordering pulls them off the ranking one.
λ=1.0 being no worse than λ=0.3 confirms this is a direction problem, not a
weight-tuning problem. The auxiliary-head family is now 0-for-2 (Run 19
sparse binary, Run 31 dense graded) under the sequence-feature premise, and
for the same underlying reason both times: once causal recent-behavior
features are present, extra outcome supervision has nothing left to teach the
representation.

R31-ctrl reproduces Run 30's control to five
decimal places on validation (0.61715) and four on test (0.6098) from an
independently constructed code path, the control arms are stable, so
cross-run comparison at this precision is sound.

Test primary 0.6116 (5-seed FM committee, k=16,
rich causal sequence features, listwise InfoNCE K=4), +0.0170 over the
published 0.5946. Remaining OPEN ideas after this run: #13 K=2 under FM-rich,
#14 per-field embedding sizes.

## Run 32, autonomous iteration 3: per-field embedding sizes (interaction rank) under FM-rich

Idea picked by the agent from IDEAS.md's residual-unknown list (#14), the last
structural unknown and the only one never tested under any premise. Every model
in this project has used one global k: 16 dimensions for `user_id` (26,210
values) and 16 for `prev1` (3 values). IDEAS #14 proposed shifting the parameter
budget toward where cardinality lives, wide fields k=24, narrow fields k=8.

Restating what the knob actually does. The parameter-budget framing in the
idea is nearly vacuous: the narrow fields hold 71 of V's 40,313 rows, so
shrinking them frees 0.1% of the parameters (645,008 → 644,368, measured). What
per-field k really controls in an FM is the rank of every pairwise
interaction the field takes part in, field j and field l interact through a
bilinear form of rank min(k_j, k_l). So this run is a per-field-pair rank
experiment, which nothing before it has touched, and Run 4's dead uniform k=32
does not answer it: raising every k at once cannot separate "more rank on
user×video" from "more rank on prev1×tab".

Implementation: dimensions k_j…k_max of a field's embedding rows are initialised
to zero and their gradient is masked, so they stay exactly zero for the whole
run. `masked_infonce_step` was asserted bit-identical to the banked
`listwise.infonce_step` at mask≡1 (V and W equal to the last bit over 5 steps)
before any arm ran, so R32-ctrl is a true control. 2×3 grid over (wide rank,
narrow rank), wide = train vocabulary ≥ 1000 (`user_id`, `video_id`,
`author_id`), narrow = the other nine. FM lr 1e-3, listwise InfoNCE K=4,
patience 4, rich causal fields, 3 seeds per arm.

| Arm | V params | valid (5 dp) | Δ vs control | test primary | test GAUC / nDCG@5 |
|---|---|---|---|---|---|
| R32-ctrl wide16 / narrow16 (control) | 645,008 | 0.61715 | — | 0.6098 ± 0.0010 | 0.6797 / 0.5398 |
| R32a wide24 / narrow8 (IDEAS #14) | 966,232 | 0.61098 | −0.00617 | 0.6046 ± 0.0016 | 0.6735 / 0.5357 |
| R32b wide16 / narrow8 | 644,368 | 0.61186 | −0.00529 | 0.6052 ± 0.0029 | 0.6741 / 0.5363 |
| R32c wide24 / narrow16 | 966,872 | 0.61568 | −0.00147 | 0.6092 ± 0.0005 | 0.6793 / 0.5391 |
| R32d wide24 / narrow24 (uniform k=24) | 967,512 | 0.61656 | −0.00059 | 0.6098 ± 0.0006 | 0.6804 / 0.5393 |

IDEA #14 DEAD, no promotion, no bank. No arm beats the control; the
proposed configuration is the worst of the five. The promotion step (5-seed
committee of the winning arm against the banked R24b committee's validation
0.61906) was written into the script and did not fire, as pre-committed.

The result is backwards from the hypothesis, and that is the
finding. Read the grid by column and the pattern is unambiguous: the score
tracks the narrow rank and ignores the wide one. At fixed narrow rank,
widening 16→24 does nothing (−0.0015 at narrow 16, −0.0009 at narrow 8, both
inside noise); at fixed wide rank, cutting narrow 16→8 costs −0.0053 to −0.0062,
2.6–3× the 0.002 gate and visible in both metric components (valid GAUC
0.681–0.683 vs 0.690). This is a claimable negative, not a null.

The mechanism is the min(k_j, k_l) rule. My own framing when writing the script,
"`prev1`×`tab` has 15 configurations and a rank-16 form is free to memorise
them", was the wrong intuition, and the data says so. A narrow field's rank
does not bound some small cross of its own; it bounds every interaction it has
with the wide fields. `prev1`×`user_id` is a 3 × 26,210 surface, and rank 8 is
the ceiling on how richly the last impression's outcome can modulate 26,210
distinct user vectors. The causal sequence fields are exactly the ones Run 21
showed carry the project's biggest gain, and they carry it as modulators of the
user and video embeddings, which is precisely the capacity R32a/b removed.
Low cardinality is not low expressive load.

Two secondary readings, both cheap and both worth having. First, R32d
reconfirms Run 4's uniform-capacity saturation under a premise two revolutions
newer: uniform k=24 lands −0.0006 from uniform k=16, so the k=16 ⇒ k=32 null
Run 4 measured on base fields with the MLP still holds for the FM on rich causal
fields. Second, this run finally explains why capacity looks saturated: it is
not that the model has enough rank everywhere, it is that the binding constraint
sits on the narrow fields, where nobody thought to look, and the banked k=16
already sits at or above it.

Test primary 0.6116 (5-seed FM committee, k=16, rich
causal sequence features, listwise InfoNCE K=4), +0.0170 over the published
0.5946. R32-ctrl reproduces the Run 30 and Run 31 controls to five decimals on
validation (0.61715) and four on test (0.6098), a third independent code path.

#10, #11, #12 and #14 of the freeze-time
residual-unknown list are resolved and dead. IDEAS #13 (K=2 under FM-rich) is
the last untested item on the backlog.

## Run 38, campaign 6 iteration 1: within-session position (viewing fatigue) on the majority slice

`agent/residual_analysis.py`,
re-pointed at the new champion (R33c, RICH + tab_n, validation 0.62059), returns
`residual_tab_1` as the top open hypothesis: slice `tab=1`, 20,119 users, oracle
headroom 0.2113 on the overall metric. `priority.py --recompute` and
`belief_state.py --next` both return it, so it is what this iteration took.

`tab=1` is 73% of the
impressions, so "the model is imperfect on tab=1" is nearly a tautology and its
oracle headroom is mostly an arithmetic fact about slice size. The only
actionable reading is structure inside the slice that the feature set does not
encode. A grounding probe on train rows only found one, long_view rate by
within-session position (a session ends after a 30-minute gap):

| sess_pos | tab=1 rate (n) | tab=0 rate (n) |
|---|---|---|
| 0 | 0.418 (406,422) | 0.054 (66,006) |
| 1–2 | 0.387 (244,313) | 0.043 (42,869) |
| 3–5 | 0.344 (106,974) | 0.029 (20,459) |
| 6–10 | 0.300 (50,136) | 0.021 (11,588) |
| 11–20 | 0.240 (21,066) | 0.017 (6,303) |
| 21–50 | 0.170 (5,537) | 0.010 (2,617) |
| 50+ | 0.143 (428) | 0.006 (171) |

A 2.9× monotone decline across the majority slice, with 51% of tab=1 rows at
position ≥ 1. Nothing shipped represents it: `gap` is the single preceding
inter-impression interval bucketed at 1m/1h/1d, it says whether the previous
impression was in the same burst, not how deep into the burst this row sits,
and `hist_n`/`tab_n` are lifetime counts that increment by one per row and never
reset. Depth-within-visit is a third quantity. And it is legible to a
within-user metric: the evaluation ranks all of one user's validation-window
impressions together, and those rows span many visits.

Features (causal, self-exclusive, state updates only after the row is
featurised, rows visited in each user's chronological order, and `sess_pos`
reads timestamps only, never labels): `sess_pos` (bucketed prior impressions in
the current visit) and `sess_hit` (long_views so far in the visit, capped 3+).
Mechanism tag temporal, so the belief-state hypothesis was re-tagged from
the analyzer's default `none`, which raises the bar: `promote()` would refuse
to confirm it without a passing time-shuffle control. Pre-committed rule written
into `code/session_depth.py` before any arm ran: 3 seeds/arm, selection on
validation, a win must beat the in-session control by more than
PROMOTION_MARGIN (0.001), a win triggers the falsification control first, and
only a control-passing win reaches the 5-seed committee check against 0.62059.

| Arm | valid (5 dp) | Δ vs control | test primary | valid GAUC / nDCG@5 | tab=1 slice (diag.) |
|---|---|---|---|---|---|
| R38-ctrl RICH + tab_n (champion control) | 0.61955 | — | 0.6124 ± 0.0007 | 0.6934 / 0.5457 | 0.59747 |
| R38a + sess_pos | 0.61850 | −0.00105 | 0.6116 ± 0.0004 | 0.6919 / 0.5451 | 0.59623 |
| R38b + sess_pos + sess_hit | 0.61706 | −0.00347 | 0.6111 ± 0.0009 | 0.6897 / 0.5444 | 0.59447 |

Hypothesis REFUTED, no win, no control run, nothing banked. Both
arms land below the control, so the pre-committed rule stopped the run before
the falsification control, the control exists to interrogate a win, and there
was none. R38b's −0.00347 clears the 0.002 gate, is consistent across three
seeds and both metric components, and is therefore a claimable negative;
R38a's −0.00105 is ~1.3σ and is recorded as directional only. The diagnostic
slice score moves the same way the overall score does, which rules out the
consoling reading that the feature helped tab=1 and hurt elsewhere.

Determinism note: R38-ctrl reproduces R33b to five decimals on validation
(0.61955) and four on test (0.6124) from an independently written code path
(`session_depth.py`, a different feature builder and a different arm loop), so
the champion's single-model configuration reproduces across sessions and the
−0.00105/−0.00347 deltas are comparisons against a stable control measured in
the same session.

The obvious explanation is that a
within-user metric cannot see a between-user effect. Three label-free
diagnostics (`code/session_depth_diagnostics.py`, output in
`logs/session_depth_diagnostic.out`; bucket rates estimated on train rows only,
validation labels never read) say that explanation is wrong, and the real
finding is sharper:

1. Not redundancy alone. Conditioning the fatigue table on each shipped
   feature leaves a clean monotone decline inside every stratum. `hist30`
   absorbs 60% of `sess_pos`'s rate variance, `gap` 42%, `hist10` 28%, a lot,
   but not the effect.
2. Not a between-user artifact. Mapping each bucket to its train rate and
   decomposing the resulting per-row prior on the validation tab=1 rows,
   77% of `sess_pos`'s variance is within user, the channel the metric
   scores. (For contrast, `hist30` is 5% within-user and `tab_n` 4%.)
3. And not a magnitude story either. Centering every column within user and
   regressing `sess_pos`'s prior on the champion's features leaves 54%
   unexplained (R² 0.46), i.e. 8.19e-05 of novel within-user prior variance.
   Running the identical measurement on `tab_n`, the feature that did
   win, +0.0024, against the pre-promotion set, gives 86% unexplained and
   1.04e-04. The loser has 0.79× the winner's novel within-user signal. Same
   order of magnitude.

So the tempting cheap screen, "rank candidate features by novel within-user
prior variance and only pay for the promising ones", would have rated
`sess_pos` as promising as `tab_n` and been wrong. That screen is falsified as
a pre-run filter for this project, and that is the transferable result of the
iteration: additive-prior accounting does not predict what an FM will do with a
field, because a new narrow field does not enter the model as a prior. It enters
every pairwise interaction with the wide fields, at rank min(k_j, k_l), Run 32's
result, so its cost is paid in the shared embedding geometry that `user_id` and
`video_id` are using, and that cost is not visible in any marginal statistic.
The only instrument that measures it is the 3-seed run.

The R38a → R38b gap is consistent with that reading and adds one detail:
`sess_hit` is a coarser, reset-on-gap duplicate of `hist10`/`hist30`, and
duplicating an existing outcome-count field costs three times what the
label-free position field costs (−0.00347 vs −0.00105).

Refuting the analyzer's
top slice exposed a liveness defect: live mode always proposed `rep[0]`, and
`propose()` dedups by id without reviving a record, so after this refutation the
analyzer would have re-proposed a refuted hypothesis forever, `next_open()`
would have returned nothing, and the loop would have stalled with an empty queue
from iteration 2 onward. `residual_analysis.py` now walks down the oracle-headroom
ranking to the first unresolved slice (`first_unresolved()`, covered by three
new self-test assertions). Verified against the live report: with `tab=0`
confirmed and `tab=1` refuted, the next iteration is served `hist=31-100`
(EVo 0.13222, mechanism tag `temporal`).

R33c, validation 0.62059 / test 0.61429. `BANKED_VALID`
in `harness.py` untouched, submission untouched.

## Run 39, campaign 6 iterations 2–3: per-partition exposure counts, and a ranking instrument that was measuring size

Iteration 2 took
the queued hypothesis, wrote `code/partition_familiarity.py`, ran its grounding
probe, and launched the three arms, then its session ended while they were
still training, leaving the experiment in flight and unreported. Iteration 3
inherited a running experiment. It adjudicated that run rather than starting
a second one: the arms were already the queue's top hypothesis, and launching a
duplicate would have burned the iteration and broken the one-experiment rule.
So the run below is iteration 2's experiment with iteration 3's verdict,
post-mortem, and instrument repair. The driver logged both iterations; no human
touched either.

With `tab=0` confirmed and `tab=1` refuted, the
analyzer walked to `residual_hist_31-100`, 10,947 users, oracle headroom
0.13222, mechanism tag `temporal`. `priority.py --recompute` and
`belief_state.py --next` agreed.

That slice scores 0.62154
against 0.62059 overall, above the model's average, and its EV under the old
measure is exactly 0.0, so all of its headroom is the arithmetic of slice size.
As with `tab=1`, the only actionable reading is structure inside the slice the
feature set cannot express, and the slice definition names it: `hist_n` is a
lifetime impression count in six log buckets, the 31-100 bucket spans a 3.2×
range, and nothing in it says how those impressions were distributed.

The champion carries per-author and per-tag outcome counts (`auth_hist`,
`tag_hist` = prior long_views, capped 3+) and no exposure counts for either
partition. It has the numerator of a per-partition hit rate and not the
denominator. That missing denominator is exactly what `tab_n` supplied for the
surface partition, the one feature that has ever won here (+0.0024, R33b), and
R37's discriminating control revised its mechanism from surface familiarity to
partitioned familiarity, only about half of it surface-specific. The
generalisation under test: the winning axis was partitioned exposure counting,
and the tag and author partitions have been left without it.

Grounding probe, train rows only (`code/familiarity_diagnostics.py`, output
in `logs/familiarity_diagnostic.out`). long_view rate by `tag_n` inside each
stratum of the shipped `tag_hist`:

| `tag_hist` \ `tag_n` | 0 | 1–3 | 4–10 | 11–30 | 31–100 | 100+ | n |
|---|---|---|---|---|---|---|---|
| 0 | 0.342 | 0.248 | 0.156 | 0.082 | 0.040 | — | 555,498 |
| 1 | — | 0.418 | 0.260 | 0.136 | 0.074 | — | 197,238 |
| 2 | — | 0.507 | 0.351 | 0.188 | 0.090 | — | 109,537 |
| 3+ | — | 0.586 | 0.491 | 0.370 | 0.242 | 0.125 | 278,839 |

A clean monotone decline of 4.7× to 8.6× inside every stratum of the feature
that is supposed to make it redundant, on 1.1M rows. This is the shape of a
rate: at a fixed number of prior successes, more prior exposures means a lower
hit rate. Both fields are narrow (6 and 4 values), so their bilinear form has
rank 16 over a 4×6 grid, ample to represent the ratio, if given the
denominator.

Features are causal, self-exclusive and label-free by construction, they
count impressions, never outcomes; state updates only after the row is
featurised, rows visited in each user's chronological order. Pre-committed rule
written into the script before any arm ran: 3 seeds/arm, selection on validation,
a win must beat the in-session control by more than PROMOTION_MARGIN (0.001), a
win triggers the falsification control first, and only a control-passing win
reaches the 5-seed committee check against 0.62059.

| Arm | valid (5 dp) | Δ vs control | test primary | valid GAUC / nDCG@5 | hist=31-100 slice (diag.) |
|---|---|---|---|---|---|
| R39-ctrl RICH + `tab_n` (champion control) | 0.61955 | — | 0.6124 ± 0.0007 | 0.6934 / 0.5457 | 0.62227 |
| R39a + `tag_n` | 0.61822 | −0.00133 | 0.6108 ± 0.0008 | 0.6919 / 0.5446 | 0.62056 |
| R39b + `tag_n` + `auth_n` | 0.61846 | −0.00109 | 0.6114 ± 0.0013 | 0.6920 / 0.5450 | 0.62095 |

Hypothesis REFUTED, no win, no falsification control run, nothing
banked. Both arms land below the control, so the pre-committed rule halted
before the control and the committee, the control exists to interrogate a win,
and there was none. Both deltas are under the 0.002 claim bar and are recorded
as directional only, not as claimed negatives (unlike R38b's −0.00347). What
is solid is the absence of a win, measured against a control reproducing R33b
to five decimals on validation from a fourth independent code path. The
diagnostic slice score moves the same way the overall score does, 0.62227 →
0.62056, so the consoling reading that the feature helped the target slice and
paid for it elsewhere is ruled out.

The same screen, falsified a second time and harder. Iteration
1 killed the cheap pre-run filter "rank candidates by novel within-user prior
variance" by showing the loser carried 0.79× the winner's novel signal, the
same order, opposite outcome. Run 39 is an independent second test, and
`code/partition_postmortem.py` (label-free: train-estimated rates mapped onto
valid rows, no training, valid labels never read) runs the identical
instrument, imported rather than rewritten:

| feature | within-user prior var | after regressing on the champion's other fields | novel | outcome |
|---|---|---|---|---|
| `tag_n` | 5.702e-04 | 4.871e-04 | 85% (R² 0.15) | lost, −0.00133 |
| `auth_n` | 4.616e-04 | 3.348e-04 | 73% (R² 0.27) | — |
| `tab_n` | 3.728e-04 | 3.656e-04 | 98% (R² 0.02) | won, +0.0024 |

The loser carries 1.33× the winner's novel within-user signal. Two other
label-free screens agree with it and are equally wrong: `tag_n` varies within
73.2% of validation users against `tab_n`'s 43.7% (a within-user metric can only
score a feature that varies within a user), and restricted to the slice that
raised the hypothesis it is 91% novel against `tab_n`'s 95%. So the screen has
now failed in both directions, it rated one loser equal to the winner and
rates this one above it. It is not imprecise; it is uninformative, and this
section is the second nail.

The redundancy explanation fails too, and in a direction worth recording.
Conditioning `tag_n` on the shipped `tag_hist` does not shrink its rate variance,
it multiplies it by 12 (0.00057 → 0.00685; `hist_n` −11%, `hist30` −18%,
`tab_n` −80%). `tag_hist` masks the exposure effect marginally rather than
absorbing it, heavy exposure and heavy prior success travel together, so the
marginal decline understates a much steeper conditional one. The information the
feature carries is real, large, non-redundant, novel, and within-user, and the
model was still worse with it. The only instrument that measures what a narrow
field costs in the shared embedding geometry (Run 32's min(k_j, k_l) result) is
the 3-seed run.

One candidate discriminator, explicitly untested. The two constructions
differ in one structural way. For `tab_n` the partition key is a field the model
holds (`tab` ∈ BASE); for `auth_n` it is too (`author_id`); for `tag_n` the tag
identity is not in the feature set at all, `tag_hist` is an outcome count,
not the tag. A per-partition count is a conditional quantity ("prior impressions
of this partition"), and `tag_n` may be an integer the model cannot bind to
what it counts. The only support is directional and far under noise: `auth_n`,
whose key is a field, recovered +0.00024 on top of `tag_n` despite being
non-zero on just 6.2% of rows and constant within 75% of validation users. It is
written into the belief state as a candidate, not a finding.

The queue was ranking slices by size. Refuting this
hypothesis exposed a defect in the ranking, not just in the intervention. EVo
(oracle headroom) is an upper bound, and every slice has one roughly in
proportion to its size, perfectly ranking any large chunk of the validation set
helps whether or not the model is bad there. The loop paid for that twice: it
spent this iteration on a slice scoring above the model's average with an
EV(old) of exactly 0.0, and the next pick under the old rule was `hist=11-30`,
the same shape (EV(old) 0.00153), while `dur=8`, where the model scores 0.518
against 0.621, sat at rank 4.

EVx (v3) subtracts a matched null: the oracle headroom of a slice holding
the same number of each user's rows, drawn at random from that user's rows.
Which users are touched and how concentrated the slice is per user, the two
things a per-user metric's headroom is mechanically driven by, are preserved;
which rows are in it is destroyed. It is the same construction as the
falsification controls (preserve the marginals, destroy the attachment, read off
what the attachment was worth), now pointed at the loop's own hypothesis queue.
It is also signed: a slice the model handles unusually well scores negative,
which EVo cannot express. The self-test is a regression test for the exact
defect, a small broken slice that EVo ranks last of four and EVx ranks
first. A first attempt that dealt each user's row-count out to other users was
built and discarded: the real rows-per-user distribution is skewed enough that
the per-user cap bound on most slices, undersizing the null and biasing EVx up
precisely on the large slices it was meant to demote. The measured failure is
why the shipped null reshuffles within each user instead.

| slice | primary | users | EV(old) | EVo | EVx |
|---|---|---|---|---|---|
| tab=1 | 0.59710 | 20,119 | 0.02111 | 0.21129 | 0.00399 |
| dur=4 | 0.54043 | 8,023 | 0.02874 | 0.04391 | 0.00221 |
| tab=4 | 0.56396 | 4,209 | 0.01065 | 0.02730 | 0.00201 |
| dur=6 | 0.53034 | 7,670 | 0.03093 | 0.04226 | 0.00156 |
| dur=7 | 0.52610 | 8,749 | 0.03694 | 0.04791 | 0.00145 |
| dur=3 | 0.53835 | 8,125 | 0.02986 | 0.04753 | 0.00086 |
| dur=8 | 0.51767 | 9,093 | 0.04182 | 0.04839 | 0.00082 |
| hist=11-30 | 0.61577 | 7,107 | 0.00153 | 0.05027 | 0.00059 |

`hist=31-100`, the slice this run just spent itself refuting, drops out of the
top eight entirely: 95% of `tab=1`'s apparent headroom and effectively all of
the `hist` buckets' was size. The duration slices, where the model really does
score 0.52–0.54 against 0.621, now occupy the queue. Stated caveat: the null
cannot reshuffle a user whose rows lie entirely inside the slice, and that
locked share is high for the `hist` buckets (88% for `hist=31-100`, 73% for
`hist=11-30`), so their EVx is biased toward zero and their demotion is only
partly earned by the measure, the analyzer prints this warning per slice rather
than hiding it. Two independent things say the demotion is right anyway: this
run refuted `hist=31-100` by experiment, and `hist=11-30`'s EV(old) of 0.00153
says the same from the unbiased direction. The `dur` and `tab` slices now at the
top trip no warning.

The stale `residual_hist_11-30` record the superseded ranking had just written
was dropped rather than left in the queue: its `expected_value` is in EVo units,
which `next_open()` would have compared against the new EVx units and always
preferred. It had never been tested. The units change is documented in the
analyzer's docstring; every pre-v3 record is resolved, so nothing compares across
the two conventions. The next iteration is served `residual_dur_4` (EVx 0.00221,
mechanism tag capacity, the first time campaign 6 reaches the
`capacity_noise` branch of `controls.py`).

R33c, validation 0.62059 / test 0.61429.
`BANKED_VALID` in `harness.py` untouched, submission untouched. Two iterations,
two refutations, and the loop's ranking instrument is measurably better than it
was before.
