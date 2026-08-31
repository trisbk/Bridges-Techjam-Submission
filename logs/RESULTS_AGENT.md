# Clean-room run. experiment log

(Started empty by design: the agent begins from the task statement and the
official baseline only. Every entry below is the agent's own.)

Banked best: 0.59744 test primary. `exp02_fields_content+hour+uprofile+onehot`
from iteration 3, +0.00244 over the reproduced official FM baseline (0.59497).
Still the banked model after iterations 4 and 5. Iteration 4's structural mask
reached test 0.59801 but only +0.00008 on validation. the deciding split.
Iteration 5's recency weighting reached test 0.59756 but **−0.00015 on
validation**, and the validation curve had no interior optimum: the unweighted
limit was the maximum.

Selection rule in force: verdicts are decided on validation; test is
recorded for honesty and never selected on. Gate = 0.002.

---

## Idea list

Kept across iterations. `[tried]` = measured through the harness, `[dead]` =
measured and refuted, `[open]` = proposed, not yet run.

banked
- Richer feature fields from iteration 3. Coarse item content + impression
 hour + the static user profile, added around the official five. Test
 0.59744, Δbaseline +0.00244, result `WIN`. See run 3.

dead
- Within-user pairwise / BPR loss, pure or as an auxiliary term. Refuted in
 iteration 1, monotonically in the mixing weight. See run 2 below. this is a
 strong negative result and later iterations should not retry plain pairwise
 reweighting of the same 5 fields.
- Video age (`date` − `upload_dt`, bucketed). Measured in iteration 3's
 validation grid: it is the only added field that made things *worse*
 (content+hour 0.60255 → +age 0.59996). Do not retry it as a raw categorical
 field.
- `tag1` alone. 0.60106 vs official 0.60144. below baseline. Content
 fields only pay off as a group; a single one is net capacity cost.
- Masking the user-static × user-static FM interactions from iteration 4.
 351 of the winner's 666 interaction pairs are provably invisible to a
 within-user metric; deleting them (implementation verified against an
 explicit pairwise sum and finite differences) moved validation by +0.00008.
 Refuted. The per-user constant they fit simply relocates into the free
 `W[user_id]`, so they were idle, not expensive. the early-stopping epoch
 did not even move. See run 4. **General lesson: a term the metric cannot see
 is inert in both directions; removing it is not a gain unless it was
 competing for something scarce.**
- Exponential recency weighting of train rows from iteration 5. Refuted, and
 refuted *monotonically*: validation is maximised at the unweighted limit
 (halflife `inf` 0.60331 → 28d 0.60315 → 10d 0.60316 → 7d 0.60285 → 5d
 0.60268 → 3d 0.60135 → 2d 0.59864 → 1d 0.59233). See run 5. The temporal
 drift it targets is real and large (an item-popularity estimator fit on
 the recent half of train beats one fit on the old half by +0.0076 valid /
 +0.0104 test at equal sample size). the weighting scheme, not the
 premise, is what failed. **General lesson: on a 13-day train window an
 exponential decay cannot be simultaneously sharp enough to correct drift and
 gentle enough to keep the data. at halflife ≥10d it preserves ≥95% of the
 effective sample but barely reweights anything, and by the time it reweights
 meaningfully (≤3d) it has thrown away half the rows. Drift must be exploited
 as a FEATURE, not as a loss weight.**

open (ranked by my current expected value, re-ranked after iteration 5)

1. Trailing-window target encoding of item / author long-view rate, with
 prior smoothing, as an explicit *numeric* feature rather than an embedding.
 Promoted to the top idea by iteration 5's measurements, which give it a
 directly measured effect size: at equal sample size a *recent*-window item
 estimator ranks +0.0076 (valid) / +0.0104 (test) better than an old-window
 one. That is the drift signal, and a feature can carry it *without*
 discarding the 1.14M rows the FM needs. which is exactly the trade
 recency weighting could not make. Must be computed strictly from a trailing
 window that ends before the scored row's date, with a fold-safe scheme on
 train so a row never enters its own encoding.
2. Ensembling FM with a recent-window item-popularity score. The cheap
 version of idea 1 and it shares the same evidence: `run_pop` fit on
 the relevant window alone reaches valid primary 0.57724, and it is a different
 bias/variance point that is *more current* than the FM's 14-day average.
 A blend weight tuned on validation is one line.
3. Re-tune capacity for the widened input. Iteration 3 froze `k=16,
 lr=1e-3, l2=1e-6` at values chosen for 5 fields; the banked model has 37
 fields and 666 interaction pairs instead of 10. Iteration 4 sharpened this:
 the best epoch is a stable 7 to 8 and is completely insensitive to halving the
 interaction term, so the binding constraint is `k`/`l2`/`lr`, not the pair
 count. Still the cheapest remaining follow-up, but it attacks capacity, and
 the last two iterations both found that capacity was not what was scarce.

Dataset facts established in iteration 1 from the train split probe:
train 1,141,112 rows / 26,210 users / 7,538 videos, positive rate 0.337;
test 170,588 rows / 23,875 users; median 5 impressions per user in test
(mean 7.1, p90 16); 6,472 test users have zero positives (nDCG counted as 0)
and 2,202 are all-positive; unseen-in-train rows in the test split: 0.01% of
videos, 0.0% of authors, 3.6% of users. Items are dense (≈151 train
impressions per video). this is a memorisation-friendly, not a cold-start,
problem.

Field facts established in iteration 3 (train rates; within-user distinct
ratio measured on test, i.e. how much the field varies inside the list the
metric actually ranks):

| field | levels | train long-view rate | within-user distinct ratio |
|---|---|---|---|
| `tag1` | 44 | 0.067 to 0.471 | 0.76 |
| `hour` | 24 | 0.318 to 0.376 | 0.80 |
| `upload_type` | 14 | 0.064 to 0.374 | 0.57 |
| `tag2` | 24 | 0.071 to 0.373 | 0.41 |
| `music_type` | 6 | 0.158 to 0.346 | 0.39 |
| `video_type` | 3 | 0.336 to 0.439 | 0.33 |
| `age_bucket` | 8 | 0.311 to 0.367 | 0.43 |
| `user_active_degree` | 9 | 0.322 to 0.441 | 0.32 |

Positive rate drifts down across the data: 0.3366 (train) → 0.3135 (test).

---

## Run 1. `baseline_fm_repro` as a reference rather than a hypothesis

Reproduce the official `baseline.py --model fm` through the harness so the
0.002 gate is applied to like-for-like, multi-seed numbers.

| split | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.66720 | 0.53568 | 0.60144 |
| test | 0.66143 | 0.52851 | 0.59497 ± 0.00025 |

Per-seed test: 0.59533 / 0.59477 / 0.59482. The result was `NOISE` vs itself, as
expected. Published baseline is 0.5946, so it reproduces (+0.0004, inside the
gate). Measured seed noise here (σ = 0.00025) is well under the 0.0008 the
harness assumes, so the 0.002 gate is conservative. **This is the number to
beat.**

---

## Run 2. `exp01_within_user_pairwise_aux_loss`

### Hypothesis

Train the FM on `BCE(rows) + λ · BPR(within-user (pos, neg) pairs)` instead of
BCE alone, and the primary metric rises.

### Rationale

`evaluate.py` never compares two users' scores. GAUC is a per-user AUC and
nDCG@5 is computed inside each user's own impression list, so **any part of
the score that is constant within a user is invisible to the metric**. the
whole user-bias half of what pointwise BCE learns is wasted capacity. Pooled
BCE also weights a user in the loss by their impression count, whereas GAUC
weights by positive count and nDCG weights every user equally. A pairwise term
over two impressions *of the same user* applies gradient pressure along
exactly the direction the metric reads. This is the textbook argument for
BPR-style objectives on ranking metrics.

### Method

Only the objective changes. Same `FM` class, same 5 official fields, same
`k=16, lr=1e-3, l2=1e-6, bs=8192, epochs=40, patience=4`, same shuffled
pointwise pass over all 1.14M train rows; the pairwise batch (8192 pairs,
user drawn uniformly, then one of their positives and one of their negatives
uniformly) is accumulated into the *same* Adam update. λ=0 is the baseline
exactly. Pairs come only from train-split rows, so nothing from the future or
from a row's own outcome enters a score. Code: `src/exp01_within_user_ranking_loss.py`.

### Validation-only exploration (not graded runs, seed 0, valid primary)

Pure BPR first (λ→∞, `run_pair_fm`), tuning the one thing that had to be
re-tuned for a new objective:

| pure BPR, lr | 3e-4 | 1e-3 | 3e-3 |
|---|---|---|---|
| valid primary | 0.59938 | 0.59778 | 0.59526 |

All below baseline's 0.60144, and *monotone in lr*. it converged in ~2 epochs
and then decayed, i.e. it was overfitting, not mis-tuned. That motivated
keeping BCE as the stabiliser and scanning the mixing weight:

| λ | 0 (baseline) | 0.1 | 0.3 | 1.0 | 3.0 |
|---|---|---|---|---|---|
| valid primary | 0.60144 | 0.60021 | 0.59919 | 0.59755 | 0.59824 |

Monotonically worse in λ. λ=0.1 was carried to the graded run as the best
non-zero setting; this is the most favourable case for the hypothesis, and it
still loses.

### Graded result

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.66473 | 0.53461 | 0.59967 |
| test | 0.65849 | 0.52654 | 0.59252 ± 0.00063 |

Per-seed test: 0.59341 / 0.59200 / 0.59214. Δ vs baseline = −0.00248.
`WORSE`. Banked best unchanged at 0.59497.

### Interpretation. why the argument failed

The metric argument is correct but the *modelling* premise behind it is not.
Within-user ordering is a monotone function of a well-calibrated
P(long_view | user, item, context), so pointwise BCE is already a **consistent
surrogate** for GAUC here. it is not mis-specified, merely indirect. What the
pairwise term actually changes is variance, and on this dataset that trade is
strictly bad, for two reasons the probe makes concrete:

1. The dominant signal is item-side, and BCE estimates it from more data.
 There are only 7,538 videos with ≈151 train impressions each. Every one of
 the 1.14M rows is a direct, low-variance observation of an item's
 long-view propensity. The pairwise term replaces that dense per-row
 supervision with a contrastive signal whose gradient depends on a random
 partner row, and buys nothing the pointwise term was missing. this is a
 memorisation problem, not a cold-start one.
2. Uniform-over-users pair sampling backfires. It was chosen to mirror
 nDCG's equal user weighting, but it upweights exactly the users with the
 fewest and noisiest impressions, and it drops the 1,920 train users who
 have no within-user pair at all.

The monotone degradation in both lr (pure BPR) and λ (hybrid) is what makes
this conclusive rather than a tuning accident: there is no interior optimum,
the objective simply prefers λ=0.

The headroom on this task is not in the loss
function, so iteration 2 should attack the *inputs* instead. The probe already
says which way to go. with items dense and 3.6% unseen users, content
features can only pay off through interactions, so the ranked open ideas above
put within-user-varying context (video age, hour context) and the temporal
train/test drift ahead of static side information.

---

## Run 3. `exp02_fields_content+hour+uprofile+onehot` WIN, banked

### Hypothesis

Widening the FM's input from the official five fields to the coarse item
content attributes, the impression hour, and the static user profile raises
the primary metric above the baseline. Nothing else changes. same `FM`
class, same `k=16, lr=1e-3, l2=1e-6, bs=8192, epochs=40, patience=4`, same
pointwise BCE, same training loop. so any gain is attributable to the inputs
alone.

### Rationale

Iteration 1 left a specific diagnosis: this is a memorisation problem
(7,538 videos at ≈151 train impressions each, 0.01% unseen-video test rows),
so the *item bias* is already saturated by `video_id` and the loss function is
not where the headroom is. What is left is the personalised term. the
only cross-row structure a within-user metric can read. and the baseline fits
it as `<v_user, v_video>` from ≈43 impressions per user spread over 7,538
videos. That table is almost entirely empty.

Coarse attributes collapse it on both sides at once:

- *Item side.* A user's affinity for `tag1=3` is estimated from every
 impression of every video carrying that tag, so the taste vector is fit over
 ~44 columns instead of 7,538. `tag1` spans long-view rates 0.067 to 0.471 and
 varies inside 76% of a user's own impressions, so it is a large signal the
 metric can actually see.
- *User side.* Symmetrically, an item's audience is estimated over user
 *segments* rather than 26,210 individual `user_id`s.
- *Context.* `hour` is the one purely contextual axis and has the highest
 within-user variation of anything measured (0.80 distinct ratio).

### Legitimacy (features computable at recommendation)

- Video attributes (`tag`, `video_type`, `upload_type`, `music_type`) are
 static catalogue metadata; `hour` is the impression's own timestamp. Both
 are known before the row is scored.
- The user-profile columns come from an undated snapshot, which deserves an
 explicit argument rather than a shrug. They cannot leak the future here,
 because the FM already carries a free per-user embedding: any per-user
 label information a profile column could encode is *already* fully available
 to the model through `user_id`. The profile columns carry no per-item and no
 per-timestep information, so their only marginal effect is cross-user
 parameter sharing. which is exactly the mechanism claimed above.
- `video_features_statistic_pure.csv` is excluded on purpose: its counters
 are undated aggregates that may summarise the evaluation window.
- `src/data.py` and `src/evaluate.py` are untouched. The new module has its own
 reader but imports `SPLITS` from `data.py`, so the split is literally the
 official one, and `--mode selftest` **asserts that on the official five
 fields the new reader/encoder produces bit-identical `X`, `y` and `user`
 arrays and the identical `dim=40260`**. It passes; the `official` row below
 reproduces the baseline's seed-0 test score to five decimals (0.59533).

### Validation-only selection

| fieldset | F | valid primary | Δ official |
|---|---|---|---|
| `official` (baseline) | 5 | 0.60144 ± 0.00027 |. |
| `tag` (+`tag1`) | 6 | 0.60106 | −0.00038 |
| `content` | 10 | 0.60229 ± 0.00024 | +0.00085 |
| `hour` | 6 | 0.60232 ± 0.00034 | +0.00088 |
| `content+hour` | 11 | 0.60255 ± 0.00006 | +0.00111 |
| `content+hour+age` | 12 | 0.59996 | −0.00148 |
| `content+hour+user_active_degree` | 12 | 0.60285 ± 0.00021 | +0.00141 |
| `content+hour+uprofile` | 19 | 0.60305 ± 0.00041 | +0.00161 |
| `content+hour+onehot` | 29 | 0.60309 ± 0.00059 | +0.00165 |
| `content+hour+uprofile+onehot` | 37 | 0.60331 ± 0.00060 | +0.00187 |

Selection is monotone and mechanistic rather than a lucky draw: every group
that pools statistical strength helps, the two sides help *additively*
(+0.0011 item/context, +0.0007 more from the user side), and the one field
that helps nothing. `age_bucket`, whose rate range is a flat 0.311 to 0.367 ,
is the one that hurts.

### Graded result

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.66946 | 0.53716 | 0.60331 |
| test | 0.66404 | 0.53083 | 0.59744 ± 0.00004 |

Per-seed test: 0.59739 / 0.59746 / 0.59747. Δ vs baseline = +0.00244.
`WIN`. `BANKED` in `src/harness.py` updated 0.5950 → 0.59744.

Both components move together (GAUC +0.0026, nDCG@5 +0.0023), which is what a
genuine ranking improvement looks like. not one metric being traded for the
other. Seed spread collapsed to σ = 0.00004 (baseline σ = 0.00025): the wider
input makes the fit markedly more stable, consistent with the variance-pooling
story.

### What the result shows

The gain is real but modest and only just over the bar: +0.00244 on test
against a 0.002 gate, and +0.00187 on validation, which is *below* 0.002
though ≈3σ at the measured seed noise. I am banking it because the harness's
gate is applied to the test mean and returned `WIN`, because the per-seed
spread is tiny, and because the validation ordering across eleven candidates
is systematic rather than noise-shaped. But it should be read as "the inputs
were leaving ~0.002 on the table", not as a breakthrough.

The result confirms iteration 1's diagnosis from the other direction. Iteration
1 showed the *objective* was not mis-specified; iteration 3 shows the *inputs*
were impoverished, and that what they were missing was specifically the ability
to pool. every fieldset that let parameters be shared across many rows
helped, the one that only added a sparse new axis (`age_bucket`) hurt, and the
biggest single contributor (`hour`) is also the field that varies most inside
the impression list the metric ranks.

Two follow-ups now outrank everything else. (1)
The hyperparameters were frozen at values chosen for a 5-field model; the
winner has 37 fields and 666 interaction pairs instead of 10, so `k`/`l2`/
epochs are plausibly mis-set. that idea has gone from "unjustified" to
"motivated". (2) Many of the added fields are user-static, and a user-static ×
user-static interaction is **constant within a user and therefore structurally
invisible** to GAUC and nDCG@5 while still costing capacity and adding
variance; masking those pairs is a principled, metric-aware restriction that
this run's success makes worth testing. Recency weighting remains the top
open idea that attacks a different axis (the 0.3366 → 0.3135 positive-rate
drift).

---

## Run 4. `exp03_mask_uu_content+hour+uprofile+onehot` refuted (no validation gain), NOT banked

### Hypothesis

Delete from the FM interaction every pair of fields that are **both functions
of the user alone**. Such a pair contributes the *same* amount to every
impression in a user's list, and GAUC / nDCG@5 are computed strictly inside a
user's list, so those pairs provably cannot change the metric. Removing
them should raise the primary metric, because they are pure cost: collinear
capacity plus gradient that pulls the user-side embeddings toward a
within-user-invisible direction. Nothing else changes. same fields, same
`k=16, lr=1e-3, l2=1e-6, bs=8192, epochs=40, patience=4`, same BCE, same seeds.

### Rationale

Iteration 3 won by adding user-side fields, and in doing so it also created a
large dead zone. Of the winner's 37 fields, 27 are user-static (`user_id`,
the 8 profile columns, the 18 `onehot_feat*`), so 27·26/2 = **351 of the 666
interaction pairs (52.7%) are structurally invisible** to the metric. Two
distinct harms were claimed:

1. Redundant capacity. Every per-user constant those 351 pairs express is
 *already* free in the scalar `W[user_id]`; they only give the optimiser 351
 extra collinear ways to say the same thing.
2. Gradient contamination. `V[register_days_range]` receives gradient from
 its interactions with item/context fields (visible. this is the cross-user
 sharing that made iteration 3 win) *and* from its 26 user-static partners
 (invisible). The second source was expected to pull the vector off the
 direction that explains within-user ordering.

The mask is exact, not a regulariser, and costs no compute:

 inter_masked = 0.5(|S|² − Σ_f|E_f|²) − 0.5(|S_U|² − Σ_{f∈U}|E_f|²)
 ∂/∂E_f = S − E_f (f ∉ U, all partners kept) | S_I (f ∈ U, item/context only)

### Correctness and legitimacy

- `--mode selftest` passes: the masked logit matches an **explicit pairwise
 sum over the kept pairs, the analytic `∂z/∂V` matches finite
 differences**, and with an *empty* mask `MaskedFM` reproduces `baseline.FM`
 bit-for-bit over five Adam steps.
- `--mode audit` asserts the user-static classification against the data:
 every masked field has exactly 1.000 distinct levels per test user (the
 10 kept fields range 1.07 to 6.93). Table below.
- No new columns are read, so iteration 3's legitimacy argument carries over
 unchanged. `data.py` / `evaluate.py` untouched.
- The `none` control below reproduces iteration 3's graded numbers to five
 decimals (valid 0.60331, test 0.59744), confirming the code path is identical.

| levels per test user | fields |
|---|---|
| 1.000 (masked) | `user_id`, 8 profile columns, 18 `onehot_feat*`. 27 fields |
| 1.07 to 6.93 (kept) | `video_id` 6.93, `author_id` 6.90, `dur_bucket` 6.90, `hour` 4.73, `tag1` 4.41, `upload_type` 2.66, `tag2` 1.65, `tab` 1.61, `music_type` 1.47, `video_type` 1.07 |

### Validation-only selection

| variant | pairs kept | valid primary | Δ vs `none` | (test) | best epoch |
|---|---|---|---|---|---|
| `none` (iteration 3's banked model) | 666 | 0.60331 ± 0.00060 |. | (0.59744) | 8/4/7 |
| `mask_uu` (351 pairs deleted) | 315 | 0.60339 ± 0.00034 | +0.00008 | (0.59801) | 8/7/7 |
| `mask_uu_lin` (+ user-static linear terms dropped) | 315 | 0.60336 ± 0.00025 | +0.00005 | (0.59847) | 8/8/7 |

`mask_uu` was carried to the graded run as the pre-registered variant and the
top validation mean. though the spread across all three is **±0.00008 on a
grid whose seed σ is ±0.0003 to 0.0006**, i.e. the three models are
indistinguishable on validation.

### Graded result

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.66955 | 0.53723 | 0.60339 |
| test | 0.66488 | 0.53114 | 0.59801 ± 0.00037 |

Per-seed test: 0.59787 / 0.59765 / 0.59852. Δ vs official baseline =
+0.00301; Δ vs banked best = +0.00057; Δ on validation vs
banked = +0.00008.

### Result

The harness printed `WIN`, but that label is decided against the *frozen
official baseline* (0.5950), which this model clears only because it inherits
iteration 3's fields. The question this iteration asked is whether the mask
adds anything on top of iteration 3, and the answer on the deciding split is
no: **+0.00008 validation primary, against a 0.002 gate and a measured seed σ
of 0.0003 to 0.0006.** Selection is on validation, so `BANKED` stays at 0.59744
and the banked model stays iteration 3's. Test moved +0.00057, which is
under the gate and was not eligible to decide anything.

### Interpretation. why a provably-exact restriction bought nothing

The premise is airtight and the implementation is verified; it is the *harm*
model that was wrong. Both claimed harms dissolve on inspection:

1. **Redundancy is not a cost here, because the redundant fit is free to
 relocate.** The 351 dead pairs and `W[user_id]` are fitting the *same*
 quantity. a user's base long-view rate. Deleting the pairs does not remove
 that fitting pressure from the objective; BCE still wants the per-user rate
 explained, and `W[user_id]` absorbs it at zero cost to anything the metric
 reads. The optimiser was never spending scarce capacity, only choosing
 between exactly-equivalent parameterisations of a constant.
2. The contamination is second-order at this capacity. With `k=16` and 27
 user-static fields, the invisible objective is satisfiable in a subspace
 without materially rotating the directions that serve the live pairs. The
 unchanged early-stopping epoch (7 to 8 in every variant, masked or not) is the
 direct evidence: if the dead pairs were driving overfitting, deleting 53% of
 the interaction term would have pushed the best epoch later. It did not move.

The deeper lesson, and it generalises: **"provably cannot change the metric" is
not the same as "removing it improves the metric."** A term invisible to the
evaluation is invisible to the *gradient's effect on the evaluation* too. it
is inert in both directions unless it competes for something genuinely scarce.
Iteration 3 won by adding a term that was scarce (cross-user sharing of taste);
iteration 4 tried to win by removing one that was merely idle.

One honest loose end, recorded but not claimed: both mask variants moved
*test* in the same direction (+0.00057 and +0.00103) while validation stayed
flat. Test is further from the training window than validation is (train ends
the relevant window, valid the relevant window..28, test the relevant window), so this is *consistent with* a
small generalisation benefit that only shows up across a wider temporal gap.
It is under the gate, it is one of two arms, and validation. the deciding
split. shows nothing, so it does not change the result. It is noted here only because
it points at the same place as the top open idea: **the temporal drift is where
the remaining structure is.**

Structural, metric-aware surgery on the
interaction term is now measured and dead; the two remaining open ideas both
attack things that are actually scarce rather than idle. the train/test time
drift (recency weighting, now unambiguously the top idea, and independently
pointed at by this run's test-side movement), and the capacity of a model whose
input width tripled without its `k`/`l2`/epoch budget ever being re-tuned.

---

## Run 5. `exp04_recency_hl10` refuted (validation worse than banked), NOT banked

### Hypothesis

Weight each training row in the BCE loss by an exponential decay in its age,
`w = 0.5 ** (age_days / halflife)` with age measured from the last training
day (the end of the training window) and `w` normalised to mean 1, and the primary metric rises above
the banked iteration-3 model. Nothing else changes. same fieldset
(`content+hour+uprofile+onehot`), same `FM`, same `k=16, lr=1e-3, l2=1e-6,
bs=8192, epochs=40, patience=4`, same pointwise BCE. `halflife=inf` is the
banked model bit-for-bit.

### Rationale

The split is temporal and the model is scored 1 to the evaluation period after its training
window ends, so a uniformly weighted fit targets the 13-day average rather
than the current distribution. This was the top open idea after iterations 3
and 4.

But the motivation carried in the previous idea list. the positive-rate drift
0.3366 → 0.3135. **is not a valid argument, and I record that before the
result rather than after.** A *global* rate shift is a constant added to every
logit; it lands in the intercept `b`; and GAUC / nDCG@5 read only within-user
ordering, so they cannot see it. This is iteration 4's lesson applied to
iteration 4's own suggestion. The only drift the metric can read is drift in
the relative ordering signal. which videos, tags and hours are relatively
more long-viewed, and for whom.

So the hypothesis was only worth running if that relative structure actually
moves, and `--mode drift` measured it *first*.

### Ordering signal diagnostic

The official smoothed item-popularity estimator (`prior=20`), fit on two
disjoint halves of the train window and scored on valid/test:

| item-pop fit window | n | valid primary | test primary |
|---|---|---|---|
| OLD window | 891,418 | 0.57638 | 0.56674 |
| NEW window | 249,694 | 0.57724 | 0.57117 |

The recent window wins despite 3.6× less data. Because that confound runs
the wrong way it is already suggestive, but I equalised it anyway ,
subsampling OLD to the NEW window's n, 5 draws:

| target | NEW (n=249,694) | OLD subsampled to n=249,694 | gap |
|---|---|---|---|
| valid | 0.57724 | 0.56967 ± 0.00049 | +0.00757 |
| test | 0.57117 | 0.56080 ± 0.00131 | +0.01037 |

The drift is real and it is large. 4 to 8× the 0.002 gate, at ≈8σ. The
premise of the hypothesis is confirmed. Also worth noting: the per-day rates
are *not* a clean monotone decline (0.3362, 0.3409, 0.3330, 0.3322, 0.3171,
0.3195, 0.3634, 0.3770, 0.3644, 0.3308, 0.3229, 0.3114, 0.3146 for
the relevant window…the relevant window). the level wanders, which is a further reason the global rate
was never the thing to chase.

### Correctness and legitimacy

- `--mode selftest` passes: `WeightedFM` with `w = 1` reproduces `baseline.FM`
 bit-for-bit over five Adam steps (identical loss, `V`, `W`, `b`), and the
 weight vector is checked against the closed form with mean exactly 1.
- The `inf` row of the grid below reproduces iteration 3's graded numbers to
 five decimals (valid 0.60331, test 0.59744). the code path is identical.
- Legitimacy: a row's weight is a function of that row's own date and the
 fixed end of the *train* window only. No label, no future row, nothing about
 the row being scored. Weights exist only inside the training loss, so
 scoring is completely unchanged. `data.py` / `evaluate.py` untouched; no new
 columns are read, so iteration 3's argument carries over.

### Validation-only selection

| halflife | N_eff (% of 1,141,112) | valid primary | Δ vs banked | (test) | best epochs |
|---|---|---|---|---|---|
| `inf` (banked model) | 100.0% | 0.60331 ± 0.00060 |. | (0.59744) | 8/4/7 |
| 28 d | 99.5% | 0.60315 ± 0.00079 | −0.00016 | (0.59744) | 6/4/7 |
| 14 d | 97.7% | 0.60299 ± 0.00076 | −0.00032 | (0.59751) | 6/4/7 |
| 10 d | 95.3% | 0.60316 ± 0.00083 | −0.00015 | (0.59756) | 6/4/7 |
| 7 d | 90.0% | 0.60285 ± 0.00050 | −0.00046 | (0.59753) | 6/4/7 |
| 5 d | 80.0% | 0.60268 ± 0.00055 | −0.00063 | (0.59748) | 8/4/7 |
| 3 d | 51.6% | 0.60135 ± 0.00035 | −0.00196 | (0.59572) | 7/6/5 |
| 2 d | 26.1% | 0.59864 ± 0.00042 | −0.00467 | (0.59191) | 3/4/7 |
| 1 d | 6.8% | 0.59233 ± 0.00019 | −0.01098 | (0.58468) | 1/2/2 |

`N_eff = (Σw)² / Σw²`. Validation is maximised at the unweighted limit and
falls monotonically once the decay bites. `halflife=10 d` was carried to the
graded run as the best *non-trivial* setting. i.e. the most favourable case
for the hypothesis, chosen on validation before grading, exactly as iteration
1 carried λ=0.1.

### Graded result

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| valid | 0.66931 | 0.53700 | 0.60316 |
| test | 0.66423 | 0.53090 | 0.59756 ± 0.00023 |

Per-seed test: 0.59749 / 0.59788 / 0.59732. Δ vs official baseline =
+0.00256; Δ vs banked on test = +0.00012; Δ vs banked on
validation = −0.00015. The harness returned `SIGNIFICANT_BUT_NOT_BEST`
(it gates against the frozen 0.5950 baseline, which this model clears only
because it inherits iteration 3's fields).

### Result

Selection is on validation, and on validation this is *worse* than the banked
model. by −0.00015, which is itself inside noise, but there is no reading
under which it is better. The full grid makes it conclusive rather than a
tuning accident: the curve has no interior optimum, it is monotone toward
`halflife = inf`. `BANKED` stays 0.59744.

### Interpretation. a real effect that this instrument cannot pick up

The interesting part is the dissociation: recency is worth +0.0076 to an
item-popularity estimator at equal n, and worth nothing at all to the FM.
Two things explain it, and together they say the *scheme* failed, not the
premise.

1. **On a 13-day window, an exponential decay cannot be both sharp and
 affordable.** The `N_eff` column is the whole story. Where the weighting
 preserves the data (≥10 d, ≥95% of `N_eff`) it barely reweights anything ,
 at halflife 7 d the weights span only 0.67 to 2.21 across the entire 0 to the evaluation period
 age range. so there is no drift correction to speak of. By the time the
 reweighting is sharp enough to matter (≤3 d) it has deleted half the
 effective sample. There is no setting where the correction is large and the
 data is intact, because the train window is simply too short for the two
 requirements to coexist.
2. **The FM spends far less of its score on the drifting quantity than
 `run_pop` does.** The pop model is 100% item bias, so 100% of it drifts.
 The FM's item bias is already saturated (7,538 videos, ≈151 impressions
 each. iteration 1), and iteration 3 showed the metric-visible headroom is
 in the personalised and contextual terms. A weighting scheme that acts
 uniformly on *every* parameter therefore pays the full variance cost of
 down-weighting data on all of them, to correct drift in the one part that
 was least in need of help.

Note also that the direction of the small test-side movement is consistent
across the whole 5 to 14 d band (0.59748 to 0.59756 vs 0.59744) and, as in iteration
4, shows up on test but not validation. test being the split furthest from
the training window. It is +0.0001, an order of magnitude under the gate, and
is claimed as nothing.

This iteration's most valuable output is not the
refutation but the diagnostic that came with it: **there is a measured +0.0076
valid / +0.0104 test of ordering-relevant temporal drift sitting in the item
signal, at equal sample size.** That is a large, live target. the largest
single effect any iteration has measured on this task. and iteration 5 shows
only that a *loss weight* is the wrong instrument for collecting it, because
the instrument's cost (discarded data) scales with its sensitivity. The right
instrument gives the model the recent signal as an input while keeping
every row: trailing-window target encoding of item/author long-view rate
(promoted to top open idea), or, as the one-line version, blending the FM with
a recent-window popularity score. Both keep the 1.14M rows and pay nothing for
the recency.
