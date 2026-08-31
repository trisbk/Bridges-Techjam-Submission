# Idea backlog

Statuses: OPEN untested · DEAD tested and rejected ·
BANKED in the current best recipe. Every DEAD entry links the run that
killed it in RESULTS_CODE.md; read the reason before proposing anything similar.

## Banked recipe (test primary 0.5986)
Listwise InfoNCE objective (K=4), mixed ensemble of 5× MLP (k=16, H=64,
lr 3e-4) + 5× FM (k=16, lr 1e-3), predictions z scored per model before
averaging. Reproduce via experiments/ensemble.py.

## Open ideas

1. DEAD (Run 9: 0.5978 = plain FM; interactions already balanced); FwFM weighted by field FM; learn a scalar weight per field *pair*, so
 e.g. user×video interactions can matter more than tab×dur_bucket. Small
 param count, direct expressiveness gain over FM's uniform interactions.
2. BANKED (Run 10: 0.5984 single, best model; k=16/H=64 optimal per Run 12); MLP head over embeddings; concat field embeddings → 1 hidden layer
 for example 64, ReLU → score. First nonlinear model in the series; needs its own
 lr search. Highest ceiling, highest variance risk.
3. DEAD (Run 13a: no gain; with Run 2, pointwise adds nothing in any mixture); Objective annealing; start pointwise with a fast, calibrated start, switch
 to listwise for late epochs. Cheap to test; small expected gain.
4. DEAD (Run 13b: −0.0101; top scored negatives are near positives, over sampling them suppresses taste signal); Hard negative curriculum; sample negatives proportional to current
 model score using the model's own scores rather than uniformly. Medium effort;
 interacts with InfoNCE's existing hard negative weighting, may double dip
 or may be redundant.
5. UNPARKED, IN FLIGHT (Run 18); Causal sequence features; everything so
 far is static; use the user's strictly prior behavior: last impression
 label, rolling long_view rate over last 10, prior impression count, prior
 long_views with this author. Sorted chronologically; a row's features
 never include its own or any later label. The production recsys #1 lever,
 untried here.
6. UNPARKED, QUEUED (Run 19); Multi task auxiliary labels; train log also records is_click /
 is_like / is_comment / is_hate; predict them jointly with shared embeddings
 (long_view stays the graded head). Better representations from data we
 already legally hold.
7. RETIRED UNUSED (legality analysis,): every row of log_random falls inside the valid/test window; training on it would be temporal leakage.; Random exposure log; kit ships log_random with videos
 shown at random; a small unbiased sample usable for debiasing the
 algorithmically biased main log.
8. DEAD (Run 16a: user×tab cross 0.5973, sparse cross overfits); Per tab models or tab conditioned interactions; the log spans distinct
 surfaces across tabs; one global model may blur tab specific taste. Check tab
 cardinality/skew first.
9. DEAD (Run 16b/c: τ=1 already optimal; sharpened and softened both worse); nDCG targeted top slot loss; extra weight on getting the single best
 item right (nDCG@5's log discount makes slot 1 worth most). Risk: fights
 GAUC half of the metric; mind the Run-1 lesson.

## DEAD (with the reason) 

- Row reweighting by 1/impressions (Run 1): GAUC is positive count weighted;
  helping nDCG's equal user view hurts GAUC more.
- Dense play ratio target (Run 1): duration mediated → learns "prefer short
  videos". (Duration normalised variant technically still open but
  low priority.)
- BPR lr/pair count scaling (Run 3): gain capped by objective, not hyperparams;
  lr ≥ .002 destructive.
- Embedding capacity k=32 (Run 4): does nothing under either objective.
- Categorical side features ;  hour/content/uactive/vpop (Runs 5-6): +0.0003
  noise level, redundant with each other; do NOT stack.
- Explicit user×author / user×tag affinity rates (Runs 6-7): naive encoding
  self leaks (−0.012); LOO fixed version is redundant with FM factorization.
- Bigger K in InfoNCE (Run 4): K=8 ≈ K=4, saturated.
- Seed ensembling beyond 5 (Run 8): +0.0004 total; FM seeds converge to
  near identical solutions. Already banked; more seeds won't add.
- Play ratio as an AUXILIARY head over shared FM embeddings (Run 31): a dense
  graded engagement target, correlated 0.79 with long_view, still costs
  ~0.0011 validation whether it is duration normalised or not and at either
  aux weight. Run 1 blamed duration mediation; Run 31 shows that was not the
  binding problem ;  the ratio is a *different ranking* of the same rows
  (watch completion, not within user preference), and any gradient that pulls
  the shared embeddings toward it pulls them off the GAUC/nDCG objective. Do
  not retry with a third normalisation.
- Per field embedding sizes / interaction rank (Run 32): k in an FM is the RANK
  of every interaction a field joins (pair j,l interacts at rank min(k_j,k_l)),
  not a parameter budget ;  the nine narrow fields are 71 of V's 40,313 rows, so
  shrinking them frees 0.1% of parameters. Cutting the NARROW fields to rank 8
  costs −0.0053 to −0.0062 on validation at either wide rank; widening the wide
  fields to 24 does nothing at either narrow rank (−0.0006 to −0.0015). The
  binding constraint is the narrow fields, because their rank caps
  prev1 x user_id (a 3 x 26,210 surface) ;  the causal sequence fields earn their
  Run 21 gain by MODULATING the user/video embeddings, so low cardinality is not
  low expressive load. Do not retry any variant that gives a sequence/context
  field less rank than the wide fields. Also: uniform k=24 = k=16 −0.0006, which
  reconfirms Run 4's capacity saturation under the FM rich premise.

- Side/content fields, re tested under the FM rich premise (Run 30): hour and
  video_type/music_type/tag are worth +0.0003 on validation under the FM on
  rich causal features ;  the *same* +0.0003 Run 5 measured under the MLP on
  base fields. Two model classes and two feature regimes apart, the number is
  unchanged: these fields carry ~no incremental ranking signal here, and the
  FM's pairwise factorization does not rescue them.

## Standing rules

- ≥3 seeds; claim nothing under ±0.002 (noise σ≈0.0008).
- Selection uses validation; test recorded for audit, never tuned on.
- Any per entity rate feature over sparse keys → leave one out encoding
  (Run 6's lesson).
- Log intent BEFORE training (survives kills), result after.

## Remaining open ideas

These are genuinely untested combinations. Each earlier test was under different
premises, noted per item.

10. DEAD (Run 30a: valid −0.00004 vs a same run control; the premise change
 does not revive it); hour field under FM rich; Run 5 tested hour
 under the MLP on base fields only; never under the FM on rich fields.
11. DEAD (Run 30b: valid −0.00003 vs control; Run 30c stacking both = +0.00031,
 still 6× under the gate); content fields (video_type/music_type/tag)
 under FM rich; same premise gap as #10.
12. DEAD (Run 31: all three aux arms BELOW a same run control on validation,
 −0.0011 to −0.0013; λ=0.3 and λ=1.0 alike, and the normalized by duration and
 duration mediated targets are indistinguishable from each other); normalized by duration play ratio as auxiliary signal; Run 1 killed the
 naive ratio TARGET but explicitly left the normalized by duration variant
 open; never revisited after the objective/feature revolutions.
13. K=2 under FM rich; K saturated upward (4≈8) on base fields; the
 cheap downward direction was never measured on rich fields.
14. DEAD (Run 32: the proposed wide24/narrow8 config is the WORST of five
 arms, valid −0.00617 vs a same run control; narrow8 costs −0.0053 even at
 unchanged wide rank, 2.6x the gate); per field embedding sizes; video/user k=24, low cardinality fields k=8; parameter budget shifted
 toward where cardinality lives. Never tested in any premise.
