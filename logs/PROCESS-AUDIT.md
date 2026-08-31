# Process audit

This audit is organized around the seven stages the project moved
through. The aim is to show how the research process developed, what the
review found, what was corrected, and which parts of the result remained
valid after those corrections.

The score itself was not the issue raised by the review. The earlier
frozen checkpoint was independently recomputed at 0.61164 with the
official evaluator. The main questions were about model selection, the
harness, autonomous iteration, and whether the written process
accurately matched what the code did.

## Stage 1. Harness and selection rules

The first major correction was in `harness.py`. The harness originally
derived its printed result label and banked-best tracking from the test
mean even though the intended research process selected models using
validation. The driver also inherited a test-derived convergence
constant.

This created a mismatch between the automated label and the decisions
that were actually made. The experiment trail shows that banking
decisions were based on validation. `LOG.jsonl` carries `valid_mean`,
and several test-favorable alternatives were refused because their
validation results did not justify replacing the incumbent.

The harness was corrected so the result label, banked-best tracking, and
printed marker use `BASELINE_VALID` and `BANKED_VALID`. Test scores are
still recorded for evaluation and audit, but they do not determine
promotion. `driver.py` now reads the validation constant as well, and
the iteration prompt follows the same rule.

The promotion rule was also made explicit. A challenger has to beat the
incumbent's validation score by more than `PROMOTION_MARGIN = 0.001`.
Ties and differences inside that margin keep the incumbent. This matches
the selection discipline used when test-favorable alternatives were
rejected.

Historical `LOG.jsonl` labels were kept rather than rewritten. They
should be treated as old display markers, with the validation fields and
the reconstructed trajectory serving as the authoritative selection
record. `code/replay_verdicts.py` now replays the historical decisions
using validation means alone and writes a separate companion log instead
of modifying the original history.

The verification layer was tightened at the same time.
`code/verify_claims.py` re-derives the oracle ceiling, user composition,
split sizes, starter-kit integrity hashes, and seed-noise summaries from
shipped artifacts. The official starter-kit archive is included under
`third_party/`, allowing those checks to be made against the actual
files.

The important distinction after these corrections is simple: test was
visible throughout the project, but model selection was based on
validation.

## Stage 2. From the harness into exploration

With the experiment machinery in place, the project moved into broad
exploration. This stage tested objectives, model classes, feature
families, capacity, ensembling, and other alternatives rather than
committing early to one explanation.

The audit clarified how this stage should be described. The interactive
research phase was supervised work that was later mapped onto the run
formalism. Its strict-rule segmentation is therefore retrospective. It
should not be read as evidence that every early experiment originally
ran under the autonomous protocol developed later.

Several smaller corrections belong here. Run 25b used a cross-view blend
whose selected alpha was 1.0, making it identical to Run 24b. It is
therefore a search that returned to the incumbent rather than an
independent result. The experiment counts were also reconciled instead
of being carried forward from older summaries.

The review challenged whether some evidence used in the exploration
narrative actually existed. Those challenges were checked against the
repository. The Run 15 alternatives have their validation means in
`LOG.jsonl`, and the final committee's individual seed scores are
preserved and summarized by the verification script.

The exploration record remains useful, but the corrected version
separates two things more carefully: what each experiment demonstrated,
and the later process framework used to organize the overall research
trail.

## Stage 3. From exploration to the frozen model

Exploration eventually converged on the causal sequence-feature recipe
and the FM committee. At that point the model was frozen instead of
treating every observed test increase as a reason to switch.

The earlier frozen checkpoint reached 0.6116 on the local test split, an
improvement of 0.0170 over the published baseline. It could be
reproduced from raw data using the shipped pipeline, and the research
trail contained substantially more failed or neutral ideas than
successful ones.

The audit then examined an important assumption behind the
sequence-feature gain. The causal history features update continuously,
so a row can use realized outcomes from strictly earlier impressions
belonging to the same user. The current row never sees its own label or
future information, but the setup assumes that recent interaction state
can be refreshed during serving.

`code/staleness_ablation.py` was added to measure that dependence while
keeping the frozen weights unchanged.

| Test-time feature regime | primary | vs baseline |
|---|---|---|
| Continuous updates | 0.61164 | +0.0170 |
| Daily batch refresh | 0.60828 | +0.0137 |
| Frozen at the test boundary | 0.59429 | -0.0003 |

Most of the gain survives under daily refresh. The fully frozen number
is treated as a lower bound rather than the expected performance of a
model designed for that regime because fields such as `gap` and `hist_n`
become distribution-shifted.

A committee retrained specifically for daily-batch features reached
0.6106, or +0.0160 over baseline. The model was also evaluated on the
random-exposure log to test whether the advantage survived without the
same exposure bias. A stricter seed-matched comparison reduced the
reported advantage, and the smaller comparison was retained rather than
the larger initial estimate.

The mechanism behind the sequence gain was challenged as well. The
falsification machinery in `code/controls.py` generated placebo tests
from mechanism tags. Time-shuffling the sequence information removed
almost all of the gain, narrowing the claim from "sequence features
help" to the more specific conclusion that their timing carries the
important signal.

These analyses tested the assumptions around the frozen checkpoint
without replacing it.

## Stage 4. Autonomous clean-room phase

The clean-room phase tested whether an autonomous agent could continue
the research from the established state without relying on the
interactive process that created it.

This phase needs a careful distinction between autonomous activity and
completed experiments. Two clean-room iterations ended while an
experiment grid was still training. Under the official convergence rule,
iterations without improvement still count toward the below-epsilon
streak, so the recorded convergence stands under that definition. Under
a stricter interpretation that counts only iterations producing an
adjudicated experiment, another iteration would have been needed.

Both readings are kept because the difference concerns process
accounting, not the measured score.

The verification runs after the original freeze are treated similarly.
Runs 30 to 32 started from the frozen state and banked nothing. Their
value is as a non-regression check showing that the converged state
survived another round of challenges, not as evidence of autonomous
improvement.

The clean-room transcripts are included so the iteration record can be
checked directly instead of reconstructed from summary prose.

This phase also exposed what the autonomous system still lacked. That
led to a more structured research framework built around a belief state,
residual-driven hypothesis generation, mechanism-tagged controls,
cost-aware experiment ordering, and promotion rules enforced in code.
Those pieces became the basis of the v2 loop.

## Stage 5. Residual-driven v2 loop

The v2 loop changed how the autonomous system chose what to test. Rather
than working only from a static backlog, it used residual analysis to
identify weaknesses in the current champion, stored hypotheses in a
structured belief state, attached controls based on the claimed
mechanism, and used the validation promotion rule.

Its first live iteration surfaced `tab=0` as a major residual weakness.
During that investigation, the agent found that the analyzer's
expected-value measure was misleading when a slice contained many users
with no positive label inside that slice. The analyzer was rewritten to
use oracle headroom and a self-test was added.

The agent then wrote `code/tab_surface.py` and ran a control, two
candidate arms, and a time-shuffle placebo.

| Arm | validation |
|---|---|
| R33 control | 0.61715 |
| R33a | 0.61806 |
| R33b | 0.61955 |
| Time-shuffle placebo | 0.61612 |

The placebo was useful because the apparent improvement disappeared when
the temporal attachment was broken. R33b was still not promoted. Its
gain over the banked validation score of 0.61906 was only +0.00049,
below the required 0.001 margin.

The iteration also exposed two process problems. The agent session
completed, but the driver wrapper failed before recording the
iteration-end event. In addition, the session recorded its evidence and
control through the harness without writing them back into
`belief_state.json` through `attach_evidence` and `attach_control`.
The code-enforced promotion gate existed and passed its self-test, but
it was not exercised live during this iteration.

The state was preserved as the agent left it rather than cleaned up
afterward. This stage is therefore best understood as one completed
v2-loop iteration that produced a promising but sub-margin result and
revealed weaknesses in the loop itself, not as a converged campaign.

## Stage 6. Campaign 5 and checkpoint promotion

Campaign 5 resolved the unfinished question from the v2 iteration:
whether the R33 idea would survive the stronger committee-level check
already written into the experiment rule.

The earlier 5-seed committee check could not be verified because it had
run outside the harness and its output was lost with the driver failure.
Instead of changing the rule, the missing step was rerun through the
harness using `tab_committee_check.py`.

That produced R33c with validation 0.62059. The incumbent was 0.61906
and the required margin was 0.001, so the committee cleared the
pre-committed promotion rule.

This is why the earlier 3-seed result was correctly declined while the
later committee result was correctly promoted. The evidence changed, but
the rule did not.

Campaign 5 then carried the promoted checkpoint through convergence. R34
was -0.00355, R35 was -0.00170, and R36 was -0.00070 relative to the
relevant comparison, producing the required below-epsilon streak.

The checkpoint was rebuilt from scratch with `final_model.py`, and the
submission was regenerated with independent alignment checks in
`make_final_submission.py`. The belief state was brought up to date
through its own API with `close_the_loop.py`, which meant the
control-gated promotion path was finally exercised live.

Every selection decision in this chain used validation. Test scores were
recorded but did not decide the promotion.

The promoted champion added the label-free `tab_n` count feature to the
previous recipe and reached validation 0.62059 with test primary
0.61429.

A later mechanism control refined the explanation for why `tab_n`
helped. Scrambling surface labels while preserving the chronological
counting structure left roughly half of the measured gain. This weakened
the original pure surface-familiarity explanation and supported the
broader idea of partitioned familiarity: some of the effect depends on
the true surface, while some comes from the additional counting
structure. Because the measurement is close to the noise scale, this is
treated as an approximate split rather than an exact decomposition.

The mechanism explanation changed, but the promotion did not. The
checkpoint had been selected because it cleared the validation rule, not
because a particular explanation of the feature had to be true.

## Stage 7. Campaign 6 and the final autonomous check

Campaign 6 challenged the promoted champion using the repaired
autonomous loop. The driver had been hardened so session output was
written to files and timed-out process groups could be terminated
cleanly. No human input was provided after launch.

Nothing in the campaign replaced the champion. Validation remained
0.62059 and test primary remained 0.61429. The useful outputs were
failed hypotheses and repairs to the research machinery.

The first iteration tested session depth on the dominant `tab=1` slice.
Train-only diagnostics made the feature look plausible, but both
experimental arms were worse than the in-session control. The stronger
arm fell by 0.00347, making it a meaningful negative rather than an
ambiguous null.

The post-mortem showed why simple pre-run feature statistics were not
enough. Session position contained novel within-user signal and was not
fully redundant with the existing fields, yet adding it still hurt the
FM. A feature can therefore look informative in isolation while damaging
the shared embedding geometry once it participates in the model's
pairwise interactions.

The next experiment tested exposure counts for tag and author
partitions. Again, the descriptive statistics looked strong. At a fixed
number of prior successes, more prior exposures corresponded to a lower
hit rate, suggesting that the missing denominator carried useful
information. Even so, `tag_n` and `auth_n` did not beat the control.

This gave the project a second failure of the same cheap screening idea.
`tag_n` actually carried more measured novel within-user prior variance
than the successful `tab_n`, but it still lost when trained. Marginal
novelty was therefore rejected as a reliable predictor of whether a new
FM field would improve ranking.

More importantly, the failed experiment exposed a flaw in the loop's own
residual-ranking instrument. The expected-value queue was favoring large
slices because oracle headroom naturally grows with slice size. That
could push a large but well-handled slice ahead of a smaller slice where
the model was genuinely weak.

The agent replaced the measure with EVx, a matched-null calculation that
preserves how many rows each user contributes while randomizing which of
that user's rows belong to the slice. This subtracts much of the
headroom created mechanically by slice size and user composition. A
regression self-test was added for the failure that motivated the
repair, and an earlier null design that behaved badly was documented
rather than hidden.

One procedural qualification remains. An iteration whose experiment was
still training counted toward the below-epsilon streak, and the
following iteration adjudicated that same experiment rather than
launching a duplicate. Under the official rule, an iteration without
improvement counts, so convergence stands. Under a stricter
result-bearing interpretation, another completed iteration would have
been required.

That qualification does not change the selected model. Campaign 6 banked
nothing, and replaying the trajectory left `BANKED_VALID` unchanged.

## Final audit position

Across these seven stages, the corrections changed how the process
should be described, but they did not require selecting a different
final checkpoint.

The harness was repaired so its automated decisions match the validation
rule. The exploration record was separated from the autonomous formalism
developed later. The frozen model's serving assumptions and mechanism
claims were measured rather than left implicit. The clean-room phase was
reported with its incomplete iterations intact. The v2 loop exposed both
a promising feature and weaknesses in its own infrastructure. Campaign 5
completed the missing evidence step and promoted the new champion only
after it cleared the written validation margin. Campaign 6 challenged
that champion, found no justified replacement, and improved the loop's
own hypothesis-ranking machinery.

The final selected checkpoint remains validation 0.62059 and test
primary 0.61429. The audit does not claim that the process was flawless
from the beginning. It shows that the important gaps were made visible,
checked against the shipped evidence, corrected where necessary, and
prevented from changing model selection unless validation supported the
change.
