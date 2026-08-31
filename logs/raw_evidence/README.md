# Raw Evidence

This directory contains the original logs, transcripts, diagnostics, and
experiment outputs supporting the summarized results in `../RESULTS.md`.

These files are preserved for auditability and reproducibility; judges do not
need to read them in order to understand the project.

## Autonomous campaigns

- `cleanroom/` — zero-intervention clean-room campaign
- `demoA_transcripts/` + `demoA_driver_log.jsonl` — unattended agent sessions
- `campaign4_driver_log.jsonl` — autonomous campaign driver history
- `campaign6_driver.out` — Campaign 6 autonomous research and self-correction

## Final model and promotion

- `final_model_r33c.out` — final champion reproduction
- `tab_committee_check.out` — five-seed `tab_n` promotion check
- `tab_mechanism_control.out` — follow-up mechanism control

## Diagnostics and negative results

- `familiarity_diagnostic.out`
- `session_depth.out`
- `session_depth_diagnostic.out`
- `partition_familiarity.out`
- `partition_postmortem.out`
- `residual_analysis_v3.out`

## Robustness / alternative regimes

- `protocolB_retrain.out` — conservative no-test-window-feedback retraining
- `bonus_1k.out` — KuaiRand-1K transfer result

## Historical experiment code

- `experiment_scripts/` — scripts used for historical experiments