# Historical / Post-Hoc Analysis Scripts

The following scripts evaluate the test split directly and were run
before any structural test-isolation work: `staleness_ablation.py`,
`protocolB_retrain.py`, `daily_retrain.py`, `debias_frontier.py`,
`mechanism_test.py`, `unbiased_eval.py`, `bonus_1k.py`. Their outputs
are already recorded (see logs/*.out and RESULTS-SUMMARY.md) and are
not meant to be re-executed. No live research module imports or calls
any test-evaluation function from these scripts; `residual_analysis.py`
reuses `staleness_ablation.py`'s `build_features`/`RICH` feature-
construction helpers for convenience, which involve no test labels.
