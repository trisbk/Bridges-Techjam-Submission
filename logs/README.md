# Logs folder guide

| File | What it is |
|---|---|
| ITERATION-LOGS.md | The compiled per-iteration index: hypothesis, code reference, metrics, verdict for every run, plus the strict-rule segmentation and compliance tables |
| INTERVENTIONS.md | The manual interventions summary (count, enumeration, classification) |
| RESULTS_CODE.md | The full research narrative for the main interactive campaign, run by run, with mechanism analysis |
| RESULTS_AGENT.md | The clean-room campaign's experiment log with zero-intervention, agent-only, started from the task statement and the official baseline |
| LOG.jsonl | Machine readable log. Intent records are written before training, results after |
| LOG-replay.jsonl | Validation-only reconstruction of the banking trajectory (see `code/replay_verdicts.py`) as it independently reproduces the shipped champion using validation data alone |
| IDEAS.md | The agent's idea backlog: banked, dead, and open, each with the run that decided it |
| raw_evidence/ | Full transcripts, driver logs, diagnostics, and preserved historical experiment code, organized by campaign and topic — see `raw_evidence/README.md` for the detailed index |
| PROCESS-AUDIT.md | Findings and corrections from a pre-submission adversarial review, stated plainly |
