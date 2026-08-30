# Written Project Description (for Devpost)

## How our solution addresses the problem statement

Track 2 asks for an autonomous ML research agent that improves a recommender
model on the KuaiRand-Pure within-user ranking task. The label is long_view,
the primary metric is the mean of GAUC and nDCG@5, and the official baseline
scores 0.5946.

We built that agent. It runs the loop a TikTok ML engineer runs daily:
propose a hypothesis, implement it in code, train, evaluate, reflect, and
decide what to try next. It iterates until the official convergence rule
fires (no validation improvement above 0.002 for 3 consecutive iterations).

Result: test primary **0.6116** (GAUC 0.6825, nDCG@5 0.5408), which is
**+0.0170 over the official baseline** (GAUC +0.0215, nDCG@5 +0.0126, and
the score under the official formula is the mean of those deltas). The
record behind it: 32 logged runs covering about 70 configurations, plus a
separate clean-room run where the agent, restarted with zero prior knowledge
and zero human input, independently reached 0.59744 (+0.0028 over baseline)
in 6 iterations and 1 hour 48 minutes.

The two discoveries that carried the score, in the order the agent found
them:

1. Match the training objective to the metric. The baseline trains pointwise
   but is graded on within-user ranking. A listwise objective (softmax over
   one positive and four within-user negatives) gave +0.0028, with a
   capacity matched control proving the gain came from the objective alone.
2. Causal sequence features. Seven features computed strictly from each
   impression's past: previous impression outcome, rolling watch rates, per
   author and per tag history, session gap. Never the row's own label, never
   anything later in time. This was the largest single gain (+0.013) and it
   explained an earlier plateau: the models had been information starved,
   not under powered.

The final model is deliberately simple: a five seed committee of
Factorization Machines (k=16), reproducible from raw data in one command in
about 5 minutes on a laptop CPU.

Three properties of the process matter as much as the score:

- Enforced discipline. Every experiment runs through a harness that requires
  at least 3 seeds, a pre-committed 0.002 significance bar, and logging of
  intent before training, so failed runs cannot be hidden. All selection
  uses validation only. The logs document five separate occasions where a
  configuration with a better looking test score was refused because
  validation did not justify it. Two of those refusals were made by the
  unattended agent with nobody watching.
- Honest negatives. The log contains more rejected ideas than accepted ones,
  each with a diagnosed mechanism, including a target-encoding leakage trap
  the agent caught and fixed, and a legality analysis that retired the
  dataset's random exposure file unused because its rows overlap the
  evaluation window.
- Autonomy with a defined boundary. A driver loops fresh agent sessions to
  convergence under the official rule, restarting crashes automatically
  (the organizers ruled restarts are not interventions). The interactive
  campaign needed 3 strategic human decisions in total, enumerated in
  `logs/INTERVENTIONS.md`. The two unattended campaigns needed zero.

## Development tools used

- Claude Code (Anthropic): the autonomous agent itself. It wrote the
  experiment code, ran the research loop, and authored the run logs and the
  experimental commits, which are visible in the git history.
- Git and GitHub for version control. The commit trail doubles as a
  timestamped record of the agent's iterations.
- macOS terminal, zsh, and caffeinate for unattended runs on a laptop.
- Python 3.12 (CPython).

## APIs used

- Anthropic Claude API, through the Claude Code CLI. This powers the agent's
  reasoning and code generation. No other external APIs. Training and
  evaluation are fully local and offline.

## Libraries and frameworks used

- NumPy. The only numerical dependency. All models and training loops are
  implemented from scratch in NumPy. No ML framework.
- Python standard library: csv, json, collections, subprocess, os, time.

## Datasets and assets used

- KuaiRand-Pure (official Track 2 dataset, Kuaishou research release):
  1.14M train, 125K validation, 171K test logged impressions, used only
  through the official date split.
- The official Track 2 starter kit. Its scoring code and split are used byte
  identical and unmodified.
- The dataset's random exposure file was deliberately not used. Its rows
  fall inside the evaluation window, so training on it would leak future
  information. The analysis is recorded in the logs.
- No other datasets, no pretrained models, no manually labelled data.
