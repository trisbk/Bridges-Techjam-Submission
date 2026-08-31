# Written Project Description

## Project description

Track 2 asks teams to build an autonomous machine-learning research agent that improves a recommender system on the KuaiRand benchmark. Our project gives the agent a research process rather than a fixed list of model changes to try.

Think about it this way. The agent starts with the best model it currently has and looks for places where that model still makes large mistakes. It then proposes a possible reason, writes and runs an experiment, checks the result on validation data, and decides whether the change deserves to be kept. Failed ideas are recorded as well.

The final recommendation model uses signals about what a user has been doing recently. A standard recommender can learn that a person generally prefers certain videos, creators, or topics. Recent behavior adds another piece of information: what appears to interest that person at the current point in time.

All of these history features use information available before the recommendation being scored. They include recent interaction outcomes, recent author and tag behavior, timing information, and prior experience with the current recommendation area.

On the required KuaiRand-Pure benchmark, the official baseline has a primary score of **0.5946**. Our final model scores **0.6143**, an absolute improvement of **+0.0197**. Its test components are **GAUC 0.6857** and **nDCG@5 0.5429**.

We also tested the finished recipe on KuaiRand-1K without changing its model settings specifically for that benchmark. The reproduced baseline scores **0.6293**, while our model scores **0.6931**, a difference of **+0.0637**. KuaiRand-1K provides much more interaction history per user, so the larger improvement is consistent with our hypothesis that recent behavior becomes more useful when more history is available. This transfer result is evidence of that pattern, not proof that history depth alone caused the entire difference.

A separate control experiment tested the explanation more directly. We kept the behavioral information but attached it to the wrong moments in each user's history. About **95% of the measured sequence-feature gain disappeared**. The incorrectly timed features also performed almost exactly like matched random information. That result supports the claim that the timing of recent behavior carries useful signal.

One final feature, `tab_n`, came from the autonomous research loop. The agent found a weak validation slice and proposed measuring how much prior experience a user had with the current recommendation area. The feature improved the final model. A later control showed that the agent's first explanation was only partly correct, so the interpretation was narrowed while the validated feature was retained.

## Development tools used

- **Claude Code (Anthropic):** used as the reasoning component in the autonomous research workflow. The agent works with measured model errors and structured hypotheses, then interacts with the controlled experiment pipeline under predefined validation and falsification rules.
- **Git and GitHub:** used for version control, the public code repository, and the research record.
- **macOS Terminal and zsh:** used to run training, evaluation, and unattended research sessions.
- **Python 3.12 / CPython:** used for model training, evaluation, experiment scripts, and agent utilities.

## APIs used

- **Anthropic Claude API through Claude Code:** provides the language-model reasoning used by the autonomous research agent.
- Model training and evaluation run locally and do not require an external API.

## Libraries and frameworks used

- **NumPy:** numerical operations and model training.
- **Python standard library:** file handling, CSV and JSON processing, subprocess management, timing, collections, and other supporting tasks.
- No deep-learning framework is required.
- No GPU is required.

## Datasets and assets used

- **KuaiRand-Pure:** required Track 2 benchmark.
- **KuaiRand-1K:** optional transfer benchmark.
- **Official Track 2 starter kit:** split definitions, baseline code, and evaluation tools.
- **KuaiRand random-exposure data:** used for post-freeze evaluation of exposure bias, not for selecting the final competition model.

No manually labeled dataset or pretrained recommendation model was added.

## Public repository

https://github.com/trisbk/Bridges-Techjam-Submission

