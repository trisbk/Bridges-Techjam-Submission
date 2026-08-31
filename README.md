# An Autonomous Research Agent for Better Recommendations

**TikTok TechJam 2026, Track 2**

## What we built

We built an AI research agent that can investigate and improve a recommendation model.

The basic process is simple. The agent finds where the current model is making its largest mistakes, proposes a possible explanation, runs an experiment, and checks whether the result improves on validation data. It keeps useful changes and records failed ones.

The agent can also test its own explanations. When it claims that a feature works for a particular reason, we can run a control that deliberately removes that reason while keeping other information as similar as possible. This matters because a higher score does not automatically mean the explanation behind it is correct.

The research process produced our final recommendation model.

## Main result

On the required **KuaiRand-Pure** benchmark:

| Metric | Official baseline | Our model | Improvement |
|---|---:|---:|---:|
| GAUC | 0.6610 | **0.6857** | **+0.0247** |
| nDCG@5 | 0.5282 | **0.5429** | **+0.0147** |
| **Primary score** | **0.5946** | **0.6143** | **+0.0197** |

The final model improves the primary score from:

## **0.5946 to 0.6143**

The submitted file contains **170,588 test predictions** in the format required by the starter kit.

## What changed in the recommendation model?

A recommender usually needs to learn what a user tends to like. Our model also pays attention to what that user has been doing recently.

Consider a user who usually watches one type of content but spends the current session interacting with something different. Long-term preferences still matter, but recent behavior may give a better picture of what the user wants at that moment.

Our model therefore keeps a small amount of recent history. The signals include previous interaction outcomes, recent behavior around authors and tags, timing information, and how much prior experience the user has with the current recommendation area.

Every history feature is built from information that existed **before** the recommendation being scored. Future outcomes are not used to construct earlier predictions.

## How do we know the timing matters?

We tried to break our own explanation.

Our hypothesis was that recent behavior helps because the model sees what happened at the correct point in a user's history. To test that idea, we shuffled the behavioral features within each user's history.

The model still received the same general type of information about the same users. What changed was the timing. A feature describing one moment could now be attached to another.

| Experiment | Test primary |
|---|---:|
| **Behavior attached to the correct moment** | **0.61164** |
| Remove sequence features | 0.59808 |
| Attach the features to the wrong moments | 0.59872 |
| Replace them with matched random information | 0.59876 |

About **95% of the measured sequence-feature gain disappeared when the timing was broken**.

The shuffled features also scored almost exactly the same as matched random information. That comparison supports a narrower conclusion: most of the measured benefit comes from correctly timed recent behavior, rather than simply giving the model more user-related inputs.

## The agent found a feature that entered the final model

One of the final improvements, called `tab_n`, came from the autonomous research loop.

The agent examined the current champion's validation errors and found a recommendation area where performance was weak. It proposed measuring how much prior experience a user had with the area currently being viewed.

That feature helped move the model from the previous champion score of about **0.6116** to the final **0.6143**.

The first explanation was not completely right. A later control suggested that part of the gain came from genuine familiarity with the recommendation area, while part came from the feature's broader counting structure. We therefore kept the feature because it passed the validation rule, but narrowed the claim about why it worked.

That distinction is part of the research design. A useful result can survive even when its first explanation needs revision.

## What happens on a much larger dataset?

We transferred the finished recipe to **KuaiRand-1K** without changing its model settings specifically for that dataset.

| KuaiRand-1K | Test primary |
|---|---:|
| Reproduced starter-kit baseline | 0.6293 |
| **Our frozen recipe** | **0.6931** |
| **Improvement** | **+0.0637** |

The improvement is more than three times the margin measured on KuaiRand-Pure.

KuaiRand-Pure contains about **53 logged interactions per user** on average. KuaiRand-1K contains roughly **11,700**. Our model uses recent behavioral history, so the larger dataset gives those features much more information to work with.

The result is consistent with the project's main hypothesis: the behavioral approach becomes more useful when deeper user histories are available. Other differences between the datasets may also contribute, so we treat the transfer result as supporting evidence rather than a controlled causal test.

## How the AI researcher works

The research loop follows this process:

```text
Find where the current model is weak
              ↓
Propose a possible reason
              ↓
Choose an experiment
              ↓
Train and evaluate it
              ↓
Test the explanation when needed
              ↓
Keep or reject the change
              ↓
Record what was learned
```

Several small modules support that process.

### Finding mistakes

`agent/residual_analysis.py` looks at validation errors from the current best model. It helps the agent start with an observed weakness instead of proposing model changes without evidence.

### Remembering what has been tested

`agent/belief_state.py` stores hypotheses and the evidence for or against them. For claims about *why* something works, the code can require a suitable control before treating the explanation as confirmed.

### Choosing what to test

`agent/priority.py` compares candidate experiments using their expected value and measured running cost. The purpose is practical: limited research time should go to experiments that appear useful enough to justify their cost.

### Running experiments consistently

`code/harness.py` gives experiments a common structure. The intended change is logged before training, and results are recorded afterwards.

## How reliable is the model selection?

We choose model changes using validation performance rather than whichever configuration happens to score best on the final test set.

The clean-room autonomous run provides a useful example. During that run, the agent encountered configurations with attractive test results but weaker validation evidence. It rejected them.

This rule reduces the risk of choosing a model because of a lucky test result. It does not remove all uncertainty, which is why the project also uses several random seeds and a fixed improvement threshold for promotion.

## Would the behavioral features work without instant updates?

The main behavioral features are easiest to understand as a small running memory for each user. In a live system, that memory could be updated after each interaction.

We also tested less frequent updates.

| History update method | Test primary | Improvement over baseline |
|---|---:|---:|
| Continuous updates | 0.6116 | +0.0170 |
| Daily refresh, retrained | 0.6106 | +0.0160 |
| Frozen history, retrained | 0.5979 | +0.0033 |

A model trained for daily updates retains about **94% of the measured continuous-update gain**.

Even when user history is frozen throughout the test period, a model trained under the same restriction remains above the official baseline. The size of the gain is much smaller in that setting.

## Are we only learning what the old recommender showed users?

Logged recommendation data has a basic limitation. Users can react only to videos they were shown, and those videos were selected by an earlier recommendation system.

We therefore tested the model on **897,721 randomly exposed impressions**, where exposure is less dependent on the previous recommender's choices.

| Random-exposure evaluation | Primary score |
|---|---:|
| Baseline committee | 0.3707 |
| **Our model** | **0.3777** |
| **Improvement** | **+0.0070** |

The absolute scores are lower in this evaluation because positive interactions are rarer. Still, the model remains ahead of the matched baseline.

This result suggests that the measured advantage is not entirely explained by the old recommender's exposure choices. The test uses the pre-promotion R24b model and continuous history updates, so its scope is narrower than the final R33c headline result.

## A second look at exposure bias

We also changed training so that videos shown very often in the historical data received less influence.

| Exposure correction | Standard logged test | Random-exposure test |
|---|---:|---:|
| None | **0.6009** | 0.3785 |
| Mild | 0.5840 | 0.4122 |
| Strong | 0.5654 | **0.4181** |

Correcting for historical exposure improves the random-exposure score but reduces the ordinary competition score.

The catch is that these two evaluations reward somewhat different behavior. The competition measures ranking on the logged test set, while the random-exposure test reduces the influence of what an earlier recommender chose to show.

We therefore keep the uncorrected version for the competition submission. The experiment is included because it makes the tradeoff visible rather than assuming that one score represents every recommendation objective.

## Setup

### Requirements

- Python 3.10 or newer
- NumPy
- No GPU required

Install NumPy:

```bash
pip install numpy
```

Download and extract KuaiRand-Pure:

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzvf KuaiRand-Pure.tar.gz -C code/
```

The extracted data should be available under:

```text
code/KuaiRand-Pure/data/
```

## Reproduce the results

Run model commands from `code/`:

```bash
cd code
```

### Reproduce the starter-kit baseline

```bash
python3 baseline.py --model fm
```

Expected test primary is approximately:

```text
0.595
```

### Retrain the final model

```bash
python3 final_model.py
```

Expected test primary:

```text
0.6143
```

Training takes about five minutes on CPU.

### Score the shipped model without retraining

```bash
python3 score_frozen.py
```

### Check the final submission

```bash
python3 submit.py --check --split test ../final_output/submission.csv
```

### Verify reported constants and dataset checks

```bash
python3 verify_claims.py
```

### Run the KuaiRand-1K transfer experiment

```bash
python3 bonus_1k.py
```

Expected primary improvement over the reproduced baseline:

```text
+0.0637
```

### Run the autonomous research loop

From the repository root, with the research agent environment configured:

```bash
python3 agent/driver.py
```

## Final model files

The repository includes:

- `final_output/submission.csv`: the final **170,588 test predictions**.
- `final_output/frozen_model/`: the final R33c model and five seed weight files.
- `final_output/frozen_model_r24b/`: the earlier 0.6116 champion used for several post-freeze analyses.
- `final_output/RESULTS-SUMMARY.md`: the detailed numerical record.

## Run and iteration logs

The research record is under `logs/`.

Important files include:

- `logs/ITERATION-LOGS.md`: per-iteration hypotheses, code references, metrics, verdicts, and convergence information.
- `logs/INTERVENTIONS.md`: the manual-intervention record.
- `logs/LOG.jsonl`: machine-readable experiment intents and results.
- `logs/RESULTS.md`: the longer research record.
- `logs/PROCESS-AUDIT.md`: documented review findings and corrections.
- `logs/cleanroom/`: the zero-intervention clean-room campaign.
- `logs/experiment_scripts/`: preserved experiment scripts.

## Computing cost

No GPU was used.

**GPU-hours: 0**

Training runs on a single laptop CPU with NumPy. A typical three-seed experiment takes about **40 to 90 seconds** of model training. Retraining the final KuaiRand-Pure model takes about **five minutes**.

The unattended verification run used **3 iterations** and took **52 minutes 32 seconds**. The clean-room run used **6 iterations** and took **1 hour 47 minutes 41 seconds**. Detailed resource records for the remaining campaigns are included in `final_output/RESULTS-SUMMARY.md`.

## Limitations

The main evaluation is offline. It ranks impressions already present in the KuaiRand logs rather than retrieving videos from a complete live catalog.

The dataset also provides limited content information. Once several model types receive the same behavioral features, their scores become relatively similar. Larger improvements may therefore require richer content or sequence representations.

Most experiment decisions use three to five random seeds. That is sufficient for the project's predefined **0.002** promotion threshold relative to the measured run-to-run variation, but we do not treat smaller differences as precise effects.

Autonomy also has a boundary. The clean-room, verification, v2-loop, and later unattended campaigns demonstrate research without iteration-level human intervention. The highest-scoring final system comes from the longer research process, which included a small number of strategic human decisions that are recorded in `logs/INTERVENTIONS.md`.

Given more time, we would test richer models over ordered user histories, examine training methods that use random-exposure data without violating time order, and add more automatic checks inside each research iteration.

## Team contributions

### Steve Wilson Koesasih: Autonomous research system and experimentation

Steve worked on the agent's research process, including how it proposes experiments, runs them, checks validation results, and decides whether a change should be kept. He also worked on connecting the research agent to the model-training pipeline, debugging experiments, and selecting the final validated model.

### Farren Ananda Widjaja: Recommendation model and behavioral features

Farren worked on the recommendation model and the signals that describe recent user behavior. His work included recent-interaction features, author and tag history, timing signals, feature experiments, and tests that measured how those additions changed recommendation performance.

### Devin Nathaniel: Evaluation and scientific testing

Devin worked on evaluating whether improvements were reliable and whether our explanations matched the evidence. His work included benchmark reproduction, GAUC and nDCG@5 evaluation, control experiments, the timing test, random-exposure evaluation, and checks that model selection remained based on validation results.

### Tristan Benedict Kandou: Scaling and reproducibility

Tristan worked on transferring the model to the larger KuaiRand-1K dataset and on the engineering changes needed to run it efficiently. He also worked on frozen-model verification, serving-update experiments, resource measurements, and the scripts used to reproduce the final results.

## Repository

https://github.com/trisbk/Bridges-Techjam-Submission
