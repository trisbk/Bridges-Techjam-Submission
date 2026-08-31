# Final Submission and Results Summary

## Final model files

The required output is stored under `final_output/` in the public repository.

`final_output/submission.csv` contains one prediction for every row in the KuaiRand-Pure test split. It follows the starter-kit schema:

```text
row_id, user_id, video_id, score
```

The file contains **170,588 aligned rows** and has been checked with the starter kit's own validation code.

`final_output/frozen_model/` contains the final **R33c** checkpoint. It includes five seed-specific weight files and the configuration used to train them.

`final_output/frozen_model_r24b/` contains the earlier **0.6116** champion. We keep it because several post-freeze analyses were run before R33c was promoted.

Retrain the final model with:

```bash
cd code
python3 final_model.py
```

Training takes about **five minutes on CPU**.

Score the frozen model without retraining with:

```bash
cd code
python3 score_frozen.py
```

## Required benchmark: KuaiRand-Pure

The final submission was selected using validation results and then evaluated on the test split.

| Metric | Official baseline | Our validation | Our test | Improvement over baseline |
|---|---:|---:|---:|---:|
| GAUC | 0.6610 | 0.6948 | **0.6857** | **+0.0247** |
| nDCG@5 | 0.5282 | 0.5464 | **0.5429** | **+0.0147** |
| **Primary** | **0.5946** | **0.6206** | **0.6143** | **+0.0197** |

The official score uses the mean absolute improvement across GAUC and nDCG@5:

```text
(0.0247 + 0.0147) / 2 = +0.0197
```

The final primary score therefore improves from:

## **0.5946 to 0.6143**

R33c combines the established behavioral recipe with `tab_n`, a feature proposed through the autonomous research loop. The previous champion, R24b, scored about **0.6116**.

## Bonus benchmark: KuaiRand-1K

We transferred the frozen recipe to KuaiRand-1K without changing its model settings specifically for the larger dataset.

The label definition, split logic, feature definitions, training objective, model type, hyperparameters, and committee construction remained fixed.

| KuaiRand-1K, 11.7M rows and 1,000 users | Validation | Test primary |
|---|---:|---:|
| Reproduced starter-kit baseline |  | 0.6293 |
| **Our frozen recipe** | **0.6868** | **0.6931** |
| **Improvement** |  | **+0.0637** |

The committee's test components are:

- GAUC: **0.7017**
- nDCG@5: **0.6844**
- Individual primary scores: **0.6874 to 0.6920**

KuaiRand-1K provides much deeper interaction histories per user than KuaiRand-Pure. Because our model uses recent behavioral history, the larger improvement is consistent with the hypothesis that these features become more useful when more history is available. The transfer is not a controlled test of history depth alone, so other dataset differences may also contribute.

KuaiRand-27K was not attempted.

## Resource usage

No GPU was used.

**GPU-hours: 0**

All model training ran on a single laptop CPU with NumPy.

| Run | Iterations | Wall-clock | LLM input + output tokens | Including cache reads/writes |
|---|---:|---:|---:|---:|
| Verification run | 3 | 52 min 32 s | 149,658 | 6,188,547 |
| Clean-room run | 6 | 1 h 47 min 41 s | 325,476 | 17,978,927 |
| Interactive culminating run | 11 | At most 1 h 38 min measured | Not separately metered | n/a |
| v2-loop iteration | 1 | About 18 min | Not fully metered | n/a |
| Campaign 5 completion | 4 | About 13 min measured training | No agent sessions | n/a |
| Campaign 6 | 3 | 49 min 11 s | Not separately metered | n/a |

Completed campaigns remained within the **50-iteration limit** and **six-hour wall-clock limit**.

A typical three-seed experiment requires about **40 to 90 seconds** of model training. Much of the remaining wall-clock time comes from the agent's research process and experiment management.

## Why we think recent behavior matters

The model gains much of its improvement from features describing recent user behavior. We tested whether the timing of those features was actually responsible.

The control keeps the same general behavioral information but shuffles when it is attached within each user's history.

| Experiment | Test primary |
|---|---:|
| **Correctly timed behavioral features** | **0.61164** |
| No sequence features | 0.59808 |
| Behavioral features attached to the wrong times | 0.59872 |
| Matched random information | 0.59876 |

About **95% of the measured sequence-feature gain disappears when the timing is broken**.

Incorrectly timed features also perform almost exactly like matched random information. The result supports the claim that the model benefits mainly from behavior attached to the correct moment, rather than from simply receiving additional user-related inputs.

This experiment uses the R24b champion because it was run before the final R33c promotion.

## What happens when user history is less fresh?

The main model can update its small user-history state after each interaction. We also measured systems that update less often.

| History update method | Test primary | Improvement over baseline |
|---|---:|---:|
| Continuous | 0.6116 | +0.0170 |
| Daily refresh, retrained | 0.6106 | +0.0160 |
| Frozen history, retrained | 0.5979 | +0.0033 |

Daily refresh retains about **94% of the measured continuous-update gain** when the model is trained under that same serving condition.

The fully frozen version still remains above the official baseline after retraining, although its advantage is much smaller.

## Random-exposure evaluation

Historical recommendation logs reflect choices made by the previous recommender. Users cannot respond to videos they were never shown.

We therefore evaluated the pre-promotion model on **897,721 randomly exposed impressions**.

| Random-exposure evaluation | Primary |
|---|---:|
| Matched baseline | 0.3707 |
| **Our model** | **0.3777** |
| **Improvement** | **+0.0070** |

The model remains ahead under random exposure. This suggests that the improvement is not entirely explained by the earlier recommender's exposure choices.

The test has a limited scope. It uses R24b rather than the final R33c model, and it retains continuous history updates.

## Exposure-bias experiment

A second experiment changed how training treats videos that appeared very frequently in the historical data.

| Exposure correction | Standard test | Random-exposure test |
|---|---:|---:|
| None | **0.6009** | 0.3785 |
| Mild | 0.5840 | 0.4122 |
| Strong | 0.5654 | **0.4181** |

Correcting more strongly for historical exposure improves the random-exposure evaluation but lowers the ordinary logged-test score.

For the competition, we use no exposure correction because the competition score is based on the standard logged test. The experiment is reported to show that performance on historical recommendation logs and performance under more neutral exposure are related but not identical objectives.

## Public files

Repository:

https://github.com/trisbk/Bridges-Techjam-Submission

Final output:

https://github.com/trisbk/Bridges-Techjam-Submission/tree/main/final_output

Run logs:

https://github.com/trisbk/Bridges-Techjam-Submission/tree/main/logs
