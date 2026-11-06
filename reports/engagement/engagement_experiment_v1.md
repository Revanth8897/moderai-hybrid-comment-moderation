# MODERAI — Engagement Classifier V1 Experiment Report

## 1. Experiment Overview

- Experiment: Engagement Intent Classification V1
- Model: TF-IDF + Logistic Regression
- Task: Single-label, five-class engagement intent classification
- Status: Baseline experiment completed
- Test dataset: Not accessed during training or validation

## 2. Objective

Build an initial classifier that identifies the primary intent of a
comment so MODERAI can make better response-routing decisions.

The five supervised classes are:

| Class | Meaning |
|---|---|
| POSITIVE | Appreciation, satisfaction, or praise |
| NEGATIVE | Dissatisfaction without constructive suggestions |
| CRITICAL | Specific criticism or improvement suggestions |
| QUESTION | Request for information or clarification |
| NEUTRAL | Factual statement without clear affect or request |

UNCLEAR comments are excluded from supervised training and retained
for validation-time abstention analysis.

## 3. Dataset

### Training data

- Total comments: 298
- Real-world comments: 273
- Synthetic seed comments: 25
- Synthetic UNCLEAR examples excluded from supervised training

Training class distribution:

| Intent | Comments |
|---|---:|
| POSITIVE | 97 |
| NEGATIVE | 66 |
| CRITICAL | 53 |
| QUESTION | 41 |
| NEUTRAL | 41 |
| **Total** | **298** |

### Validation data

- Total comments: 57
- Clear-intent comments: 54
- UNCLEAR comments: 3

Validation class distribution:

| Intent | Comments |
|---|---:|
| POSITIVE | 18 |
| NEGATIVE | 12 |
| CRITICAL | 10 |
| QUESTION | 7 |
| NEUTRAL | 7 |
| UNCLEAR | 3 |
| **Total** | **57** |

Data was split by video title to prevent comments from the same video
appearing across training, validation, and test sets.

Synthetic examples were used only in training.

## 4. Model Architecture

Pipeline:

Comment text
    |
    v
TF-IDF Vectorizer
    |
    v
Logistic Regression
    |
    v
Five intent probabilities
    |
    v
Predicted intent + confidence signals

### TF-IDF configuration

- lowercase: True
- strip_accents: unicode
- max_features: 20,000
- ngram_range: (1, 2)
- min_df: 1
- max_df: 1.0
- sublinear_tf: True
- dtype: float32

Actual vocabulary size: 5,943 features.

### Logistic Regression configuration

- C: 1.0
- class_weight: balanced
- max_iter: 2,000
- solver: lbfgs
- random_state: 42

## 5. Validation Results

Evaluation was performed on 54 clear-intent validation comments.

| Metric | Result |
|---|---:|
| Accuracy | 55.56% |
| Macro F1 | 0.5171 |
| Weighted F1 | 0.5470 |

### Per-class results

| Intent | Precision | Recall | F1 |
|---|---:|---:|---:|
| POSITIVE | 0.62 | 0.72 | 0.67 |
| NEGATIVE | 0.55 | 0.50 | 0.52 |
| CRITICAL | 0.50 | 0.50 | 0.50 |
| QUESTION | 0.50 | 0.57 | 0.53 |
| NEUTRAL | 0.50 | 0.29 | 0.36 |

These results are preliminary because the validation set is small.

## 6. Confidence Threshold Analysis

A threshold sweep was performed using validation predictions.

At a threshold of 0.32:

- Accepted clear comments: 6 of 54
- Accuracy among accepted comments: 100%
- All six accepted comments were POSITIVE.
- No other intent class was accepted.
- All three UNCLEAR comments were rejected.

This threshold is not suitable as a production threshold.
It accepts very few examples and is biased toward POSITIVE predictions.

Decision: Do not enable automatic acceptance based on this threshold.

## 7. Confidence Margin Analysis

Margin is calculated as:

Highest class probability - Second-highest class probability

Validation results for clear-intent comments:

| Margin group | Comments | Correct | Accuracy |
|---|---:|---:|---:|
| Very low | 18 | 7 | 38.9% |
| Low | 15 | 7 | 46.7% |
| Medium | 13 | 10 | 76.9% |
| High | 8 | 6 | 75.0% |

### Observations

- Very low and low margins were associated with lower accuracy.
- Medium and high margin groups performed better in this validation sample.
- High margin did not outperform medium margin.
- Some incorrect predictions had relatively large margins.
- Some correct predictions had very small margins.

Decision: Store confidence and margin as informational signals.
Do not treat either as a guarantee of correctness.

## 8. V1 Decision

The engagement classifier will initially operate in assisted mode.

Its output will contain:

- predicted_intent
- max_probability
- confidence_margin
- class_probabilities

The response-routing engine will use these signals alongside
deterministic rules and the independent safety engine.

Low-confidence or ambiguous cases should be routed for human review.

No automatic publishing or moderation action should be based solely
on engagement classification.

## 9. Known Limitations

1. The dataset is small for a six-intent real-world classification task.
2. The UNCLEAR class has only three validation examples.
3. The test dataset has not yet been evaluated.
4. Synthetic seed examples may not represent natural comment diversity.
5. Validation accuracy and margin patterns may not generalize.
6. Class-level confidence calibration is not yet reliable.
7. The model may struggle with sarcasm, mixed intent, slang, and
   context-dependent comments.

## 10. Next Improvements

- Expand human-reviewed real-world annotations.
- Include more diverse examples for underrepresented intents.
- Evaluate on the untouched test dataset.
- Inspect class-specific errors.
- Improve confidence calibration using a larger validation set.
- Compare future models against this V1 baseline.
- Integrate the classifier into the MODERAI inference pipeline.

## 11. Saved Artifacts

- Model: models/engagement/engagement_tfidf_logreg_v1.joblib
- Validation report: reports/engagement/engagement_validation_report_v1.json
- Validation predictions: reports/engagement/engagement_validation_predictions_v1.csv
- Confusion matrix: reports/engagement/engagement_validation_confusion_v1.csv
- Training split: datasets/engagement/splits/train_v2.csv
- Validation split: datasets/engagement/splits/validation_v2.csv
- Test split: datasets/engagement/splits/test_v2.csv

---

**Experiment status:** V1 baseline completed.
**Production status:** Not production-ready.
**Evaluation policy:** Keep the test set untouched until final evaluation.
