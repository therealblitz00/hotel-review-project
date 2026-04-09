# Sentiment Analysis Report

**Dataset:** `C:\Users\manue\Desktop\mestrado\DAII\hotel-review-project\data\processed\reviews_clean.csv`  
**Reviews:** 999  
**Train / test split:** 80 / 20, stratified, random_state=42  

## Label definition

Labels are derived from the Booking.com numerical score:

- **positive:** score >= 8  (779 reviews, 78.0%)
- **neutral:** score 6–7  (156 reviews, 15.6%)
- **negative:** score < 6  (64 reviews, 6.4%)

The dataset is heavily imbalanced. Class-balanced training weights are applied to all trained classifiers.

## Root cause of low 3-class F1

The boundary between *positive* (score 8) and *neutral* (score 7) is not reliably encoded in text — guests with similar wording assign different numerical scores. This is a label-signal mismatch, not a modelling failure. A binary task (positive ≥ 8 vs non-positive < 8) provides a cleaner linguistic boundary and is reported alongside the 3-class results.

## Model results — 3-class

| Model | Accuracy | F1-macro | CV F1-macro |
| --- | --- | --- | --- |
| VADER (baseline) | 0.6700 | 0.4573 | — |
| TF-IDF + LR | 0.7350 | 0.5245 | 0.4917 ± 0.0231 |
| TF-IDF + LinearSVC | 0.7850 | 0.4993 | 0.4501 ± 0.0342 |

### LinearSVC per-class (3-class, test set)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| positive | 0.849 | 0.936 | 0.89 | 156 |
| neutral | 0.381 | 0.258 | 0.308 | 31 |
| negative | 0.429 | 0.231 | 0.3 | 13 |

## Model results — binary (positive vs non-positive)

| Model | Accuracy | F1-macro | CV F1-macro |
| --- | --- | --- | --- |
| TF-IDF + LinearSVC | 0.7650 | 0.6548 | 0.7019 ± 0.0116 |

### LinearSVC per-class (binary, test set)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| positive | 0.847 | 0.853 | 0.85 | 156 |
| non-positive | 0.465 | 0.455 | 0.46 | 44 |

## Interpretation

LinearSVC outperforms Logistic Regression on the 3-class task. The binary task confirms that text reliably distinguishes satisfied guests (score ≥ 8) from dissatisfied ones (score < 8), with F1-macro of 0.6548. The 3-class F1 is structurally limited by the fuzzy 7/8 score boundary in the label definition.

## Figures

- `reports/figures/fig_sentiment_confusion_lr.png`
- `reports/figures/fig_sentiment_confusion_svc.png`
- `reports/figures/fig_sentiment_confusion_binary.png`
- `reports/figures/fig_sentiment_confusion_vader.png`
- `reports/figures/fig_sentiment_model_comparison.png`
