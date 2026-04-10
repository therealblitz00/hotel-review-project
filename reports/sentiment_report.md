# Sentiment Analysis Report

**Dataset:** `C:\Users\manue\Desktop\mestrado\DAII\hotel-review-project\data\processed\reviews_clean.csv`  
**Reviews:** 999  
**Train / test split:** 80 / 20, stratified, random_state=42  

## Label definition

Labels are derived from Booking.com numerical scores:

- positive: score >= 8
- neutral: score 6-7
- negative: score < 6

## Root cause of low 3-class F1

The boundary between positive (score 8) and neutral (score 7) is weakly encoded in text. This causes structural ambiguity for 3-class text classifiers. Binary classification (positive vs non-positive) remains more stable.

## Model results - 3-class

| Model | Accuracy | F1-macro | CV F1-macro |
| --- | --- | --- | --- |
| VADER (baseline) | 0.6700 | 0.4573 | - |
| TF-IDF + LR | 0.7350 | 0.5245 | 0.4917 +/- 0.0231 |
| TF-IDF + LinearSVC | 0.7850 | 0.4993 | 0.4501 +/- 0.0342 |

### LinearSVC per-class (3-class, test set)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| positive | 0.849 | 0.936 | 0.89 | 156 |
| neutral | 0.381 | 0.258 | 0.308 | 31 |
| negative | 0.429 | 0.231 | 0.3 | 13 |

## Model results - binary (positive vs non-positive)

| Model | Accuracy | F1-macro | CV F1-macro |
| --- | --- | --- | --- |
| TF-IDF + LinearSVC | 0.7650 | 0.6548 | 0.7019 +/- 0.0116 |

### LinearSVC per-class (binary, test set)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| positive | 0.847 | 0.853 | 0.85 | 156 |
| non-positive | 0.465 | 0.455 | 0.46 | 44 |

## Transformer upgrade (XLM-RoBERTa)

- Status: skipped
- Reason: transformer dependencies unavailable: No module named 'torch'

## Interpretation

LinearSVC remains a robust baseline and binary classification is still the most stable task. The transformer model is included as an advanced multilingual upgrade path for richer NLP depth.

## Figures

- `reports/figures/fig_sentiment_confusion_lr.png`
- `reports/figures/fig_sentiment_confusion_svc.png`
- `reports/figures/fig_sentiment_confusion_binary.png`
- `reports/figures/fig_sentiment_confusion_vader.png`
- `reports/figures/fig_sentiment_model_comparison.png`
