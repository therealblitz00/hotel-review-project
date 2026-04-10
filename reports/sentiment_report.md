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
| VADER (baseline) | 0.7150 | 0.4511 | - |
| TF-IDF + LR | 0.7400 | 0.5293 | 0.4909 +/- 0.0388 |
| TF-IDF + LinearSVC | 0.8000 | 0.5344 | 0.4436 +/- 0.0434 |
| XLM-RoBERTa (fine-tuned) | 0.7800 | 0.2921 | - |

### LinearSVC per-class (3-class, test set)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| positive | 0.859 | 0.936 | 0.896 | 156 |
| neutral | 0.478 | 0.355 | 0.407 | 31 |
| negative | 0.429 | 0.231 | 0.3 | 13 |

## Model results - binary (positive vs non-positive)

| Model | Accuracy | F1-macro | CV F1-macro |
| --- | --- | --- | --- |
| TF-IDF + LinearSVC | 0.7500 | 0.6471 | 0.6939 +/- 0.0110 |

### LinearSVC per-class (binary, test set)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| positive | 0.849 | 0.827 | 0.838 | 156 |
| non-positive | 0.438 | 0.477 | 0.457 | 44 |

## Transformer upgrade (XLM-RoBERTa)

- Status: success
- Model: xlm-roberta-base
- Accuracy: 0.7800
- F1-macro: 0.2921

### XLM-R per-class (3-class, test set)

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| positive | 0.78 | 1.0 | 0.876 | 156 |
| neutral | 0.0 | 0.0 | 0.0 | 31 |
| negative | 0.0 | 0.0 | 0.0 | 13 |

## Interpretation

LinearSVC remains a robust baseline and binary classification is still the most stable task. The transformer model is included as an advanced multilingual upgrade path for richer NLP depth.

## Figures

- `reports/figures/fig_sentiment_confusion_lr.png`
- `reports/figures/fig_sentiment_confusion_svc.png`
- `reports/figures/fig_sentiment_confusion_binary.png`
- `reports/figures/fig_sentiment_confusion_vader.png`
- `reports/figures/fig_sentiment_confusion_xlmr.png`
- `reports/figures/fig_sentiment_model_comparison.png`
