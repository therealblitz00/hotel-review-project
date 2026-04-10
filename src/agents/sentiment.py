from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from src.utils.paths import NLTK_DATA_DIR

NLTK_DATA_DIR.mkdir(exist_ok=True)
if str(NLTK_DATA_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DATA_DIR))
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon", download_dir=str(NLTK_DATA_DIR), quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.orchestration.state import WorkflowState
from src.utils.logging_utils import get_logger
from src.utils.paths import ARTIFACTS_DIR, FIGURES_DIR, PROCESSED_DIR, REPORTS_DIR

logger = get_logger(__name__)

CLEAN_CSV = PROCESSED_DIR / "reviews_clean.csv"
SENTIMENT_REPORT = REPORTS_DIR / "sentiment_report.md"
METRICS_ARTIFACT = ARTIFACTS_DIR / "sentiment_metrics.json"
PREDICTIONS_ARTIFACT = ARTIFACTS_DIR / "sentiment_predictions.csv"
TRANSFORMER_METRICS_ARTIFACT = ARTIFACTS_DIR / "sentiment_transformer_metrics.json"
TRANSFORMER_PREDICTIONS_ARTIFACT = ARTIFACTS_DIR / "sentiment_transformer_predictions.csv"

LABEL_ORDER_3 = ["positive", "neutral", "negative"]
TRANSFORMER_MODEL_NAME = "xlm-roberta-base"
TRANSFORMER_MAX_LENGTH = 192
TRANSFORMER_EPOCHS = 2
TRANSFORMER_TRAIN_BATCH_SIZE = 8
TRANSFORMER_EVAL_BATCH_SIZE = 16

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - environment dependent
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_IMPORT_ERROR = str(exc)


if TRANSFORMERS_AVAILABLE:
    class _TextDataset(torch.utils.data.Dataset):
        def __init__(self, encodings: dict[str, list[int]], labels: list[int]):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx: int) -> dict[str, Any]:
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self) -> int:
            return len(self.labels)


def _vader_label(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def _tfidf_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=8000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("clf", classifier),
        ]
    )


def _plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, fname: str) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="Actual",
        title=title,
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=11,
            )
    plt.tight_layout()
    path = FIGURES_DIR / fname
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved figure: %s", path)
    return str(path)


def _eval_model(pipeline, X_train, X_test, y_train, y_test, labels, cv_X, cv_y) -> dict:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, cv_X, cv_y, cv=cv, scoring="f1_macro")
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    per_class = {
        lbl: {k: round(float(report[lbl][k]), 3) for k in ("precision", "recall", "f1-score", "support")}
        for lbl in labels
        if lbl in report
    }
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "cv_f1_macro_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_macro_std": round(float(cv_f1.std()), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "y_pred": y_pred,
        "pipeline": pipeline,
    }


def _train_eval_transformer(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    label_order: list[str],
) -> dict[str, Any]:
    if not TRANSFORMERS_AVAILABLE:
        return {
            "status": "skipped",
            "reason": f"transformer dependencies unavailable: {TRANSFORMERS_IMPORT_ERROR}",
        }

    try:
        set_seed(42)
        label2id = {lbl: i for i, lbl in enumerate(label_order)}
        id2label = {i: lbl for lbl, i in label2id.items()}

        tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            TRANSFORMER_MODEL_NAME,
            num_labels=len(label_order),
            id2label=id2label,
            label2id=label2id,
        )

        y_train_ids = [label2id[v] for v in y_train.tolist()]
        y_test_ids = [label2id[v] for v in y_test.tolist()]

        enc_train = tokenizer(
            X_train.tolist(),
            truncation=True,
            padding=True,
            max_length=TRANSFORMER_MAX_LENGTH,
        )
        enc_test = tokenizer(
            X_test.tolist(),
            truncation=True,
            padding=True,
            max_length=TRANSFORMER_MAX_LENGTH,
        )

        train_ds = _TextDataset(enc_train, y_train_ids)
        test_ds = _TextDataset(enc_test, y_test_ids)

        training_args = TrainingArguments(
            output_dir=str(ARTIFACTS_DIR / "tmp_xlmr_sentiment"),
            num_train_epochs=TRANSFORMER_EPOCHS,
            per_device_train_batch_size=TRANSFORMER_TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=TRANSFORMER_EVAL_BATCH_SIZE,
            learning_rate=2e-5,
            weight_decay=0.01,
            save_strategy="no",
            report_to="none",
            seed=42,
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
        )

        trainer.train()
        pred_logits = trainer.predict(test_ds).predictions
        pred_ids = np.argmax(pred_logits, axis=1)
        y_pred = [id2label[int(i)] for i in pred_ids]

        report = classification_report(y_test, y_pred, labels=label_order, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=label_order)
        per_class = {
            lbl: {k: round(float(report[lbl][k]), 3) for k in ("precision", "recall", "f1-score", "support")}
            for lbl in label_order
            if lbl in report
        }

        return {
            "status": "success",
            "model": TRANSFORMER_MODEL_NAME,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "y_pred": y_pred,
            "_tokenizer": tokenizer,
            "_model_obj": model,
            "_id2label": id2label,
        }
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        logger.warning("Transformer training skipped: %s", exc)
        return {
            "status": "skipped",
            "reason": f"training failed: {exc}",
        }


def _predict_transformer_full(model, tokenizer, id2label: dict[int, str], texts: pd.Series) -> list[str]:
    if not TRANSFORMERS_AVAILABLE:
        return []

    model.eval()
    preds: list[str] = []

    for i in range(0, len(texts), TRANSFORMER_EVAL_BATCH_SIZE):
        batch = texts.iloc[i : i + TRANSFORMER_EVAL_BATCH_SIZE].tolist()
        enc = tokenizer(batch, truncation=True, padding=True, max_length=TRANSFORMER_MAX_LENGTH, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()
        preds.extend([id2label[int(pid)] for pid in pred_ids])

    return preds


def run_sentiment(state: WorkflowState) -> WorkflowState:
    logger.info("Running sentiment analysis agent")
    state.current_step = "sentiment"

    if not CLEAN_CSV.exists():
        msg = f"Clean data file not found: {CLEAN_CSV}"
        logger.error(msg)
        state.warnings.append(msg)
        state.results["sentiment"] = {"status": "failed", "reason": msg}
        return state

    df = pd.read_csv(CLEAN_CSV, encoding="utf-8-sig")
    X = df["full_review_text"].fillna("")
    y3 = df["sentiment"]
    y2 = (df["score"] >= 8).map({True: "positive", False: "non-positive"})

    X_tr3, X_te3, y_tr3, y_te3 = train_test_split(X, y3, test_size=0.2, random_state=42, stratify=y3)
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y2, test_size=0.2, random_state=42, stratify=y2)
    logger.info("3-class train=%d test=%d", len(X_tr3), len(X_te3))
    logger.info("binary  train=%d test=%d", len(X_tr2), len(X_te2))

    figures: list[str] = []

    lr_pipe = _tfidf_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42, C=1.0)
    )
    lr = _eval_model(lr_pipe, X_tr3, X_te3, y_tr3, y_te3, LABEL_ORDER_3, X, y3)
    figures.append(
        _plot_confusion_matrix(np.array(lr["confusion_matrix"]), LABEL_ORDER_3, "LR - 3-class confusion matrix", "fig_sentiment_confusion_lr.png")
    )

    svc_pipe = _tfidf_pipeline(
        LinearSVC(max_iter=2000, class_weight="balanced", random_state=42, C=1.0)
    )
    svc = _eval_model(svc_pipe, X_tr3, X_te3, y_tr3, y_te3, LABEL_ORDER_3, X, y3)
    figures.append(
        _plot_confusion_matrix(np.array(svc["confusion_matrix"]), LABEL_ORDER_3, "LinearSVC - 3-class confusion matrix", "fig_sentiment_confusion_svc.png")
    )

    label_order_2 = ["positive", "non-positive"]
    bin_pipe = _tfidf_pipeline(
        LinearSVC(max_iter=2000, class_weight="balanced", random_state=42, C=1.0)
    )
    bin_ = _eval_model(bin_pipe, X_tr2, X_te2, y_tr2, y_te2, label_order_2, X, y2)
    figures.append(
        _plot_confusion_matrix(
            np.array(bin_["confusion_matrix"]),
            label_order_2,
            "LinearSVC - binary confusion matrix",
            "fig_sentiment_confusion_binary.png",
        )
    )

    sia = SentimentIntensityAnalyzer()
    vader_pred = X_te3.apply(lambda t: _vader_label(sia.polarity_scores(t)["compound"]))
    vader_acc = float(accuracy_score(y_te3, vader_pred))
    vader_f1 = float(f1_score(y_te3, vader_pred, average="macro", zero_division=0))
    vader_cm = confusion_matrix(y_te3, vader_pred, labels=LABEL_ORDER_3)
    figures.append(
        _plot_confusion_matrix(vader_cm, LABEL_ORDER_3, "VADER - 3-class confusion matrix", "fig_sentiment_confusion_vader.png")
    )

    transformer = _train_eval_transformer(X_tr3, X_te3, y_tr3, y_te3, LABEL_ORDER_3)
    if transformer.get("status") == "success":
        figures.append(
            _plot_confusion_matrix(
                np.array(transformer["confusion_matrix"]),
                LABEL_ORDER_3,
                "XLM-R - 3-class confusion matrix",
                "fig_sentiment_confusion_xlmr.png",
            )
        )

    comparison: list[tuple[str, float]] = [
        ("VADER\n(baseline)", round(vader_f1, 3)),
        ("TF-IDF+LR\n(3-class)", round(lr["f1_macro"], 3)),
        ("TF-IDF+SVC\n(3-class)", round(svc["f1_macro"], 3)),
        ("TF-IDF+SVC\n(binary)", round(bin_["f1_macro"], 3)),
    ]
    if transformer.get("status") == "success":
        comparison.append(("XLM-R\n(3-class)", round(float(transformer["f1_macro"]), 3)))

    colors = ["#90A4AE", "#1976D2", "#0D47A1", "#FF9800", "#2E7D32"][: len(comparison)]
    labels_bar, f1s = zip(*comparison)
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels_bar, f1s, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1-macro")
    ax.set_title("Sentiment model comparison - F1-macro")
    ax.spines[["top", "right"]].set_visible(False)
    compare_fig = FIGURES_DIR / "fig_sentiment_model_comparison.png"
    plt.tight_layout()
    plt.savefig(compare_fig, dpi=120, bbox_inches="tight")
    plt.close()
    figures.append(str(compare_fig))

    svc_full_pred = svc["pipeline"].predict(X)
    bin_full_pred = bin_["pipeline"].predict(X)

    df_out = df[["name", "date_parsed", "score", "sentiment"]].copy()
    df_out["svc_3class_pred"] = svc_full_pred
    df_out["svc_binary_pred"] = bin_full_pred
    df_out["svc_3class_correct"] = (df_out["svc_3class_pred"] == df_out["sentiment"]).astype(int)

    if transformer.get("status") == "success":
        transformer_full_pred = _predict_transformer_full(
            transformer["_model_obj"],
            transformer["_tokenizer"],
            transformer["_id2label"],
            X,
        )
        if transformer_full_pred:
            df_out["xlmr_3class_pred"] = transformer_full_pred
            df_out["xlmr_3class_correct"] = (df_out["xlmr_3class_pred"] == df_out["sentiment"]).astype(int)
            df_out[["name", "date_parsed", "score", "sentiment", "xlmr_3class_pred", "xlmr_3class_correct"]].to_csv(
                TRANSFORMER_PREDICTIONS_ARTIFACT,
                index=False,
                encoding="utf-8-sig",
            )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(PREDICTIONS_ARTIFACT, index=False, encoding="utf-8-sig")

    transformer_public = {
        k: v
        for k, v in transformer.items()
        if not k.startswith("_")
    }
    TRANSFORMER_METRICS_ARTIFACT.write_text(json.dumps(transformer_public, indent=2), encoding="utf-8")

    metrics = {
        "best_3class_model": "TF-IDF + LinearSVC (balanced)",
        "best_binary_model": "TF-IDF + LinearSVC (balanced)",
        "transformer_model": TRANSFORMER_MODEL_NAME,
        "train_test_split": {"test_size": 0.2, "random_state": 42, "stratified": True},
        "label_discussion": (
            "3-class labels derive from score (>=8 positive, 6-7 neutral, <6 negative). "
            "Neutral/negative F1 is structurally limited: text between score-7 and score-8 "
            "reviews is linguistically near-identical. Binary task provides a cleaner signal."
        ),
        "vader_baseline": {
            "accuracy": round(vader_acc, 4),
            "f1_macro": round(vader_f1, 4),
            "confusion_matrix": vader_cm.tolist(),
        },
        "lr_3class": {k: v for k, v in lr.items() if k not in ("y_pred", "pipeline")},
        "svc_3class": {k: v for k, v in svc.items() if k not in ("y_pred", "pipeline")},
        "svc_binary": {k: v for k, v in bin_.items() if k not in ("y_pred", "pipeline")},
        "xlmr_3class": transformer_public,
        "figures": figures,
    }
    METRICS_ARTIFACT.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def _per_class_table(pc: dict, labels: list[str]) -> list[str]:
        rows = ["| Class | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |"]
        for lbl in labels:
            if lbl in pc:
                r = pc[lbl]
                rows.append(
                    f"| {lbl} | {r['precision']} | {r['recall']} | {r['f1-score']} | {int(r['support'])} |"
                )
        return rows

    lines: list[str] = [
        "# Sentiment Analysis Report",
        "",
        f"**Dataset:** `{CLEAN_CSV}`  ",
        f"**Reviews:** {len(df)}  ",
        "**Train / test split:** 80 / 20, stratified, random_state=42  ",
        "",
        "## Label definition",
        "",
        "Labels are derived from Booking.com numerical scores:",
        "",
        "- positive: score >= 8",
        "- neutral: score 6-7",
        "- negative: score < 6",
        "",
        "## Root cause of low 3-class F1",
        "",
        "The boundary between positive (score 8) and neutral (score 7) is weakly encoded in text. "
        "This causes structural ambiguity for 3-class text classifiers. "
        "Binary classification (positive vs non-positive) remains more stable.",
        "",
        "## Model results - 3-class",
        "",
        "| Model | Accuracy | F1-macro | CV F1-macro |",
        "| --- | --- | --- | --- |",
        f"| VADER (baseline) | {vader_acc:.4f} | {vader_f1:.4f} | - |",
        f"| TF-IDF + LR | {lr['accuracy']:.4f} | {lr['f1_macro']:.4f} | {lr['cv_f1_macro_mean']:.4f} +/- {lr['cv_f1_macro_std']:.4f} |",
        f"| TF-IDF + LinearSVC | {svc['accuracy']:.4f} | {svc['f1_macro']:.4f} | {svc['cv_f1_macro_mean']:.4f} +/- {svc['cv_f1_macro_std']:.4f} |",
    ]

    if transformer.get("status") == "success":
        lines.append(
            f"| XLM-RoBERTa (fine-tuned) | {float(transformer['accuracy']):.4f} | {float(transformer['f1_macro']):.4f} | - |"
        )

    lines += [
        "",
        "### LinearSVC per-class (3-class, test set)",
        "",
        *_per_class_table(svc["per_class"], LABEL_ORDER_3),
        "",
        "## Model results - binary (positive vs non-positive)",
        "",
        "| Model | Accuracy | F1-macro | CV F1-macro |",
        "| --- | --- | --- | --- |",
        f"| TF-IDF + LinearSVC | {bin_['accuracy']:.4f} | {bin_['f1_macro']:.4f} | {bin_['cv_f1_macro_mean']:.4f} +/- {bin_['cv_f1_macro_std']:.4f} |",
        "",
        "### LinearSVC per-class (binary, test set)",
        "",
        *_per_class_table(bin_["per_class"], label_order_2),
        "",
        "## Transformer upgrade (XLM-RoBERTa)",
        "",
    ]

    if transformer.get("status") == "success":
        lines += [
            f"- Status: success",
            f"- Model: {TRANSFORMER_MODEL_NAME}",
            f"- Accuracy: {float(transformer['accuracy']):.4f}",
            f"- F1-macro: {float(transformer['f1_macro']):.4f}",
            "",
            "### XLM-R per-class (3-class, test set)",
            "",
            *_per_class_table(transformer["per_class"], LABEL_ORDER_3),
        ]
    else:
        lines += [
            "- Status: skipped",
            f"- Reason: {transformer.get('reason', 'unknown')}",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "LinearSVC remains a robust baseline and binary classification is still the most stable task. "
        "The transformer model is included as an advanced multilingual upgrade path for richer NLP depth.",
        "",
        "## Figures",
        "",
    ]

    for fig_path in figures:
        fname = fig_path.replace("\\", "/").split("/")[-1]
        lines.append(f"- `reports/figures/{fname}`")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SENTIMENT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    best_3class_name = "LR" if lr["f1_macro"] >= svc["f1_macro"] else "SVC"
    best_3class = max(lr["f1_macro"], svc["f1_macro"])
    state.logs.append(
        f"Sentiment complete: best 3-class F1-macro={best_3class:.4f} ({best_3class_name}), "
        f"binary SVC F1-macro={bin_['f1_macro']:.4f} (CV={bin_['cv_f1_macro_mean']:.4f}), "
        f"XLM-R status={transformer.get('status', 'unknown')}"
    )
    state.results["sentiment"] = {
        "status": "success",
        "svc_3class_f1_macro": svc["f1_macro"],
        "svc_3class_cv_f1": svc["cv_f1_macro_mean"],
        "svc_binary_f1_macro": bin_["f1_macro"],
        "svc_binary_cv_f1": bin_["cv_f1_macro_mean"],
        "vader_f1_macro": round(vader_f1, 4),
        "xlmr_3class_status": transformer_public.get("status", "skipped"),
        "xlmr_3class_f1_macro": transformer_public.get("f1_macro"),
        "report": str(SENTIMENT_REPORT),
        "metrics_artifact": str(METRICS_ARTIFACT),
        "predictions_artifact": str(PREDICTIONS_ARTIFACT),
        "transformer_metrics_artifact": str(TRANSFORMER_METRICS_ARTIFACT),
        "transformer_predictions_artifact": str(TRANSFORMER_PREDICTIONS_ARTIFACT),
    }
    return state
