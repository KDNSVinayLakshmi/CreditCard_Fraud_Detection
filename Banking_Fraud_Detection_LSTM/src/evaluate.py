"""Held-out evaluation and report/plot generation."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

from src.data_loader import analyze_dataset, load_dataset
from src.preprocess import prepare_sequences
from src.utils import MODEL_DIR, PLOTS_DIR, SEQUENCE_LENGTH, ensure_project_directories, save_json


def threshold_for_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[dict, np.ndarray]:
    predictions = (probabilities >= threshold).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else 0.0,
        "threshold": float(threshold),
        "test_samples": int(len(labels)),
        "fraud_test_samples": int(np.sum(labels)),
    }
    return metrics, predictions


def write_evaluation_artifacts(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    training_samples: int,
    strategy: str,
    history: dict | None = None,
) -> dict:
    ensure_project_directories()
    metrics, predictions = calculate_metrics(labels, probabilities, threshold)
    metrics.update(
        {
            "training_samples": int(training_samples),
            "imbalance_strategy": strategy,
            "sequence_length": SEQUENCE_LENGTH,
        }
    )
    save_json(MODEL_DIR / "evaluation_metrics.json", metrics)
    report = classification_report(
        labels, predictions, labels=[0, 1], target_names=["Genuine", "Fraud"], zero_division=0
    )
    (MODEL_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    if len(np.unique(labels)) == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
        plt.figure(figsize=(6, 5))
        plt.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {metrics['roc_auc']:.4f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=160)
        plt.close()

    if history:
        history_payload = {key: [float(value) for value in values] for key, values in history.items()}
        save_json(MODEL_DIR / "training_history.json", history_payload)
        for metric_name, filename, title in (
            ("loss", "loss_curve.png", "Training and Validation Loss"),
            ("accuracy", "accuracy_curve.png", "Training and Validation Accuracy"),
        ):
            plt.figure(figsize=(7, 4))
            for key in (metric_name, f"val_{metric_name}"):
                if key in history_payload:
                    plt.plot(history_payload[key], label=key.replace("val_", "validation ").replace(metric_name, "training " + metric_name))
            plt.xlabel("Epoch")
            plt.ylabel(metric_name.capitalize())
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / filename, dpi=160)
            plt.close()
    return metrics


def evaluate_saved_model() -> dict:
    ensure_project_directories()
    model_path = MODEL_DIR / "fraud_lstm_model.h5"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found at {model_path}. Run python -m src.train first.")
    frame = load_dataset()
    analyze_dataset(frame)
    prepared = prepare_sequences(frame)
    model = keras.models.load_model(model_path)
    probabilities = model.predict(prepared.x_test, batch_size=1024, verbose=0).ravel()
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return write_evaluation_artifacts(
        prepared.y_test,
        probabilities,
        float(metadata["threshold"]),
        len(prepared.y_train),
        metadata["imbalance_strategy"],
    )


if __name__ == "__main__":
    result = evaluate_saved_model()
    print(json.dumps(result, indent=2))