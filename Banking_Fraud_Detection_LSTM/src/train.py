"""Train, compare imbalance strategies, select, and evaluate an LSTM model."""

from __future__ import annotations

import json

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.data_loader import analyze_dataset, load_dataset
from src.evaluate import threshold_for_f1, write_evaluation_artifacts
from src.model import build_model
from src.preprocess import prepare_sequences
from src.utils import MODEL_DIR, RANDOM_SEED, SEQUENCE_LENGTH, ensure_project_directories, save_json, set_random_seeds

EPOCHS = 10
BATCH_SIZE = 32


def _fit_candidate(name: str, x_fit: np.ndarray, y_fit: np.ndarray,
                   x_validation: np.ndarray, y_validation: np.ndarray,
                   feature_count: int, class_weight: dict[int, float] | None = None) -> tuple[keras.Model, dict, float]:
    keras.backend.clear_session()
    set_random_seeds(RANDOM_SEED)
    model = build_model(SEQUENCE_LENGTH, feature_count)
    checkpoint = MODEL_DIR / "checkpoints" / f"{name}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(checkpoint), monitor="val_loss", save_best_only=True, verbose=0
        ),
    ]
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
        shuffle=False,
        class_weight=class_weight,
    )
    best_model = keras.models.load_model(checkpoint)
    validation_probabilities = best_model.predict(x_validation, batch_size=1024, verbose=0).ravel()
    threshold = threshold_for_f1(y_validation, validation_probabilities)
    predictions = validation_probabilities >= threshold
    from sklearn.metrics import f1_score, roc_auc_score

    validation_metrics = {
        "f1_score": float(f1_score(y_validation, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_validation, validation_probabilities)),
        "threshold": float(threshold),
        "epochs_trained": len(history.history["loss"]),
    }
    return best_model, history.history, validation_metrics


def train_pipeline() -> dict:
    ensure_project_directories()
    set_random_seeds(RANDOM_SEED)
    frame = load_dataset()
    summary = analyze_dataset(frame)
    print(
        f"Dataset: {summary['rows']:,} transactions, "
        f"{summary['fraud_count']:,} fraud ({summary['fraud_percentage']:.4f}%)."
    )
    if sum(summary["missing_values"].values()):
        raise ValueError("Missing values are present; clean the dataset before training.")

    prepared = prepare_sequences(frame)
    validation_start = max(1, int(len(prepared.x_train) * 0.9))
    x_fit = prepared.x_train[:validation_start]
    y_fit = prepared.y_train[:validation_start]
    x_validation = prepared.x_train[validation_start:]
    y_validation = prepared.y_train[validation_start:]
    if not len(x_validation) or len(np.unique(y_validation)) < 2:
        raise ValueError("The chronological validation window must contain both classes.")

    class_labels = np.unique(y_fit)
    weights = compute_class_weight(class_weight="balanced", classes=class_labels, y=y_fit)
    class_weights = {int(label): float(weight) for label, weight in zip(class_labels, weights)}
    candidates = {}
    candidates["class_weights"] = _fit_candidate(
        "class_weights", x_fit, y_fit, x_validation, y_validation,
        len(prepared.feature_names), class_weight=class_weights,
    )

    flattened_fit = x_fit.reshape(len(x_fit), -1)
    smote = SMOTE(sampling_strategy=0.1, random_state=RANDOM_SEED, k_neighbors=5)
    smoted_x, smoted_y = smote.fit_resample(flattened_fit, y_fit)
    smoted_x = np.asarray(smoted_x, dtype=np.float32).reshape(-1, SEQUENCE_LENGTH, len(prepared.feature_names))
    candidates["smote"] = _fit_candidate(
        "smote", smoted_x, np.asarray(smoted_y, dtype=np.int32), x_validation, y_validation,
        len(prepared.feature_names)
    )

    comparison = {
        name: {"validation": result[2], "training_samples": int(len(y_fit) if name == "class_weights" else len(smoted_y))}
        for name, result in candidates.items()
    }
    selected_name = max(candidates, key=lambda name: (candidates[name][2]["f1_score"], candidates[name][2]["roc_auc"]))
    selected_model, history, selected_validation = candidates[selected_name]
    model_path = MODEL_DIR / "fraud_lstm_model.h5"
    selected_model.save(model_path)
    metadata = {
        "project": "Banking_Credit_Card_Fraud_Detection_Using_LSTM",
        "sequence_length": SEQUENCE_LENGTH,
        "feature_names": prepared.feature_names,
        "threshold": selected_validation["threshold"],
        "imbalance_strategy": selected_name,
        "random_seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "train_test_split": 0.8,
    }
    save_json(MODEL_DIR / "model_metadata.json", metadata)
    save_json(MODEL_DIR / "strategy_comparison.json", comparison)
    import joblib

    joblib.dump(prepared.scaler, MODEL_DIR / "feature_scaler.joblib")

    test_probabilities = selected_model.predict(prepared.x_test, batch_size=1024, verbose=0).ravel()
    metrics = write_evaluation_artifacts(
        prepared.y_test,
        test_probabilities,
        selected_validation["threshold"],
        len(prepared.y_train),
        selected_name,
        history,
    )
    metrics["validation_f1_score"] = selected_validation["f1_score"]
    save_json(MODEL_DIR / "evaluation_metrics.json", metrics)
    return metrics


if __name__ == "__main__":
    print(json.dumps(train_pipeline(), indent=2))