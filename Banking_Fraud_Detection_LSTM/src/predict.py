"""Score a ten-transaction amount history with the saved model."""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.data_loader import load_dataset
from src.utils import MODEL_DIR, SEQUENCE_LENGTH


def risk_result(probability: float) -> dict[str, str | float]:
    if probability < 0.4:
        level = "LOW"
        recommendation = "Approve transaction and continue routine monitoring."
    elif probability < 0.7:
        level = "MEDIUM"
        recommendation = "Request OTP verification before approving the transaction."
    else:
        level = "HIGH"
        recommendation = "Block transaction and request OTP verification."
    return {"fraud_probability": float(probability), "risk_level": level, "recommendation": recommendation}


def predict_features(transaction_sequence: pd.DataFrame | np.ndarray) -> dict[str, str | float]:
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    expected = metadata["feature_names"]
    rows = transaction_sequence.loc[:, expected].to_numpy(dtype=np.float64) if isinstance(transaction_sequence, pd.DataFrame) else np.asarray(transaction_sequence, dtype=np.float64)
    if rows.shape != (SEQUENCE_LENGTH, len(expected)):
        raise ValueError(f"Expected a {SEQUENCE_LENGTH} x {len(expected)} transaction-feature matrix; got {rows.shape}.")
    scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")
    scaled = scaler.transform(rows).astype(np.float32).reshape(1, SEQUENCE_LENGTH, len(expected))
    model = keras.models.load_model(MODEL_DIR / "fraud_lstm_model.h5")
    probability = float(model.predict(scaled, verbose=0)[0, 0])
    return risk_result(probability)


def predict_amounts(amounts: list[float]) -> dict[str, str | float]:
    if len(amounts) != SEQUENCE_LENGTH:
        raise ValueError(f"Provide exactly {SEQUENCE_LENGTH} transaction amounts.")
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    feature_names = metadata["feature_names"]
    if "Amount" not in feature_names:
        raise ValueError("Saved model has no Amount feature.")
    sequence = pd.DataFrame(np.zeros((SEQUENCE_LENGTH, len(feature_names))), columns=feature_names)
    sequence["Amount"] = amounts
    return predict_features(sequence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amounts", help="Exactly ten comma-separated transaction amounts.")
    args = parser.parse_args()
    model_path = MODEL_DIR / "fraud_lstm_model.h5"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found at {model_path}; run python -m src.train first.")
    if args.amounts:
        amounts = [float(item.strip()) for item in args.amounts.split(",")]
        print(json.dumps(predict_amounts(amounts), indent=2))
    else:
        frame = load_dataset()
        metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
        sequence = frame.tail(SEQUENCE_LENGTH)[metadata["feature_names"]]
        print(json.dumps(predict_features(sequence), indent=2))


if __name__ == "__main__":
    main()