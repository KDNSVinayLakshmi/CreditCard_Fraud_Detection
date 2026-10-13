"""Chronological train/test splitting, feature scaling, and sequence creation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import RANDOM_SEED, SEQUENCE_LENGTH


@dataclass
class PreparedData:
    feature_names: list[str]
    scaler: StandardScaler
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_boundary: int


def prepare_sequences(
    frame: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    test_fraction: float = 0.2,
    seed: int = RANDOM_SEED,
) -> PreparedData:
    """Create rolling windows ending at each target transaction, split by time."""
    del seed  # Chronological splitting avoids leaking future transactions into training.
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    if len(frame) <= sequence_length:
        raise ValueError(f"At least {sequence_length + 1} rows are required to make sequences.")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; impute or remove them before training.")

    ordered = frame.sort_values("Time", kind="stable").reset_index(drop=True)
    feature_names = [column for column in ordered.columns if column != "Class"]
    features = ordered[feature_names].to_numpy(dtype=np.float64)
    labels = ordered["Class"].to_numpy(dtype=np.int32)
    train_boundary = int(len(ordered) * (1 - test_fraction))
    if train_boundary <= sequence_length or train_boundary >= len(ordered):
        raise ValueError("The split leaves too few transactions for training or testing.")

    scaler = StandardScaler()
    scaler.fit(features[:train_boundary])
    scaled = scaler.transform(features).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(
        scaled, window_shape=sequence_length, axis=0
    ).transpose(0, 2, 1)
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    sequence_labels = labels[sequence_length - 1 :]
    target_rows = np.arange(sequence_length - 1, len(ordered))
    train_mask = target_rows < train_boundary
    test_mask = ~train_mask

    return PreparedData(
        feature_names=feature_names,
        scaler=scaler,
        x_train=windows[train_mask],
        y_train=sequence_labels[train_mask],
        x_test=windows[test_mask],
        y_test=sequence_labels[test_mask],
        train_boundary=train_boundary,
    )