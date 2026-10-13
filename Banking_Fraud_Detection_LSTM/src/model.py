"""Keras LSTM architecture for rolling transaction windows."""

from __future__ import annotations

from tensorflow import keras


def build_model(sequence_length: int, feature_count: int) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(sequence_length, feature_count)),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.Dropout(0.3),
            keras.layers.LSTM(32),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="banking_fraud_lstm",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model