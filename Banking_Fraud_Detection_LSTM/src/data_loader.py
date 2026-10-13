"""Load the Kaggle credit-card fraud dataset and summarize its quality."""

from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path

import pandas as pd

from src.utils import DATA_DIR, PROJECT_ROOT, ensure_project_directories, save_json

DATASET_SLUG = "mlg-ulb/creditcardfraud"
DATASET_URL = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"


def _download_dataset() -> Path:
    ensure_project_directories()
    kaggle_cli = shutil.which("kaggle")
    if kaggle_cli is None:
        local_cli = PROJECT_ROOT / ".python-packages" / "bin" / "kaggle"
        kaggle_cli = str(local_cli) if local_cli.is_file() else None
    if kaggle_cli is None:
        raise FileNotFoundError("Kaggle CLI is unavailable; install requirements.txt and retry.")
    try:
        subprocess.run(
            [
                kaggle_cli,
                "datasets",
                "download",
                "-d",
                DATASET_SLUG,
                "-p",
                str(DATA_DIR),
                "--unzip",
            ],
            check=True,
            cwd=PROJECT_ROOT,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise FileNotFoundError(
            "creditcard.csv is not present and Kaggle could not download it. "
            f"Accept the dataset terms at {DATASET_URL}, configure Kaggle API "
            "credentials (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json), "
            f"then retry: python -m src.train. You can also place {CSV_NAME} in {DATA_DIR}."
        ) from error

    csv_path = DATA_DIR / CSV_NAME
    if not csv_path.is_file():
        raise FileNotFoundError(f"Kaggle download completed but {csv_path} was not found.")
    return csv_path


def load_dataset(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read a local CSV, downloading it from Kaggle when API access is configured."""
    candidates = [Path(csv_path)] if csv_path else [DATA_DIR / CSV_NAME, PROJECT_ROOT / "data" / CSV_NAME]
    resolved = next((path for path in candidates if path.is_file()), None)
    if resolved is None:
        resolved = _download_dataset()

    frame = pd.read_csv(resolved)
    required = {"Time", "Amount", "Class"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Dataset must include {sorted(required)}; found {list(frame.columns)}")
    if not set(frame["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Class must contain only binary labels 0 (genuine) and 1 (fraud).")
    return frame


def analyze_dataset(frame: pd.DataFrame, save: bool = True) -> dict:
    """Calculate and optionally persist missing-value and class statistics."""
    class_counts = frame["Class"].value_counts().sort_index()
    fraud_count = int(class_counts.get(1, 0))
    summary = {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "feature_columns": [column for column in frame.columns if column != "Class"],
        "missing_values": {key: int(value) for key, value in frame.isna().sum().items()},
        "statistics": frame.describe().to_dict(),
        "class_distribution": {str(key): int(value) for key, value in class_counts.items()},
        "fraud_count": fraud_count,
        "fraud_percentage": float(fraud_count / len(frame) * 100) if len(frame) else 0.0,
    }
    if save:
        ensure_project_directories()
        save_json(DATA_DIR / "dataset_summary.json", summary)
    return summary


if __name__ == "__main__":
    dataset = load_dataset()
    stats = analyze_dataset(dataset)
    print(f"Rows: {stats['rows']:,} | fraud: {stats['fraud_count']:,} ({stats['fraud_percentage']:.4f}%)")
    print(f"Missing values: {sum(stats['missing_values'].values()):,}")