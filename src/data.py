"""Data loading and feature engineering for the AI4I 2020 dataset.

Shared by the training script and the Streamlit app so that the exact same
feature engineering is applied at train and inference time.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ai4i2020.csv"

TARGET = "Machine failure"
TYPE_COL = "Type"

# Raw sensor measurements
RAW_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Engineered, physics-informed features
ENGINEERED = [
    "Temp difference [K]",   # process - air; relates to heat-dissipation failures
    "Power [W]",             # mechanical power = torque * angular velocity
    "Strain [min*Nm]",       # tool wear * torque; relates to overstrain failures
]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED
CATEGORICAL_FEATURES = [TYPE_COL]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Specific failure-mode flags (kept for analysis, not used as model inputs)
FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

_DROP = ["UDI", "Product ID"]


def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV (run src/download_data.py first if it is missing)."""
    return pd.read_csv(path)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-informed engineered features. Works on full or single-row frames."""
    df = df.copy()
    df["Temp difference [K]"] = (
        df["Process temperature [K]"] - df["Air temperature [K]"]
    )
    # Mechanical power: torque [Nm] * angular velocity [rad/s] = torque * rpm * 2pi/60
    df["Power [W]"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * math.pi / 60)
    df["Strain [min*Nm]"] = df["Tool wear [min]"] * df["Torque [Nm]"]
    return df


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model feature matrix X and target y from an engineered frame."""
    df = df.drop(columns=[c for c in _DROP if c in df.columns], errors="ignore")
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def load_dataset(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience: load -> engineer -> split into X, y."""
    df = engineer_features(load_raw(path))
    return get_xy(df)
