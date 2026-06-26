"""Download the AI4I 2020 Predictive Maintenance dataset from the UCI repository.

Run: python src/download_data.py
Saves data/ai4i2020.csv
"""
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = DATA_DIR / "ai4i2020.csv"


def download() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if OUT.exists():
        print(f"Already present: {OUT}")
        return OUT
    print("Downloading AI4I 2020 dataset from UCI ...")
    resp = requests.get(URL, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        with zf.open(csv_name) as f:
            df = pd.read_csv(f)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows -> {OUT}")
    return OUT


if __name__ == "__main__":
    path = download()
    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print(df.head(3).to_string())
