"""
Continuous prediction streamer.
Sends batches of transactions to /predict every INTERVAL seconds.
Introduces gradual feature drift over time to simulate real distribution shift.
"""
import os
import sys
import time
import random
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

API_URL      = os.getenv("API_URL", "http://localhost:8000")
INTERVAL     = int(os.getenv("STREAM_INTERVAL", 20))   # seconds between batches
BATCH_SIZE   = int(os.getenv("STREAM_BATCH", 15))       # predictions per batch
DRIFT_SPEED  = float(os.getenv("DRIFT_SPEED", 0.002))  # drift increment per batch
DATA_PATH    = os.getenv("DATA_PATH", "data/creditcard.csv")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Features that will drift (PCA components most predictive of fraud)
DRIFT_FEATURES = ["V4", "V11", "V14", "V17", "V18", "V24"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
    df.drop(columns=["Time"], inplace=True)
    return df


def send_batch(df: pd.DataFrame, drift_offset: float, batch_num: int):
    # Sample a realistic fraud ratio (~0.2% fraud)
    n_fraud  = max(0, int(BATCH_SIZE * random.uniform(0.0, 0.04)))
    n_normal = BATCH_SIZE - n_fraud

    fraud_rows  = df[df["Class"] == 1].sample(min(n_fraud,  len(df[df["Class"]==1])), replace=True)
    normal_rows = df[df["Class"] == 0].sample(min(n_normal, len(df[df["Class"]==0])), replace=True)
    batch = pd.concat([fraud_rows, normal_rows]).sample(frac=1).reset_index(drop=True)

    sent = ok = 0
    for _, row in batch.iterrows():
        features = row[FEATURE_COLS].tolist()

        # Apply incremental drift to selected features
        for feat in DRIFT_FEATURES:
            idx = FEATURE_COLS.index(feat)
            features[idx] += drift_offset + np.random.normal(0, 0.1)

        # Small random noise on all features
        features = [f + np.random.normal(0, 0.05) for f in features]

        payload = {
            "features": features,
            "request_id": f"stream-{batch_num:05d}-{sent:03d}",
        }
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if r.status_code == 200:
                ok += 1
        except Exception:
            pass
        sent += 1

    return sent, ok


def main():
    print(f"Loading data from {DATA_PATH}…")
    df = load_data()
    print(f"Loaded {len(df)} rows. Streaming to {API_URL} every {INTERVAL}s (batch={BATCH_SIZE})")

    drift_offset = 0.0
    batch_num    = 0
    total_sent   = 0

    while True:
        sent, ok = send_batch(df, drift_offset, batch_num)
        total_sent  += ok
        drift_offset += DRIFT_SPEED
        batch_num    += 1

        print(f"[batch {batch_num:4d}] sent={ok}/{sent}  drift_offset={drift_offset:.3f}  total={total_sent}")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
