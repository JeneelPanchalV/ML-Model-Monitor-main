import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler

API_URL = "http://localhost:8000/predict"
DATA_PATH = "data/creditcard.csv"
N = 50

df = pd.read_csv(DATA_PATH)

# Match preprocessing from model/train.py: scale Amount, drop Time
df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
df.drop(columns=["Time"], inplace=True)

fraud = df[df["Class"] == 1].sample(10, random_state=42)
normal = df[df["Class"] == 0].sample(40, random_state=42)
sample = pd.concat([fraud, normal]).sample(frac=1, random_state=42).reset_index(drop=True)

feature_cols = [c for c in df.columns if c != "Class"]

total_sent = 0
fraud_detected = 0
normal_detected = 0

print(f"{'ID':<6} {'True Label':<12} {'Prediction':<12} {'Probability':<12}")
print("-" * 44)

for i, row in sample.iterrows():
    features = row[feature_cols].tolist()
    payload = {"features": features, "request_id": f"sim-{i:04d}"}

    try:
        resp = requests.post(API_URL, json=payload, timeout=5)
        resp.raise_for_status()
        result = resp.json()

        pred_label = "FRAUD" if result["is_fraud"] else "Normal"
        true_label = "FRAUD" if row["Class"] == 1 else "Normal"
        print(f"{i:<6} {true_label:<12} {pred_label:<12} {result['probability']:.4f}")

        total_sent += 1
        if result["is_fraud"]:
            fraud_detected += 1
        else:
            normal_detected += 1

    except Exception as e:
        print(f"{i:<6} ERROR: {e}")

print("-" * 44)
print(f"\nSummary:")
print(f"  Total sent:       {total_sent}")
print(f"  Fraud detected:   {fraud_detected}")
print(f"  Normal:           {normal_detected}")
