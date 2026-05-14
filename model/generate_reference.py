import os
import numpy as np
import pandas as pd
import joblib
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, REFERENCE_DATA_PATH

def generate_reference(n_samples=5000, seed=42):
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    feature_names = model.get_booster().feature_names
    np.random.seed(seed)

    print(f"Generating {n_samples} synthetic reference samples...")
    # Generate realistic synthetic features
    data = {}
    for feat in feature_names:
        if feat == "Amount":
            data[feat] = np.abs(np.random.exponential(scale=88, size=n_samples))
        else:
            data[feat] = np.random.normal(0, 1, size=n_samples)

    df = pd.DataFrame(data)

    # Get model predictions
    predictions = model.predict(df)
    probas = model.predict_proba(df)[:, 1]

    # Add required columns
    df["prediction"] = predictions
    df["prediction_proba"] = probas
    df["target"] = predictions  # use predicted as proxy target

    # Save
    os.makedirs(os.path.dirname(REFERENCE_DATA_PATH), exist_ok=True)
    df.to_csv(REFERENCE_DATA_PATH, index=False)
    print(f"Reference data saved to {REFERENCE_DATA_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    generate_reference()
