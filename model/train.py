import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, REFERENCE_DATA_PATH

DATA_PATH = "data/creditcard.csv"
SCALER_PATH = "model/scaler.joblib"


def load_and_preprocess(path: str):
    df = pd.read_csv(path)
    df["Amount"] = StandardScaler().fit_transform(df[["Amount"]])
    df.drop(columns=["Time"], inplace=True)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y, df


def train():
    print("Loading data...")
    X, y, df = load_and_preprocess(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    print("Training XGBoost model...")
    model.fit(X_resampled, y_resampled)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(REFERENCE_DATA_PATH), exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    reference = X_train.copy()
    reference["prediction"] = model.predict(X_train)
    reference["prediction_proba"] = model.predict_proba(X_train)[:, 1]
    reference["target"] = y_train.values
    reference.to_csv(REFERENCE_DATA_PATH, index=False)
    print(f"Reference data saved to {REFERENCE_DATA_PATH}")

    return model, auc


if __name__ == "__main__":
    train()