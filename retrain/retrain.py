import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, REFERENCE_DATA_PATH, PERFORMANCE_THRESHOLD, DRIFT_THRESHOLD
from serving.database import SessionLocal, MonitoringReport, PredictionLog, RetrainLog, init_db, get_runtime_config

BACKUP_DIR = "model/backups"
RETRAIN_COOLDOWN_HOURS = 2


def backup_model():
    if not os.path.exists(MODEL_PATH):
        return None
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = f"{BACKUP_DIR}/model_{ts}.joblib"
    model = joblib.load(MODEL_PATH)
    joblib.dump(model, backup_path)
    print(f"Model backed up to {backup_path}")
    return backup_path


def should_retrain() -> tuple[bool, str]:
    db = SessionLocal()
    try:
        # Cooldown: skip if a retrain ran within the last RETRAIN_COOLDOWN_HOURS
        from datetime import timedelta
        latest_retrain = db.query(RetrainLog).order_by(RetrainLog.id.desc()).first()
        if latest_retrain:
            last_ts = datetime.fromisoformat(latest_retrain.timestamp)
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_ts) < timedelta(hours=RETRAIN_COOLDOWN_HOURS):
                remaining = RETRAIN_COOLDOWN_HOURS - (datetime.now(timezone.utc) - last_ts).seconds / 3600
                return False, f"Cooldown active: retrained {last_ts.isoformat()}, {remaining:.1f}h remaining"

        latest = (
            db.query(MonitoringReport)
            .order_by(MonitoringReport.id.desc())
            .first()
        )
        if latest is None:
            return False, "No monitoring reports found"

        drift_threshold = get_runtime_config('drift_threshold', DRIFT_THRESHOLD)
        perf_threshold  = get_runtime_config('performance_threshold', PERFORMANCE_THRESHOLD)

        if latest.drift_score is not None and latest.drift_score > drift_threshold:
            return True, f"Data drift detected: {latest.drift_score:.2%}"

        if latest.performance_score is not None and latest.performance_score < perf_threshold:
            return True, f"Performance below threshold: AUC={latest.performance_score:.4f}"

        return False, "Model performance is acceptable"
    finally:
        db.close()


def load_combined_data() -> tuple[pd.DataFrame, pd.Series]:
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    X_ref = reference.drop(columns=["prediction", "prediction_proba", "target"], errors="ignore")

    if "target" not in reference.columns:
        raise ValueError("Reference data missing 'target' column")
    y_ref = reference["target"].astype(int)

    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(5000).all()
        new_rows = []
        for log in logs:
            features = json.loads(log.features.replace("'", '"'))
            new_rows.append(features)
        if new_rows:
            X_new = pd.DataFrame(new_rows, columns=X_ref.columns)
            X_combined = pd.concat([X_ref, X_new], ignore_index=True)
        else:
            X_combined = X_ref
    finally:
        db.close()

    y_pad = pd.Series([0] * (len(X_combined) - len(y_ref)), dtype=int)
    y_combined = pd.concat([y_ref, y_pad], ignore_index=True).astype(int)

    return X_combined, y_combined


def log_retrain(trigger_reason: str, new_auc: float, backup_path: str, triggered_by: str, status: str):
    db = SessionLocal()
    try:
        record = RetrainLog(
            trigger_reason=trigger_reason,
            new_auc=new_auc,
            backup_path=backup_path,
            triggered_by=triggered_by,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def retrain(force: bool = False, triggered_by: str = "scheduler") -> dict:
    init_db()
    trigger_reason = "Manual forced retrain" if force else ""

    if not force:
        do_retrain, reason = should_retrain()
        if not do_retrain:
            print(f"Retraining not needed: {reason}")
            return {"status": "skipped", "reason": reason}
        trigger_reason = reason

    print(f"Starting retraining. Reason: {trigger_reason}")
    backup_path = backup_model()

    X, y = load_combined_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_res, y_res)

    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"Retrained model ROC-AUC: {auc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"New model saved to {MODEL_PATH}")

    reference = X_train.copy()
    reference["prediction"] = model.predict(X_train)
    reference["prediction_proba"] = model.predict_proba(X_train)[:, 1]
    reference["target"] = y_train.values
    reference.to_csv(REFERENCE_DATA_PATH, index=False)

    result = {
        "status": "retrained",
        "trigger_reason": trigger_reason,
        "new_auc": auc,
        "backup_path": backup_path,
        "triggered_by": triggered_by,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    log_retrain(trigger_reason, auc, backup_path, triggered_by, "retrained")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of metrics")
    args = parser.parse_args()
    result = retrain(force=args.force)
    print(result)
