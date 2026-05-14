import os
import json
import joblib
import pandas as pd
from datetime import datetime, timezone
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, REFERENCE_DATA_PATH, DRIFT_THRESHOLD, PERFORMANCE_THRESHOLD
from serving.database import SessionLocal, MonitoringReport, PredictionLog, init_db, get_runtime_config

REPORTS_DIR = "monitoring_reports"
EVAL_DATA_PATH = "data/eval.csv"
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def load_reference_data() -> pd.DataFrame:
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError(f"Reference data not found at {REFERENCE_DATA_PATH}")
    return pd.read_csv(REFERENCE_DATA_PATH)


def get_current_data(limit: int = 1000) -> pd.DataFrame | None:
    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
        if not logs:
            return None
        rows = []
        for log in logs:
            features = json.loads(log.features.replace("'", '"'))
            row = dict(zip(FEATURE_COLS, features))
            row["probability"] = log.probability
            rows.append(row)
        return pd.DataFrame(rows)
    finally:
        db.close()


def run_performance_check() -> dict:
    if not os.path.exists(EVAL_DATA_PATH):
        return {"error": "eval.csv not found"}
    if not os.path.exists(MODEL_PATH):
        return {"error": "model not found"}

    eval_df = pd.read_csv(EVAL_DATA_PATH)
    X_eval = eval_df[FEATURE_COLS]
    y_eval = eval_df["target"].astype(int)

    model = joblib.load(MODEL_PATH)
    y_proba = model.predict_proba(X_eval)[:, 1]
    y_pred  = model.predict(X_eval)

    auc       = float(roc_auc_score(y_eval, y_proba))
    precision = float(precision_score(y_eval, y_pred, zero_division=0))
    recall    = float(recall_score(y_eval, y_pred, zero_division=0))
    f1        = float(f1_score(y_eval, y_pred, zero_division=0))

    return {
        "auc_roc":   auc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "n_samples": len(y_eval),
        "n_fraud":   int(y_eval.sum()),
    }


def run_concept_drift_check(reference: pd.DataFrame) -> dict:
    from scipy import stats as _stats

    ref_proba = reference["prediction_proba"].dropna().values
    if len(ref_proba) < 50:
        return {"error": "insufficient_reference_data"}

    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(1000).all()
        if not logs or len(logs) < 50:
            return {"error": "insufficient_production_data"}
        prod_proba = [l.probability for l in logs]
    finally:
        db.close()

    import numpy as _np
    prod_proba = _np.array(prod_proba)

    ks_stat, p_value = _stats.ks_2samp(ref_proba, prod_proba)

    ref_fraud_rate  = float((ref_proba  >= 0.5).mean())
    prod_fraud_rate = float((prod_proba >= 0.5).mean())
    ref_mean_proba  = float(ref_proba.mean())
    prod_mean_proba = float(prod_proba.mean())

    concept_drift_threshold = get_runtime_config("concept_drift_threshold", 0.10)
    concept_drift_detected  = bool(ks_stat > concept_drift_threshold)

    # Build histogram bins for dashboard chart (10 equal buckets 0-1)
    bins = _np.linspace(0, 1, 11)
    ref_hist,  _ = _np.histogram(ref_proba,  bins=bins, density=True)
    prod_hist, _ = _np.histogram(prod_proba, bins=bins, density=True)

    return {
        "concept_drift_detected": concept_drift_detected,
        "ks_statistic":           float(ks_stat),
        "p_value":                float(p_value),
        "ref_fraud_rate":         ref_fraud_rate,
        "prod_fraud_rate":        prod_fraud_rate,
        "ref_mean_proba":         ref_mean_proba,
        "prod_mean_proba":        prod_mean_proba,
        "ref_hist":               ref_hist.tolist(),
        "prod_hist":              prod_hist.tolist(),
        "hist_bins":              [f"{b:.1f}" for b in bins[:-1]],
        "n_reference":            len(ref_proba),
        "n_production":           len(prod_proba),
    }


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    ref_features = reference[FEATURE_COLS]
    cur_features = current[[c for c in FEATURE_COLS if c in current.columns]]

    ref_ds = Dataset.from_pandas(ref_features, data_definition=DataDefinition())
    cur_ds = Dataset.from_pandas(cur_features, data_definition=DataDefinition())

    snapshot = Report([DataDriftPreset()]).run(ref_ds, cur_ds)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    html_path = f"{REPORTS_DIR}/drift_{ts}.html"
    snapshot.save_html(html_path)

    metrics = snapshot.dict()["metrics"]
    drift_summary = next(
        (m for m in metrics if "DriftedColumnsCount" in m["metric_name"]), None
    )
    drift_share = drift_summary["value"]["share"] if drift_summary else 0.0
    drifted_count = int(drift_summary["value"]["count"]) if drift_summary else 0
    drift_threshold = get_runtime_config('drift_threshold', DRIFT_THRESHOLD)
    drift_detected = drift_share > drift_threshold

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "drifted_features": drifted_count,
        "total_features": len(FEATURE_COLS),
        "html_report": html_path,
    }


def save_report(report_type: str, drift_detected: bool, drift_score: float,
                performance_score: float, details: dict, html_filename: str = None,
                concept_drift_detected: bool = False, concept_drift_score: float = None):
    db = SessionLocal()
    try:
        record = MonitoringReport(
            report_type=report_type,
            drift_detected=int(drift_detected),
            drift_score=drift_score,
            performance_score=performance_score,
            concept_drift_detected=int(concept_drift_detected),
            concept_drift_score=concept_drift_score,
            details=json.dumps(details),
            html_filename=html_filename,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def run_monitoring_cycle() -> dict:
    print(f"[{datetime.now(timezone.utc).isoformat()}] Running monitoring cycle...")
    init_db()

    reference = load_reference_data()
    current = get_current_data()

    if current is None or len(current) < 50:
        print("Not enough current data for monitoring (need ≥50 rows).")
        return {"status": "skipped", "reason": "insufficient_data"}

    drift_result   = run_drift_report(reference, current)
    perf_result    = run_performance_check()
    concept_result = run_concept_drift_check(reference)

    auc = perf_result.get("auc_roc") if "error" not in perf_result else None
    print(f"Performance: AUC={auc:.4f}" if auc else f"Performance check skipped: {perf_result.get('error')}")

    cd_detected = concept_result.get("concept_drift_detected", False)
    cd_score    = concept_result.get("ks_statistic")
    if "error" not in concept_result:
        print(f"Concept drift: KS={cd_score:.4f} detected={cd_detected}")
    else:
        print(f"Concept drift check skipped: {concept_result.get('error')}")

    html_filename = os.path.basename(drift_result["html_report"]) if drift_result.get("html_report") else None
    save_report(
        report_type="drift",
        drift_detected=drift_result["drift_detected"],
        drift_score=drift_result["drift_share"],
        performance_score=auc,
        details={**drift_result, "performance": perf_result, "concept_drift": concept_result},
        html_filename=html_filename,
        concept_drift_detected=cd_detected,
        concept_drift_score=cd_score,
    )

    summary = {
        "status": "completed",
        "drift_detected": drift_result["drift_detected"],
        "drift_share": drift_result["drift_share"],
        "drifted_features": drift_result["drifted_features"],
        "total_features": drift_result["total_features"],
        "html_report": drift_result["html_report"],
        "performance": perf_result,
        "concept_drift": concept_result,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    print(f"Monitoring result: {summary}")
    return summary


if __name__ == "__main__":
    run_monitoring_cycle()
