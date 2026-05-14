import os
import sys
import json
import ast
import pandas as pd
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REFERENCE_DATA_PATH, DRIFT_THRESHOLD
from serving.database import SessionLocal, PredictionLog

from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
REPORTS_DIR = "monitoring_reports"


def pull_predictions(limit=100) -> pd.DataFrame:
    db = SessionLocal()
    try:
        logs = (
            db.query(PredictionLog)
            .order_by(PredictionLog.id.desc())
            .limit(limit)
            .all()
        )
        rows = []
        for log in logs:
            features = ast.literal_eval(log.features)
            rows.append(dict(zip(FEATURE_COLS, features)))
        return pd.DataFrame(rows)
    finally:
        db.close()


def run_drift_check():
    reference = pd.read_csv(REFERENCE_DATA_PATH)[FEATURE_COLS]
    current = pull_predictions(100)

    if current.empty:
        print("No prediction data found in the database.")
        return

    print(f"Reference rows : {len(reference):,}")
    print(f"Current rows   : {len(current)}")

    ref_ds = Dataset.from_pandas(reference, data_definition=DataDefinition())
    cur_ds = Dataset.from_pandas(current, data_definition=DataDefinition())
    report = Report([DataDriftPreset()]).run(ref_ds, cur_ds)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    html_path = f"{REPORTS_DIR}/drift_{ts}.html"
    json_path = f"{REPORTS_DIR}/drift_{ts}.json"

    report.save_html(html_path)

    metrics = report.dict()["metrics"]

    # Overall drift count/share
    summary_metric = next(
        (m for m in metrics if "DriftedColumnsCount" in m["metric_name"]), None
    )
    drift_share = summary_metric["value"]["share"] if summary_metric else 0.0
    drifted_count = int(summary_metric["value"]["count"]) if summary_metric else 0
    drift_detected = drift_share > DRIFT_THRESHOLD

    # Per-feature drift (ValueDrift metrics: p-value < threshold means drift)
    drifted_features = []
    feature_details = {}
    for m in metrics:
        col = m["config"].get("column")
        threshold = m["config"].get("threshold")
        if col and threshold is not None:
            p_value = float(m["value"])
            is_drifted = p_value < threshold
            feature_details[col] = {"p_value": round(p_value, 4), "drifted": is_drifted}
            if is_drifted:
                drifted_features.append(col)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "drift_detected": drift_detected,
        "drift_share": round(drift_share, 4),
        "drifted_features_count": drifted_count,
        "total_features": len(FEATURE_COLS),
        "drifted_features": drifted_features,
        "feature_details": feature_details,
        "html_report": html_path,
        "json_report": json_path,
    }

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 52}")
    print(f"  DRIFT DETECTION SUMMARY")
    print(f"{'=' * 52}")
    print(f"  Drift Detected   : {'YES ⚠' if drift_detected else 'NO ✓'}")
    print(f"  Drift Share      : {drift_share:.1%}")
    print(f"  Drifted Features : {drifted_count} / {len(FEATURE_COLS)}")
    if drifted_features:
        print(f"  Features Drifted : {', '.join(drifted_features)}")
    print(f"  HTML Report      : {html_path}")
    print(f"  JSON Report      : {json_path}")
    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    run_drift_check()
