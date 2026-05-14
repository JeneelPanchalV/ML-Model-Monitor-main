import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_PATH, APP_HOST, APP_PORT
from serving.database import SessionLocal, PredictionLog, MonitoringReport, RetrainLog, init_db, get_runtime_config, set_runtime_config

app = FastAPI(title="Fraud Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_mtime = None


def load_model():
    global model, model_mtime
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run model/train.py first.")
    model = joblib.load(MODEL_PATH)
    model_mtime = os.path.getmtime(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")


def maybe_reload_model():
    global model, model_mtime
    try:
        current_mtime = os.path.getmtime(MODEL_PATH)
        if model_mtime is None or current_mtime > model_mtime:
            print("Model file updated — reloading...")
            load_model()
    except OSError:
        pass


@app.on_event("startup")
def startup():
    init_db()
    load_model()


class PredictionRequest(BaseModel):
    features: List[float]
    request_id: Optional[str] = None


class PredictionResponse(BaseModel):
    request_id: Optional[str]
    prediction: int
    probability: float
    is_fraud: bool
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/model/info")
def model_info():
    import joblib as _joblib
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    import pandas as _pd

    info = {"loaded": model is not None}
    if model is None:
        return info

    # Basic model metadata
    info["type"]         = type(model).__name__
    info["n_features"]   = int(model.n_features_in_)
    info["feature_names"]= model.get_booster().feature_names

    # Model file modification time = last training date
    try:
        mtime = os.path.getmtime(MODEL_PATH)
        info["trained_at"] = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        info["trained_at"] = None

    # Retrain count = version
    db = SessionLocal()
    try:
        from serving.database import RetrainLog
        count = db.query(RetrainLog).count()
        info["version"] = f"v{count + 1}"
        latest_retrain  = db.query(RetrainLog).order_by(RetrainLog.id.desc()).first()
        info["last_retrain_auc"] = latest_retrain.new_auc if latest_retrain else None
    finally:
        db.close()

    # Live performance on eval set
    eval_path = "data/eval.csv"
    if os.path.exists(eval_path):
        try:
            eval_df = _pd.read_csv(eval_path)
            feat_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
            X = eval_df[feat_cols]
            y = eval_df["target"].astype(int)
            y_proba = model.predict_proba(X)[:, 1]
            y_pred  = model.predict(X)
            info["auc_roc"]   = float(roc_auc_score(y, y_proba))
            info["precision"] = float(precision_score(y, y_pred, zero_division=0))
            info["recall"]    = float(recall_score(y, y_pred, zero_division=0))
            info["f1"]        = float(f1_score(y, y_pred, zero_division=0))
        except Exception as e:
            info["perf_error"] = str(e)

    return info


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    maybe_reload_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    expected_features = model.n_features_in_
    if len(request.features) != expected_features:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {expected_features} features, got {len(request.features)}",
        )

    feature_names = model.get_booster().feature_names
    X = pd.DataFrame([request.features], columns=feature_names)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    timestamp = datetime.now(timezone.utc).isoformat()

    db = SessionLocal()
    try:
        log = PredictionLog(
            request_id=request.request_id,
            features=str(request.features),
            prediction=pred,
            probability=proba,
            timestamp=timestamp,
        )
        db.add(log)
        db.commit()
    finally:
        db.close()

    return PredictionResponse(
        request_id=request.request_id,
        prediction=pred,
        probability=proba,
        is_fraud=bool(pred),
        timestamp=timestamp,
    )


@app.get("/predictions/recent")
def recent_predictions(limit: int = 100):
    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
        return [
            {
                "id": l.id,
                "request_id": l.request_id,
                "prediction": l.prediction,
                "probability": l.probability,
                "timestamp": l.timestamp,
            }
            for l in logs
        ]
    finally:
        db.close()


@app.get("/monitoring/reports")
def monitoring_reports(limit: int = 50):
    import json as _json
    db = SessionLocal()
    try:
        rows = db.query(MonitoringReport).order_by(MonitoringReport.id.desc()).limit(limit).all()
        reports_dir = "monitoring_reports"
        result = []
        for r in rows:
            details = {}
            try:
                details = _json.loads(r.details) if r.details else {}
            except Exception:
                pass
            cd = details.get("concept_drift", {})
            result.append({
                "id": r.id,
                "report_type": r.report_type,
                "drift_detected": bool(r.drift_detected),
                "drift_score": r.drift_score,
                "drifted_features": details.get("drifted_features"),
                "total_features": details.get("total_features", 29),
                "performance_score": r.performance_score,
                "concept_drift_detected": bool(r.concept_drift_detected),
                "concept_drift_score": r.concept_drift_score,
                "concept_drift_details": cd if "error" not in cd else None,
                "timestamp": r.timestamp,
                "filename": r.html_filename or None,
                "has_report": bool(r.html_filename and os.path.exists(os.path.join(reports_dir, r.html_filename))),
            })
        return result
    finally:
        db.close()


@app.post("/monitor")
def trigger_monitor():
    try:
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from monitoring.monitor import run_monitoring_cycle
        result = run_monitoring_cycle()
        return {"status": "completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain")
def trigger_retrain():
    try:
        import threading
        import sys as _sys
        import os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
        from retrain.retrain import retrain
        def run():
            retrain(force=True, triggered_by="manual")
        threading.Thread(target=run, daemon=True).start()
        return {"status": "started", "message": "Retrain triggered in background"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/retrain/history")
def retrain_history(limit: int = 20):
    db = SessionLocal()
    try:
        logs = db.query(RetrainLog).order_by(RetrainLog.id.desc()).limit(limit).all()
        return [
            {
                "id": l.id,
                "trigger_reason": l.trigger_reason,
                "new_auc": l.new_auc,
                "backup_path": l.backup_path,
                "triggered_by": l.triggered_by,
                "status": l.status,
                "timestamp": l.timestamp,
            }
            for l in logs
        ]
    finally:
        db.close()


@app.get("/config")
def get_config():
    from config import DRIFT_THRESHOLD, PERFORMANCE_THRESHOLD
    return {
        "drift_threshold":          get_runtime_config("drift_threshold",          DRIFT_THRESHOLD),
        "performance_threshold":    get_runtime_config("performance_threshold",    PERFORMANCE_THRESHOLD),
        "concept_drift_threshold":  get_runtime_config("concept_drift_threshold",  0.10),
    }


@app.patch("/config")
def update_config(payload: dict):
    allowed = {"drift_threshold", "performance_threshold", "concept_drift_threshold"}
    updated = {}
    for key, val in payload.items():
        if key not in allowed:
            raise HTTPException(status_code=400, detail=f"Unknown config key: {key}")
        try:
            fval = float(val)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail=f"Value for {key} must be a number")
        set_runtime_config(key, fval)
        updated[key] = fval
    return {"status": "updated", "config": updated}


@app.get("/reports")
def list_reports():
    reports_dir = "monitoring_reports"
    if not os.path.exists(reports_dir):
        return {"reports": []}
    files = [f for f in os.listdir(reports_dir) if f.endswith(".html")]
    return {"reports": sorted(files, reverse=True)}


@app.get("/reports/{filename}")
def get_report(filename: str):
    report_path = os.path.join("monitoring_reports", filename)
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    return FileResponse(report_path, media_type="text/html")


@app.get("/")
async def serve_dashboard():
    return FileResponse("dashboard/index.html")


app.mount("/static", StaticFiles(directory="dashboard"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serving.app:app", host=APP_HOST, port=APP_PORT, reload=True)
