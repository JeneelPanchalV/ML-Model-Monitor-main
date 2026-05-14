import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timezone
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DRIFT_THRESHOLD, PERFORMANCE_THRESHOLD,
    ALERT_EMAIL, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD,
)
from serving.database import SessionLocal, MonitoringReport


def send_email_alert(subject: str, body: str):
    if not ALERT_EMAIL or not SMTP_USER or not SMTP_PASSWORD:
        print(f"[ALERT - email not configured] {subject}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = ALERT_EMAIL
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, ALERT_EMAIL, msg.as_string())
        print(f"Alert email sent to {ALERT_EMAIL}: {subject}")
    except Exception as e:
        print(f"Failed to send alert email: {e}")


def build_alert_body(alert_type: str, details: dict) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in details.items())
    return f"""
    <html><body>
    <h2>ML Model Monitor — {alert_type} Alert</h2>
    <p>Triggered at: {ts}</p>
    <table border="1" cellpadding="6">
      <tr><th>Metric</th><th>Value</th></tr>
      {rows}
    </table>
    <p>Please investigate and consider retraining the model.</p>
    </body></html>
    """


def check_and_alert(monitoring_result: dict):
    alerts_triggered = []

    drift_share = monitoring_result.get("drift_share", 0.0)
    if drift_share is not None and drift_share > DRIFT_THRESHOLD:
        alerts_triggered.append("Data Drift")
        send_email_alert(
            subject=f"[ML Monitor] Data Drift Detected ({drift_share:.2%})",
            body=build_alert_body("Data Drift", {
                "Drift Share": f"{drift_share:.2%}",
                "Threshold": f"{DRIFT_THRESHOLD:.2%}",
                "Drifted Features": monitoring_result.get("drifted_features", "N/A"),
            }),
        )

    perf = monitoring_result.get("performance") or {}
    roc_auc = perf.get("roc_auc")
    if roc_auc is not None and roc_auc < PERFORMANCE_THRESHOLD:
        alerts_triggered.append("Performance Degradation")
        send_email_alert(
            subject=f"[ML Monitor] Performance Degradation (AUC={roc_auc:.4f})",
            body=build_alert_body("Performance Degradation", {
                "ROC-AUC": f"{roc_auc:.4f}",
                "Threshold": f"{PERFORMANCE_THRESHOLD:.4f}",
                "F1 Score": perf.get("f1", "N/A"),
            }),
        )

    if not alerts_triggered:
        print("No alerts triggered.")

    return alerts_triggered


def get_recent_alerts(limit: int = 20) -> list:
    db = SessionLocal()
    try:
        reports = (
            db.query(MonitoringReport)
            .filter(MonitoringReport.drift_detected == 1)
            .order_by(MonitoringReport.id.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": r.id,
                "report_type": r.report_type,
                "drift_score": r.drift_score,
                "performance_score": r.performance_score,
                "details": json.loads(r.details) if r.details else {},
                "timestamp": r.timestamp,
            }
            for r in reports
        ]
    finally:
        db.close()
