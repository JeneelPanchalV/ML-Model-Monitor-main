import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apscheduler.schedulers.blocking import BlockingScheduler
from config import MONITORING_INTERVAL_MINUTES
from monitoring.monitor import run_monitoring_cycle
from alerting.alerts import check_and_alert
from retrain.retrain import retrain

scheduler = BlockingScheduler()


def monitoring_job():
    result = run_monitoring_cycle()
    if result.get("status") == "completed":
        check_and_alert(result)
        if result.get("drift_detected"):
            print("Drift detected — triggering automatic retraining...")
            retrain_result = retrain(triggered_by="scheduler")
            print(f"Retrain result: {retrain_result}")


if __name__ == "__main__":
    print(f"Starting monitoring scheduler (every {MONITORING_INTERVAL_MINUTES} minutes)...")
    scheduler.add_job(monitoring_job, "interval", minutes=MONITORING_INTERVAL_MINUTES, id="monitor")
    scheduler.start()
