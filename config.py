import os
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 8000))

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./monitor.db")

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/reference.csv")

MONITORING_INTERVAL_MINUTES = int(os.getenv("MONITORING_INTERVAL_MINUTES", 60))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", 0.05))
PERFORMANCE_THRESHOLD = float(os.getenv("PERFORMANCE_THRESHOLD", 0.80))

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
