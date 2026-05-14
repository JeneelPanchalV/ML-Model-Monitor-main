import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DATABASE_URL

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, nullable=True)
    features = Column(Text)
    prediction = Column(Integer)
    probability = Column(Float)
    timestamp = Column(String)


class MonitoringReport(Base):
    __tablename__ = "monitoring_reports"

    id = Column(Integer, primary_key=True, index=True)
    report_type = Column(String)
    drift_detected = Column(Integer, default=0)
    drift_score = Column(Float, nullable=True)
    performance_score = Column(Float, nullable=True)
    concept_drift_detected = Column(Integer, default=0, nullable=True)
    concept_drift_score = Column(Float, nullable=True)
    details = Column(Text, nullable=True)
    html_filename = Column(String, nullable=True)
    timestamp = Column(String)


class RetrainLog(Base):
    __tablename__ = "retrain_logs"

    id = Column(Integer, primary_key=True, index=True)
    trigger_reason = Column(String, nullable=True)
    new_auc = Column(Float, nullable=True)
    backup_path = Column(String, nullable=True)
    triggered_by = Column(String, default="scheduler")
    status = Column(String, default="retrained")
    timestamp = Column(String)


class ConfigStore(Base):
    __tablename__ = "config_store"

    key = Column(String, primary_key=True)
    value = Column(String)


def get_runtime_config(key: str, default: float) -> float:
    db = SessionLocal()
    try:
        row = db.query(ConfigStore).filter(ConfigStore.key == key).first()
        return float(row.value) if row else default
    finally:
        db.close()


def set_runtime_config(key: str, value: float):
    db = SessionLocal()
    try:
        row = db.query(ConfigStore).filter(ConfigStore.key == key).first()
        if row:
            row.value = str(value)
        else:
            db.add(ConfigStore(key=key, value=str(value)))
        db.commit()
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        for stmt in [
            "ALTER TABLE monitoring_reports ADD COLUMN html_filename VARCHAR",
            "ALTER TABLE monitoring_reports ADD COLUMN concept_drift_detected INTEGER",
            "ALTER TABLE monitoring_reports ADD COLUMN concept_drift_score REAL",
            "ALTER TABLE retrain_logs ADD COLUMN triggered_by VARCHAR",
        ]:
            try:
                conn.execute(__import__("sqlalchemy").text(stmt))
                conn.commit()
            except Exception:
                pass
