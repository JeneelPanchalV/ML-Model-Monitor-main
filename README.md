🚀 Live Demo: [Live Demo](https://ml-model-monitor.onrender.com)



# ML Model Monitor
![Python](https://img.shields.io/badge/Python-3.14-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-live-green)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://ml-model-monitor.onrender.com)

A production-grade ML monitoring system for credit card fraud detection. Automatically detects data drift, concept drift, and performance degradation — then retrains the model without human intervention.

Built to simulate what a real ML platform team would deploy to keep a live fraud model healthy over time.

---

## What This Project Does

When a machine learning model is deployed to production, it degrades over time. The real-world data it sees slowly shifts away from what it was trained on — this is called **drift**. Without monitoring, you won't know the model is silently making worse predictions until it causes real business damage.

This system solves that by:

1. **Continuously serving predictions** via a REST API
2. **Monitoring every hour** for three types of degradation: data drift, concept drift, and performance drop
3. **Automatically retraining** the model when a problem is detected
4. **Visualising everything** in a live dashboard — no manual inspection needed

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Docker Compose                            │
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │     API     │   │  Scheduler   │   │      Simulator       │  │
│  │  FastAPI    │   │ APScheduler  │   │  Streams batches of  │  │
│  │  port 8000  │   │  every 60min │   │  transactions with   │  │
│  │             │   │              │   │  incremental drift   │  │
│  │  /predict   │   │  → monitor   │   └──────────┬───────────┘  │
│  │  /monitor   │   │  → alert     │              │ POST /predict │
│  │  /retrain   │   │  → retrain   │              │              │
│  │  /config    │   │              │              │              │
│  └──────┬──────┘   └──────┬───────┘              │              │
│         │                 │                       │              │
│         └─────────────────┴───────────────────────┘             │
│                           │                                      │
│               ┌───────────▼────────────┐                        │
│               │       monitor.db        │                        │
│               │   SQLite (shared vol)   │                        │
│               │                        │                        │
│               │  • prediction_logs      │                        │
│               │  • monitoring_reports   │                        │
│               │  • retrain_logs         │                        │
│               │  • config_store         │                        │
│               └────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────┘
```

**Stack:** Python 3.11 · XGBoost · FastAPI · Evidently AI · SQLite · SQLAlchemy · APScheduler · Chart.js · Docker Compose

---

## Features

| Feature | Details |
|---|---|
| **Fraud detection model** | XGBoost classifier trained on 284K transactions, AUC-ROC 0.965, 29 PCA features |
| **Data drift detection** | Kolmogorov–Smirnov test on all input features via Evidently AI; full HTML report saved per cycle |
| **Concept drift detection** | K-S test comparing reference vs production output probability distributions; tracks fraud rate and mean confidence shifts |
| **Performance monitoring** | AUC-ROC, precision, recall, and F1 scored against a fixed held-out eval set every monitoring cycle |
| **Auto-retrain pipeline** | Triggers when drift or AUC crosses threshold; 2-hour cooldown prevents thrashing; each retrain is backed up and logged |
| **Live dashboard** | 7-page single-page app — Overview, Drift Analysis, Predictions, Reports, Thresholds, Auto-retrain, Alerts |
| **Runtime config** | Drift, AUC, and concept drift thresholds adjustable from the dashboard with no restart required |
| **Prediction simulator** | Docker service that continuously streams transactions with slowly increasing feature drift to keep the system exercised |
| **Model versioning** | Every retrain saves a timestamped backup and logs trigger reason, new AUC, and backup path to the database |
| **REST API** | Full API for predictions, monitoring, retraining, config, and model metadata |

---

## Quick Start

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Clone the repo
git clone https://github.com/JeneelPanchalV/ML-Model-Monitor-.git
cd ML-Model-Monitor-

# 2. Add the dataset
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it at: data/creditcard.csv

# 3. Train the initial model
docker compose run --rm api python model/train.py

# 4. Start all services
docker compose up -d

# 5. Open the dashboard
open http://localhost:8000
```

The first monitoring cycle runs automatically after 60 minutes. To trigger one immediately:

```bash
curl -X POST http://localhost:8000/monitor
```

---

## Dashboard Pages

### Overview
The home screen. Shows live API and model health pills, a 3D rotating wireframe globe (purely decorative), and four metric cards: total predictions, drift share, fraud caught, and live model AUC-ROC. The three insight cards below update dynamically — current drift status, model performance assessment, and last retrain info. A live feed table shows the 4 most recent predictions.

### Drift Analysis
Two monitoring signals side by side:

- **Feature drift (data drift)** — line chart of K-S drift score over time vs the 5% threshold. Table shows every monitoring report with drift score, concept drift KS stat, AUC-ROC, and status.
- **Concept drift** — four stat cards (KS statistic, detection status, reference vs production fraud rate) plus a grouped bar chart comparing the reference and production output probability distributions. This catches cases where input features look fine but model outputs have shifted.

### Live Predictions
Full prediction log with probability score, risk label, verdict (Fraud/Legit), and confidence. Filter buttons to view All / Fraud only / Legit only. Four summary stats: total, fraud count, legit count, fraud rate.

### Monitoring Reports
Sparkline chart of all drift scores over time, plus a table with a **View ↗** button on every row — clicking opens the full Evidently AI HTML report (interactive feature-by-feature drift breakdown) in a new tab.

### Thresholds
Three live sliders:
- **Drift threshold** (1%–50%) — fraction of features drifted to trigger alert/retrain
- **Performance threshold** (0.50–0.99) — minimum AUC-ROC before retrain triggers
- **Concept drift threshold** (0.01–0.50) — KS statistic cutoff for output distribution alert

Saving writes to the database instantly. The scheduler picks up the new values on its next cycle — no restart needed.

### Auto-Retrain
Summary card showing the last retrain: timestamp, trigger reason, new AUC, and whether it was manual or automatic. Full history table below, colour-coded amber for manual and teal for auto. The **Trigger Manual Retrain** button runs a forced retrain immediately with a progress bar that polls until the background job completes.

### Alerts
All monitoring cycles where drift was detected, sorted newest first, with severity badges (High / Medium based on drift score). Hover a card to expand the detail panel.

---

## How the Monitoring Loop Works

Every 60 minutes the scheduler runs a full monitoring cycle:

```
APScheduler fires
    │
    ▼
run_monitoring_cycle()
    │
    ├─ 1. DATA DRIFT (Evidently AI)
    │      Load reference.csv + last 1000 production predictions
    │      Run K-S test on all 29 input features
    │      Save HTML report to monitoring_reports/
    │      → drift_score = fraction of features that drifted
    │
    ├─ 2. CONCEPT DRIFT
    │      Compare reference prediction_proba distribution
    │      vs last 1000 production probability scores
    │      K-S test on the two distributions
    │      → concept_drift_score = KS statistic
    │
    ├─ 3. PERFORMANCE CHECK
    │      Load data/eval.csv (fixed 572-row held-out set)
    │      Run current model, compute AUC-ROC, precision, recall, F1
    │      → performance_score = AUC-ROC
    │
    └─ 4. Save everything to monitoring_reports table in DB
    │
    ▼
check_and_alert()
    └─ Send email if drift or performance threshold breached
    │
    ▼
drift_detected AND cooldown elapsed (2h)?
    ├── No  → done
    └── Yes → retrain()
                  ├── Backup current model with timestamp
                  ├── Combine reference.csv + last 5000 prediction logs
                  ├── SMOTE oversampling to handle class imbalance
                  ├── Train new XGBoost (200 estimators, depth 6)
                  ├── Evaluate on test split, log new AUC
                  ├── Save to models/model.joblib (API hot-reloads on file change)
                  ├── Update reference.csv with new training data
                  └── Write to retrain_logs table
```

---

## Drift Explained

**Data drift** (input drift) — the statistical distribution of input features changes. For example, transaction amounts or PCA-transformed card behaviour patterns shift because of seasonality, a new product launch, or a change in fraud attack patterns. Detected using the Kolmogorov–Smirnov test on each of the 29 features. Evidently AI produces a full HTML report with per-feature histograms and K-S statistics.

**Concept drift** (output drift) — the relationship between inputs and the correct output changes, or the model's output distribution shifts. Even if input features look the same, the model might start assigning different probability scores to similar transactions. This project detects it by comparing the distribution of fraud probability scores from the reference period against recent production scores using a K-S test. A shift here often precedes a drop in real-world accuracy.

**Performance degradation** — measured directly against a fixed held-out test set (`data/eval.csv`) that was never used in training. This gives a ground-truth signal about whether the model is still accurate, independent of what the production labels are.

---

## The Eval Set

`data/eval.csv` is the last 15% of `creditcard.csv` sorted by transaction time — a true time-ordered hold-out that was never touched during training or used as reference data. It contains 572 rows with 52 fraud cases (9.1%). Because it is fixed and never updated (unlike `reference.csv` which is refreshed after each retrain), it provides a stable, unbiased performance benchmark across all model versions.

---

## Project Structure

```
ml-model-monitor/
├── model/
│   ├── train.py              # Initial training — StandardScaler, SMOTE, XGBoost, 80/20 split
│   └── backups/              # Timestamped model snapshots saved before each retrain
├── models/
│   └── model.joblib          # Active production model (API hot-reloads on file change)
├── serving/
│   ├── app.py                # FastAPI app — all endpoints, model loading, hot-reload
│   └── database.py           # SQLAlchemy ORM models + runtime config helpers
├── monitoring/
│   ├── monitor.py            # Data drift (Evidently) + concept drift (K-S) + performance (AUC)
│   └── scheduler.py          # APScheduler — runs cycle, triggers alerts and retrain
├── retrain/
│   └── retrain.py            # Full retrain pipeline — cooldown check, SMOTE, backup, DB log
├── alerting/
│   └── alerts.py             # Email alerts via SMTP — configurable in .env
├── data/
│   ├── creditcard.csv        # Source dataset (not committed — download from Kaggle)
│   ├── reference.csv         # Reference distribution — updated after each retrain
│   ├── eval.csv              # Fixed held-out eval set — never updated
│   └── stream_predictions.py # Simulator script — sends batches with incremental drift
├── dashboard/
│   └── index.html            # 7-page SPA — Chart.js, all data from live API
├── config.py                 # Env-based config with DB override support
├── docker-compose.yml        # api · scheduler · simulator · dashboard services
├── Dockerfile                # Single image used by all services
└── requirements.txt
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Returns API status and whether model is loaded |
| `POST` | `/predict` | Run fraud inference — pass `features` array of 29 floats |
| `GET` | `/predictions/recent?limit=N` | Last N prediction logs from the database |
| `GET` | `/monitoring/reports?limit=N` | Full monitoring history with drift, concept drift, AUC, filenames |
| `POST` | `/monitor` | Trigger a full monitoring cycle immediately (synchronous) |
| `GET` | `/retrain/history?limit=N` | Retrain event log — AUC, trigger reason, backup path, timestamp |
| `POST` | `/retrain` | Force a manual retrain in the background |
| `GET` | `/model/info` | Live model metadata — type, version, trained date, AUC, precision, recall, F1 |
| `GET` | `/config` | Current runtime thresholds (drift, AUC, concept drift) |
| `PATCH` | `/config` | Update thresholds at runtime — persisted to DB, no restart needed |
| `GET` | `/reports` | List all Evidently HTML report filenames |
| `GET` | `/reports/{filename}` | Serve a specific Evidently HTML drift report |

### Example: run a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.09, 0.36, 0.09,
                    -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.02, 0.40, 0.25,
                    -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02, 0.15]}'
```

### Example: update thresholds

```bash
curl -X PATCH http://localhost:8000/config \
  -H "Content-Type: application/json" \
  -d '{"drift_threshold": 0.10, "performance_threshold": 0.85, "concept_drift_threshold": 0.15}'
```

---

## Configuration

Copy `.env.example` to `.env` and fill in values:

```env
# App
APP_HOST=0.0.0.0
APP_PORT=8000

# Database
DATABASE_URL=sqlite:///./monitor.db

# Model paths
MODEL_PATH=models/model.joblib
REFERENCE_DATA_PATH=data/reference.csv

# Monitoring
MONITORING_INTERVAL_MINUTES=60
DRIFT_THRESHOLD=0.05
PERFORMANCE_THRESHOLD=0.80

# Email alerts (optional — leave blank to disable)
ALERT_EMAIL=you@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
```

Thresholds set in `.env` are the startup defaults. Once the system is running, use `PATCH /config` or the dashboard Thresholds page to change them at runtime — these values are stored in the database and override the env defaults without requiring a restart or redeploy.

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European credit card transactions from September 2013, of which 492 (0.17%) are fraudulent.

Features V1–V28 are the result of PCA transformation applied to protect cardholder privacy. `Amount` is the transaction amount (standardized during preprocessing). `Time` is dropped. The target label `Class` is 1 for fraud, 0 for legitimate.

The extreme class imbalance (0.17% fraud) is handled during training with SMOTE oversampling on the training split.
# ML-Model-Monitor
