🚀 Live Demo: [Live Demo](https://ml-model-monitor.onrender.com)

# ML Model Monitor

![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-live-green)
[![Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?logo=render&logoColor=white)](https://ml-model-monitor.onrender.com)

A production-grade ML monitoring system for credit card fraud detection. Automatically detects data drift, concept drift, and performance degradation — then retrains the model without human intervention.

Built to simulate what a real ML platform team would deploy to keep a live fraud model healthy over time.

---

## What I Built

This is a full end-to-end MLOps system — not just a model, but the entire infrastructure around keeping a model production-ready over time. Here is everything I designed and implemented:

**Fraud Detection Model**
Trained an XGBoost classifier on 284,807 real credit card transactions. Handled extreme class imbalance (0.17% fraud rate) using SMOTE oversampling on the training split. Applied StandardScaler to the Amount feature. Achieved AUC-ROC of 0.9993 on a time-ordered held-out eval set.

**Three-Signal Monitoring Pipeline**
Built a monitoring engine that runs every 60 minutes and checks three independent signals: data drift across all 29 input features (using the Kolmogorov-Smirnov test via Evidently AI), concept drift in the model output distribution (K-S test on fraud probability scores), and performance degradation against a fixed benchmark eval set. Each signal is independent — catching different failure modes that the others would miss.

**Auto-Retrain Pipeline**
When monitoring detects a problem, the system automatically backs up the current model, retrains from scratch with SMOTE, evaluates the new model, hot-swaps it into production, and logs everything to the database — all without human intervention. A 2-hour cooldown prevents retrain thrashing.

**Persistent PostgreSQL Backend**
All prediction logs, monitoring reports, retrain history, and runtime config are stored in a managed PostgreSQL database on Render. Data persists across every deployment and restart.

**7-Page Live Dashboard**
Built a single-page application in vanilla JavaScript with Chart.js that visualises everything in real time — drift trends, feature-level K-S scores, live prediction feed, full monitoring report history, threshold controls, retrain audit log, and alert severity tracking.

**Production Deployment**
Dockerised the entire stack and deployed to Render with a managed PostgreSQL instance. The API serves predictions at ~120ms latency. Auto-deploys from GitHub on every push.

---

## The Problem This Solves

Every year, financial institutions lose billions of dollars to credit card fraud. Machine learning models are the frontline defence — but they have a silent enemy: **model decay**.

When a fraud model is deployed to production, the world keeps changing. Fraudsters adapt their attack patterns. Customer spending behaviour shifts with seasons and economic conditions. New card products change transaction distributions. The model, frozen at its training snapshot, slowly becomes blind to these changes — and no one notices until fraud losses spike.

Most teams discover model degradation the wrong way: through business metrics, customer complaints, or a quarterly audit. By then, thousands of fraudulent transactions have already slipped through.

**This project solves that problem with a fully automated monitoring and self-healing pipeline:**

- Detects the moment data distributions start to shift — before model accuracy drops
- Flags when the model's output behaviour changes, even when inputs look normal
- Measures real model performance against a fixed benchmark on every cycle
- Automatically retrains and redeploys the model the moment it degrades past a threshold
- Gives every stakeholder a live dashboard with zero manual inspection required

---



## Live Metrics

| Metric | Value |
|---|---|
| **Model AUC-ROC** | 0.9993 |
| **Precision** | 1.00 |
| **Recall** | 0.98 |
| **F1 Score** | 0.99 |
| **Fraud Detection Rate** | 22% |
| **Features Monitored** | 29 |
| **API Latency** | ~120ms |

---

## Architecture

**Stack:** Python 3.12 · XGBoost · FastAPI · Evidently AI · PostgreSQL · SQLAlchemy · APScheduler · Chart.js · Docker · Render

| Layer | Component | Details |
|---|---|---|
| **API** | FastAPI | Serves predictions, monitoring, retrain, config endpoints on port 8000 |
| **Scheduler** | APScheduler | Runs full monitoring cycle every 60 minutes automatically |
| **Dashboard** | 7-page SPA | Chart.js frontend — Overview, Drift, Predictions, Reports, Thresholds, Retrain, Alerts |
| **Database** | PostgreSQL (Render) | Persists prediction_logs, monitoring_reports, retrain_logs, config_store |
| **Hosting** | Render | Docker-based deployment with managed PostgreSQL |

---

## Features

| Feature | Details |
|---|---|
| **Fraud detection model** | XGBoost classifier trained on 284K transactions, AUC-ROC 0.9993, 29 PCA features |
| **Data drift detection** | Kolmogorov–Smirnov test on all 29 input features via Evidently AI; full HTML report per cycle |
| **Concept drift detection** | K-S test comparing reference vs production output probability distributions |
| **Performance monitoring** | AUC-ROC, precision, recall, and F1 scored against a fixed held-out eval set every cycle |
| **Auto-retrain pipeline** | Triggers when drift or AUC crosses threshold; 2-hour cooldown; each retrain is backed up and logged |
| **Live dashboard** | 7-page SPA — Overview, Drift Analysis, Predictions, Reports, Thresholds, Auto-retrain, Alerts |
| **Runtime config** | Drift, AUC, and concept drift thresholds adjustable from dashboard with no restart required |
| **Persistent storage** | PostgreSQL on Render — all predictions, reports, and retrain logs persist across deployments |
| **Model versioning** | Every retrain saves a timestamped backup and logs trigger reason, new AUC, and backup path |
| **REST API** | Full API for predictions, monitoring, retraining, config, and model metadata |

---

## Quick Start (Local)

**Prerequisites:** Docker and Docker Compose installed.

```bash
# 1. Clone the repo
git clone https://github.com/JeneelPanchalV/ML-Model-Monitor-main.git
cd ML-Model-Monitor-main

# 2. Add the dataset
# Download creditcard.csv from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Place it at: data/creditcard.csv

# 3. Start the app
docker compose up -d

# 4. Open the dashboard
open http://localhost:8000
```

Trigger a monitoring cycle immediately:

```bash
curl -X POST http://localhost:8000/monitor
```

Simulate predictions:

```bash
python data/simulate_predictions.py
```

---

## Dashboard Pages

### 1. Overview
The home screen of the monitoring system. Displays live status pills in the top-right corner showing API health, model load state, active drift alerts, and current latency. The centrepiece is a 3D rotating wireframe globe with animated node labels (XGBoost Model, FastAPI Server, Drift Monitor, Alert Engine) representing the live system components.

Below the globe, a scrolling ticker shows real-time stats: predictions monitored, drift percentage, and alerts fired. Four metric cards update every 30 seconds:
- **Total Predictions** — cumulative count logged to PostgreSQL with a live delta indicator
- **Drift Share** — percentage of the 29 features currently drifting, with an Above/Below threshold badge
- **Fraud Caught** — number of high-confidence fraud flags in the current monitoring window
- **Model AUC-ROC** — live score computed against the held-out eval set, with a health status label

A data drift over time chart and a feature drift scores bar chart (K-S test per feature, colour-coded red for drifted / teal for stable) sit side by side below the cards. A live prediction feed table at the bottom auto-refreshes showing the most recent transactions.

---

### 2. Drift Analysis
A dedicated deep-dive page for understanding model and data health over time. Split into two monitoring signals:

**Feature Drift (Data Drift)**
A line chart plots the K-S drift score for each monitoring cycle against the configurable 5% threshold line. When the drift score breaches the threshold, the chart area turns amber and a Breach Detected badge appears. Below the chart, a full history table lists every monitoring report with columns for timestamp, drift score, drifted feature count, concept drift KS stat, AUC-ROC, and detection status.

**Concept Drift — Output Distribution**
Four stat cards show the KS statistic, detection status, reference fraud rate, and production fraud rate side by side. A grouped bar chart beneath compares the reference and production output probability distributions across 10 bins (0.0 to 1.0). A large gap between the two distributions — especially at the high-probability end — indicates the model is assigning fraud probabilities differently than it did at training time, which is a strong signal of concept drift even when input features appear stable.

---

### 3. Live Predictions
The full prediction log streamed from the inference server. Four summary cards at the top show total predictions, fraud count, legit count, and fraud rate. Three filter buttons (All / Fraud / Legit) instantly filter the table below.

Each row in the table shows:
- **#** — prediction sequence number
- **Probability** — raw fraud probability score from the model (0.0 to 1.0)
- **Label** — risk tier (very low / low / medium / high / critical) based on probability bands
- **Verdict** — Fraud (red badge) or Legit (green badge)
- **Confidence** — how certain the model is, expressed as a percentage
- **Time** — timestamp of the prediction

High-probability fraud rows are highlighted in dark red for instant visual identification.

---

### 4. Monitoring Reports
An audit trail of every drift analysis cycle run by the system. A sparkline chart at the top shows drift scores across all historical reports at a glance. Three summary cards show total reports generated, how many detected drift, and the average drift score.

The main table lists every report with date, time, drift score badge, and status. Each row has a **View** button that opens the full interactive Evidently AI HTML report in a new tab — this report contains per-feature histograms, K-S statistics, p-values, and distribution comparisons for all 29 features. A **Run New Check** button in the top-right triggers an immediate monitoring cycle without waiting for the scheduler.

---

### 5. Thresholds
Runtime configuration page for all three monitoring thresholds. No restart or redeploy is needed — changes are written to PostgreSQL immediately and the scheduler picks them up on its next cycle.

Three sliders:
- **Drift Threshold (1%–50%)** — fraction of features that must drift before an alert is fired and retrain is considered. Default: 5%
- **Performance Threshold (0.50–0.99)** — minimum AUC-ROC below which a retrain is triggered. Default: 0.80
- **Concept Drift Threshold (0.01–0.50)** — KS statistic cutoff for flagging output distribution shift. Default: 0.10

Current values are loaded from the database on page open. A Save button writes all three values atomically. The page also shows the current live values returned by `GET /config` so you can confirm the update took effect.

---

### 6. Auto-Retrain
Visibility into the automated retraining pipeline. A summary card at the top shows the most recent retrain event: timestamp, trigger reason (drift / performance / manual), new AUC-ROC achieved, and whether it was triggered automatically by the scheduler or manually via the dashboard.

The full retrain history table below is colour-coded — amber rows for manual retrains, teal for automatic. Each row shows the trigger reason, new AUC, backup file path, triggered-by field, status, and timestamp.

A **Trigger Manual Retrain** button forces an immediate retrain regardless of cooldown or threshold state. A progress bar polls the backend every 2 seconds until the background job completes, then refreshes the history table automatically.

---

### 7. Alerts
A chronological log of all monitoring cycles where drift or performance thresholds were breached. Cards are sorted newest first and colour-coded by severity — High (drift score above 50%) and Medium (drift score above threshold but below 50%).

Each alert card shows the timestamp, drift score, number of drifted features, concept drift status, and AUC-ROC at the time of detection. Hovering a card expands a detail panel with the full breakdown. This page gives a quick audit view of how often the system has flagged issues and whether they are increasing or decreasing over time.

---

## How the Monitoring Loop Works

Every 60 minutes the scheduler runs a full monitoring cycle:

**Step 1 — Data Drift (Evidently AI)**
Load reference.csv and the last 1000 production predictions. Run K-S test on all 29 input features. Save HTML report. drift_score = fraction of features that drifted.

**Step 2 — Concept Drift**
Compare reference prediction_proba distribution vs last 1000 production probability scores using K-S test. concept_drift_score = KS statistic.

**Step 3 — Performance Check**
Load data/eval.csv (fixed held-out set). Run current model, compute AUC-ROC, precision, recall, F1. performance_score = AUC-ROC.

**Step 4 — Save to PostgreSQL**
All results written to monitoring_reports table.

**Step 5 — Alert**
Fire alert if drift or performance threshold is breached.

**Step 6 — Retrain (if needed)**
If drift detected and 2-hour cooldown has elapsed: backup current model, combine reference + last 5000 prediction logs, SMOTE oversampling, train new XGBoost (200 estimators, depth 6), evaluate, save to models/model.joblib (API hot-reloads), write to retrain_logs.

---

## Project Structure

```
ML-Model-Monitor-main/
├── model/
│   ├── train.py                  # Training — StandardScaler, SMOTE, XGBoost
│   └── generate_reference.py     # Generates synthetic reference data from model
├── models/
│   └── model.joblib              # Active production model
├── serving/
│   ├── app.py                    # FastAPI app — all endpoints
│   └── database.py               # SQLAlchemy ORM + PostgreSQL config
├── monitoring/
│   ├── monitor.py                # Drift + concept drift + performance monitoring
│   └── scheduler.py              # APScheduler — runs cycle, alerts, retrain
├── retrain/
│   └── retrain.py                # Full retrain pipeline
├── alerting/
│   └── alerts.py                 # Email alerts via SMTP
├── data/
│   ├── creditcard.csv            # Source dataset (not committed — download from Kaggle)
│   ├── eval.csv                  # Fixed held-out eval set
│   └── simulate_predictions.py   # Sends batches of predictions to the API
├── dashboard/
│   └── index.html                # 7-page SPA — Chart.js
├── config.py                     # Env-based config
├── Dockerfile                    # Docker image
└── requirements.txt
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API status and model loaded state |
| `POST` | `/predict` | Run fraud inference — pass features array of 29 floats |
| `GET` | `/predictions/recent?limit=N` | Last N prediction logs |
| `GET` | `/monitoring/reports?limit=N` | Full monitoring history |
| `POST` | `/monitor` | Trigger a full monitoring cycle immediately |
| `GET` | `/retrain/history?limit=N` | Retrain event log |
| `POST` | `/retrain` | Force a manual retrain |
| `GET` | `/model/info` | Live model metadata — AUC, precision, recall, F1 |
| `GET` | `/config` | Current runtime thresholds |
| `PATCH` | `/config` | Update thresholds at runtime |
| `GET` | `/reports/{filename}` | Serve Evidently HTML drift report |

### Example: run a prediction

```bash
curl -X POST https://ml-model-monitor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.09, 0.36, 0.09,
                    -0.55, -0.62, -0.99, -0.31, 1.47, -0.47, 0.21, 0.02, 0.40, 0.25,
                    -0.02, 0.28, -0.11, 0.07, 0.13, -0.19, 0.13, -0.02, 0.15]}'
```

### Example: trigger monitoring

```bash
curl -X POST https://ml-model-monitor.onrender.com/monitor
```

### Example: update thresholds

```bash
curl -X PATCH https://ml-model-monitor.onrender.com/config \
  -H "Content-Type: application/json" \
  -d '{"drift_threshold": 0.10, "performance_threshold": 0.85}'
```

---

## Configuration

Copy `.env.example` to `.env`:

```env
APP_HOST=0.0.0.0
APP_PORT=8000
DATABASE_URL=postgresql://user:pass@host/dbname
MODEL_PATH=models/model.joblib
REFERENCE_DATA_PATH=data/reference.csv
MONITORING_INTERVAL_MINUTES=60
DRIFT_THRESHOLD=0.05
PERFORMANCE_THRESHOLD=0.80
ALERT_EMAIL=you@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=you@gmail.com
SMTP_PASSWORD=your-app-password
```

Thresholds set in `.env` are startup defaults. Once running, use `PATCH /config` or the dashboard Thresholds page to change them at runtime — values are stored in PostgreSQL and override env defaults without a restart.

---

## Dataset

[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 European credit card transactions from September 2013, of which 492 (0.17%) are fraudulent. Features V1–V28 are PCA-transformed to protect cardholder privacy. Class imbalance handled with SMOTE during training.

---

<div align="center">

**ML Model Monitor** · Built by [Jeneel Panchal](https://github.com/JeneelPanchalV) · Deployed on [Render](https://ml-model-monitor.onrender.com)

![ML Model Monitor](https://img.shields.io/badge/ML%20Model%20Monitor-Production%20MLOps-46E3B7?style=for-the-badge&logo=render&logoColor=white)

</div>
