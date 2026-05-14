import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serving.database import SessionLocal, MonitoringReport, PredictionLog, init_db
from config import APP_PORT, DRIFT_THRESHOLD, PERFORMANCE_THRESHOLD

API_URL = os.getenv("API_URL", f"http://localhost:{APP_PORT}")
REPORTS_DIR = "monitoring_reports"

st.set_page_config(
    page_title="ML Model Monitor",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

init_db()

if "page" not in st.session_state:
    st.session_state.page = "Overview"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    header { visibility: hidden; }
    [data-testid="collapsedControl"] { visibility: visible !important; }
    [data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }

    .block-container { padding: 2rem 2rem 2rem !important; max-width: 100% !important; }

    /* Header title — force visible over any Streamlit theme override */
    .main-title { color: #111111 !important; font-size: 24px !important; font-weight: 600 !important; }
    [data-testid="stMarkdownContainer"] .main-title { color: #111111 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #f8fafc; border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebarContent"] { padding: 1rem 0.75rem; }
    [data-testid="stSidebar"] .stButton > button {
        background: transparent; border: none; width: 100%; text-align: left;
        padding: 0.45rem 0.75rem; border-radius: 6px; font-size: 0.875rem;
        color: #64748b; font-weight: 500; cursor: pointer; transition: all 0.15s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #e2e8f0; color: #1e293b;
    }

    /* Metric cards */
    .metric-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.25rem 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.75rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.9rem; font-weight: 800; color: #0f172a; line-height: 1.1;
    }
    .metric-value.red { color: #dc2626; }
    .metric-delta {
        font-size: 0.78rem; color: #64748b; margin-top: 4px;
    }

    /* Status pills */
    .pill {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 600;
    }
    .pill-green { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
    .pill-red   { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
    .pill-orange{ background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
    .pill-gray  { background: #f1f5f9; color: #64748b; border: 1px solid #e2e8f0; }
    .pill-dot   { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }

    /* Verdict badges */
    .badge-fraud { background: #FAECE7; color: #712B13; padding: 3px 10px;
                   border-radius: 999px; font-size: 0.72rem; font-weight: 700; }
    .badge-legit { background: #E1F5EE; color: #085041; padding: 3px 10px;
                   border-radius: 999px; font-size: 0.72rem; font-weight: 700; }

    /* Nav section label */
    .nav-section {
        font-size: 0.68rem; font-weight: 700; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.1em;
        padding: 0.5rem 0.75rem 0.25rem; margin-top: 0.5rem;
    }

    /* Active nav button */
    .nav-active > button {
        background: #eff6ff !important; color: #1d4ed8 !important;
        font-weight: 600 !important; border-left: 3px solid #1d4ed8 !important;
        border-radius: 0 6px 6px 0 !important;
    }

    /* Model info card */
    .model-card {
        background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 0.875rem 1rem; margin-top: 1rem;
    }
    .model-card-title { font-size: 0.7rem; font-weight: 700; color: #94a3b8;
                        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .model-card-row { display: flex; justify-content: space-between;
                      font-size: 0.75rem; margin-bottom: 4px; }
    .model-card-key { color: #64748b; }
    .model-card-val { color: #1e293b; font-weight: 600; }

    /* Section headers */
    .section-title {
        font-size: 0.875rem; font-weight: 700; color: #1e293b;
        margin: 1.5rem 0 0.75rem; padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Insight cards */
    .insight { border-radius: 10px; padding: 0.875rem 1rem;
               border: 1px solid; margin-bottom: 0; }
    .insight-warn  { background: #fff7ed; border-color: #fed7aa; }
    .insight-ok    { background: #f0fdf4; border-color: #bbf7d0; }
    .insight-info  { background: #eff6ff; border-color: #bfdbfe; }
    .insight-title { font-size: 0.78rem; font-weight: 700; margin-bottom: 3px; }
    .insight-title-warn { color: #c2410c; }
    .insight-title-ok   { color: #15803d; }
    .insight-title-info { color: #1d4ed8; }
    .insight-body  { font-size: 0.75rem; color: #475569; line-height: 1.5; }

    /* Prediction table */
    .pred-table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
    .pred-table th {
        text-align: left; font-size: 0.68rem; font-weight: 700; color: #94a3b8;
        text-transform: uppercase; letter-spacing: 0.07em;
        padding: 0 0.75rem 0.5rem; border-bottom: 1px solid #e2e8f0;
    }
    .pred-table td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #f1f5f9;
                     color: #1e293b; vertical-align: middle; }
    .pred-table tr:last-child td { border-bottom: none; }
    .prob-bar-wrap { display: flex; align-items: center; gap: 8px; }
    .prob-bar-bg { background: #f1f5f9; border-radius: 999px; height: 6px;
                  width: 80px; overflow: hidden; flex-shrink: 0; }
    .prob-bar-fill { height: 100%; border-radius: 999px; }

    /* Chart containers */
    .chart-card {
        background: white; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* Responsive */
    @media (max-width: 1024px) {
        [data-testid="column"] { min-width: calc(50% - 12px) !important; flex: 1 1 calc(50% - 12px) !important; }
    }
    @media (max-width: 640px) {
        [data-testid="column"] { min-width: 100% !important; flex: 1 1 100% !important; }
        .block-container { padding: 1rem !important; }
    }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 14px; }
</style>
""", unsafe_allow_html=True)

CHART_BASE = dict(
    paper_bgcolor="white", plot_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=12, color="#475569"),
    margin=dict(t=30, b=40, l=50, r=20),
    xaxis=dict(showgrid=False, linecolor="#e2e8f0", tickcolor="#e2e8f0", color="#94a3b8"),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0", color="#94a3b8"),
)


# ── Data helpers ──────────────────────────────────────────────────────────────

def get_monitoring_reports(limit=100) -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.query(MonitoringReport).order_by(MonitoringReport.id.desc()).limit(limit).all()
        return pd.DataFrame([{
            "id": r.id, "drift_detected": bool(r.drift_detected),
            "drift_score": r.drift_score, "performance_score": r.performance_score,
            "timestamp": r.timestamp,
        } for r in rows])
    finally:
        db.close()


def get_prediction_logs(limit=500) -> pd.DataFrame:
    db = SessionLocal()
    try:
        rows = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
        return pd.DataFrame([{
            "id": r.id, "request_id": r.request_id,
            "prediction": r.prediction, "probability": r.probability,
            "timestamp": r.timestamp,
        } for r in rows])
    finally:
        db.close()


def api_health() -> dict:
    try:
        t0 = time.time()
        r = requests.get(f"{API_URL}/health", timeout=2)
        latency_ms = round((time.time() - t0) * 1000)
        data = r.json()
        data["latency_ms"] = latency_ms
        return data
    except Exception:
        return {"status": "unreachable", "model_loaded": False, "latency_ms": None}


def get_latest_drift_json() -> dict:
    if not os.path.exists(REPORTS_DIR):
        return {}
    files = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")], reverse=True)
    if not files:
        return {}
    try:
        with open(os.path.join(REPORTS_DIR, files[0])) as f:
            return json.load(f)
    except Exception:
        return {}


def get_model_info() -> dict:
    info = {"version": "v1.0", "trained": "—", "dataset": "—", "auc": "—"}
    try:
        import joblib
        mtime = os.path.getmtime("models/model.joblib")
        info["trained"] = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
        model = joblib.load("models/model.joblib")
        info["version"] = f"XGBoost n={model.n_estimators}"
    except Exception:
        pass
    try:
        rows = sum(1 for _ in open("data/reference.csv")) - 1
        info["dataset"] = f"{rows:,} rows"
    except Exception:
        pass
    reports = get_monitoring_reports(5)
    _auc = reports["performance_score"].dropna()
    if not _auc.empty:
        info["auc"] = f"{_auc.iloc[0]:.4f}"
    return info


# ── Sidebar ───────────────────────────────────────────────────────────────────

def nav_button(label: str, icon: str, key: str):
    active = st.session_state.page == label
    col = st.sidebar.container()
    if active:
        col.markdown('<div class="nav-active">', unsafe_allow_html=True)
    if col.button(f"{icon}  {label}", key=key):
        st.session_state.page = label
        st.rerun()
    if active:
        col.markdown("</div>", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📊 ML Monitor")
    st.divider()

    st.markdown('<div class="nav-section">Monitor</div>', unsafe_allow_html=True)
    nav_button("Overview",       "◈",  "nav_overview")
    nav_button("Drift Analysis", "〰", "nav_drift")
    nav_button("Predictions",    "⊞",  "nav_preds")
    nav_button("Reports",        "▤",  "nav_reports")

    st.markdown('<div class="nav-section">System</div>', unsafe_allow_html=True)
    nav_button("Thresholds",   "⊙", "nav_thresh")
    nav_button("Auto-retrain", "↺", "nav_retrain")
    nav_button("Alerts",       "⚑", "nav_alerts")

    st.divider()

    # Model info card
    info = get_model_info()
    st.markdown(f"""
    <div class="model-card">
        <div class="model-card-title">Model Info</div>
        <div class="model-card-row">
            <span class="model-card-key">Type</span>
            <span class="model-card-val">{info['version']}</span>
        </div>
        <div class="model-card-row">
            <span class="model-card-key">Trained</span>
            <span class="model-card-val">{info['trained']}</span>
        </div>
        <div class="model-card-row">
            <span class="model-card-key">Dataset</span>
            <span class="model-card-val">{info['dataset']}</span>
        </div>
        <div class="model-card-row">
            <span class="model-card-key">AUC-ROC</span>
            <span class="model-card-val">{info['auc']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

health      = api_health()
reports_df  = get_monitoring_reports(50)
preds_df    = get_prediction_logs(500)

is_ok      = health.get("status") == "ok"
model_ok   = health.get("model_loaded", False)
latency_ms = health.get("latency_ms")
_drift     = reports_df["drift_score"].dropna() if not reports_df.empty else pd.Series(dtype=float)
latest_drift = _drift.iloc[0] if not _drift.empty else None
drift_alerts = reports_df[reports_df["drift_detected"]].shape[0] if not reports_df.empty else 0

api_pill   = '<span class="pill pill-green"><span class="pill-dot"></span>API live</span>' if is_ok else '<span class="pill pill-red"><span class="pill-dot"></span>API offline</span>'
model_pill = '<span class="pill pill-green"><span class="pill-dot"></span>Model loaded</span>' if model_ok else '<span class="pill pill-gray"><span class="pill-dot"></span>Model not loaded</span>'
alert_cls  = "pill-red" if drift_alerts > 0 else "pill-green"
alert_pill = f'<span class="pill {alert_cls}"><span class="pill-dot"></span>{drift_alerts} drift alert{"s" if drift_alerts != 1 else ""}</span>'

st.markdown(f"""
<div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px;margin-bottom:1.25rem">
    <div>
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="font-size:22px;font-weight:600;color:#111111 !important;-webkit-text-fill-color:#111111 !important;">📊 ML Model Monitor</span>
        </div>
        <div style="font-size:0.82rem;color:#94a3b8;margin-top:4px">
            Fraud detection &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; Credit card transactions
        </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        {api_pill} {model_pill} {alert_pill}
    </div>
</div>
<hr style="border:none;border-top:1px solid #e2e8f0;margin:0 0 1.5rem 0">
""", unsafe_allow_html=True)


# ── Pages ─────────────────────────────────────────────────────────────────────

page = st.session_state.page


# ── Overview ──────────────────────────────────────────────────────────────────
if page == "Overview":

    total       = len(preds_df)
    fraud_count = int(preds_df["prediction"].sum()) if not preds_df.empty else 0
    fraud_rate  = fraud_count / total if total else 0
    prev_total  = max(total - 50, 0)
    drift_color = "red" if latest_drift and latest_drift > DRIFT_THRESHOLD else ""
    drift_val   = f"{latest_drift:.1%}" if latest_drift is not None else "—"
    lat_str     = f"{latency_ms} ms" if latency_ms is not None else "< 100 ms"

    # 4 metric cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-delta">+50 since last check</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Drift Share</div>
            <div class="metric-value {drift_color}">{drift_val}</div>
            <div class="metric-delta">Threshold: {DRIFT_THRESHOLD:.0%}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Fraud Caught</div>
            <div class="metric-value">{fraud_count:,}</div>
            <div class="metric-delta">Rate: {fraud_rate:.1%} of all transactions</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Latency</div>
            <div class="metric-value">{lat_str}</div>
            <div class="metric-delta">Live API response time</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)

    # Two side-by-side panels
    left, right = st.columns(2)

    with left:
        st.markdown('<div class="section-title">Drift Over Time</div>', unsafe_allow_html=True)
        if not reports_df.empty:
            df_r = reports_df.copy()
            df_r["timestamp"] = pd.to_datetime(df_r["timestamp"])
            df_r = df_r.sort_values("timestamp")
            scores = df_r["drift_score"].fillna(0)
            peak_idx = scores.idxmax()
            peak_val = scores.max()
            peak_ts  = df_r.loc[peak_idx, "timestamp"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_r["timestamp"], y=scores,
                mode="lines+markers",
                line=dict(color="#dc2626", width=2),
                marker=dict(size=5, color="#dc2626"),
                fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
                name="Drift Score",
                hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Drift: %{y:.1%}<extra></extra>",
            ))
            fig.add_hline(
                y=DRIFT_THRESHOLD, line_dash="dash", line_color="#16a34a", line_width=1.5,
                annotation_text=f"Threshold {DRIFT_THRESHOLD:.0%}",
                annotation_font_color="#16a34a", annotation_font_size=11,
            )
            if peak_val > 0:
                fig.add_annotation(
                    x=peak_ts, y=peak_val,
                    text=f"Peak {peak_val:.2f}",
                    showarrow=True, arrowhead=2, arrowcolor="#dc2626",
                    font=dict(color="#dc2626", size=11),
                    bgcolor="white", bordercolor="#dc2626", borderwidth=1, borderpad=4,
                    ay=-35,
                )
            fig.update_layout(**CHART_BASE, yaxis_tickformat=".0%", height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No monitoring reports yet.")

    with right:
        st.markdown('<div class="section-title">Feature Drift Scores</div>', unsafe_allow_html=True)
        drift_json = get_latest_drift_json()
        if drift_json.get("feature_details"):
            fd = drift_json["feature_details"]
            feat_df = pd.DataFrame([
                {"feature": k, "p_value": v["p_value"], "drifted": v["drifted"]}
                for k, v in fd.items()
            ])
            feat_df["drift_score"] = 1 - feat_df["p_value"]
            # Top 8 sorted descending — highest drift score at top
            feat_df = feat_df.sort_values("drift_score", ascending=False).head(8)
            # Red = score > 0.5 (drifted), green = score <= 0.5 (stable)
            feat_df["bar_color"] = feat_df["drift_score"].apply(
                lambda s: "#E74C3C" if s > 0.5 else "#27AE60"
            )

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=feat_df["feature"], x=feat_df["drift_score"],
                orientation="h",
                marker_color=feat_df["bar_color"].tolist(),
                text=[f"K-S: {p:.3f}" for p in feat_df["p_value"]],
                textposition="outside",
                textfont=dict(size=10, color="#64748b"),
                hovertemplate="<b>%{y}</b><br>p-value: %{text}<extra></extra>",
                showlegend=False,
            ))
            fig.add_vline(x=0.5, line_dash="dash",
                          line_color="#94a3b8", line_width=1,
                          annotation_text="Drift threshold",
                          annotation_font_size=10, annotation_font_color="#94a3b8")
            fig.update_layout(
                **CHART_BASE, height=280,
                xaxis_title="Drift Score (1 − p-value)",
                xaxis_range=[0, 1.15],
                yaxis_showgrid=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run drift check to see feature breakdown.")

    # Live Prediction Feed
    st.markdown('<div class="section-title">Live Prediction Feed</div>', unsafe_allow_html=True)
    if not preds_df.empty:
        feed = preds_df.head(15).copy()
        feed["timestamp"] = pd.to_datetime(feed["timestamp"]).dt.strftime("%H:%M:%S")
        feed["request_id"] = feed["request_id"].fillna("—")
        feed["confidence"] = feed["probability"].apply(
            lambda p: p if p >= 0.5 else 1 - p
        )

        rows_html = ""
        for _, row in feed.iterrows():
            is_fraud  = row["prediction"] == 1
            badge     = '<span style="background:#FFEBEB;color:#C0392B;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">Fraud</span>' if is_fraud else '<span style="background:#EAFAF1;color:#1E8449;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;">Legit</span>'
            prob      = row["probability"]
            bar_color = "#dc2626" if is_fraud else "#16a34a"
            bar_pct   = round(prob * 100)
            conf_pct  = round(row["confidence"] * 100)
            row_bg    = ' style="background-color:#FFF0F0;"' if is_fraud else ' style="background-color:transparent;"'
            rows_html += f"""
            <tr{row_bg}>
                <td style="font-family:monospace;color:#64748b">#{row['id']}</td>
                <td>
                    <div class="prob-bar-wrap">
                        <span style="font-weight:600;color:#1e293b;width:38px">{prob:.3f}</span>
                        <div class="prob-bar-bg">
                            <div class="prob-bar-fill" style="width:{bar_pct}%;background:{bar_color}"></div>
                        </div>
                    </div>
                </td>
                <td>{badge}</td>
                <td style="color:#475569">{conf_pct}%</td>
                <td style="color:#94a3b8;font-family:monospace">{row['timestamp']}</td>
            </tr>"""

        st.markdown(f"""
        <div class="chart-card" style="overflow-x:auto">
            <table class="pred-table">
                <thead><tr>
                    <th>#</th>
                    <th>Fraud Probability</th>
                    <th>Verdict</th>
                    <th>Confidence</th>
                    <th>Timestamp</th>
                </tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No predictions yet.")

    # System Insights
    st.markdown('<div class="section-title">System Insights</div>', unsafe_allow_html=True)

    _auc2 = reports_df["performance_score"].dropna() if not reports_df.empty else pd.Series(dtype=float)
    auc_val2 = _auc2.iloc[0] if not _auc2.empty else None
    last_retrain2 = reports_df[reports_df["drift_detected"]]["timestamp"].dropna() if not reports_df.empty else pd.Series(dtype=str)
    last_ts2 = pd.to_datetime(last_retrain2.iloc[0]).strftime("%Y-%m-%d %H:%M") if not last_retrain2.empty else "Never"

    drift_title  = "⚠ Drift Exceeds Threshold" if (latest_drift and latest_drift > DRIFT_THRESHOLD) else "✓ Drift Within Threshold"
    drift_body   = f"Drift share is <b>{drift_val}</b>, {'above' if (latest_drift and latest_drift > DRIFT_THRESHOLD) else 'below'} the {DRIFT_THRESHOLD:.0%} threshold. {'Auto-retrain has been triggered.' if (latest_drift and latest_drift > DRIFT_THRESHOLD) else 'Feature distributions are stable.'}"
    perf_title   = "✓ Model Performing Well" if (auc_val2 and auc_val2 >= PERFORMANCE_THRESHOLD) else "ℹ Model Performance"
    perf_body    = f"ROC-AUC is <b>{auc_val2:.4f}</b>, above the {PERFORMANCE_THRESHOLD:.2f} threshold. Fraud rate <b>{fraud_rate:.1%}</b> within baseline." if (auc_val2 and auc_val2 >= PERFORMANCE_THRESHOLD) else f"Fraud rate is <b>{fraud_rate:.1%}</b>. Run a monitoring cycle to compute the latest AUC score."

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:8px;">
        <div style="background:#FFF8F7;border-left:3px solid #E74C3C;border-radius:8px;padding:14px 16px;">
            <div style="font-size:13px;font-weight:600;color:#C0392B;margin-bottom:6px;">{drift_title}</div>
            <div style="font-size:12px;color:#555;">{drift_body}</div>
        </div>
        <div style="background:#F7FFF9;border-left:3px solid #27AE60;border-radius:8px;padding:14px 16px;">
            <div style="font-size:13px;font-weight:600;color:#1E8449;margin-bottom:6px;">{perf_title}</div>
            <div style="font-size:12px;color:#555;">{perf_body}</div>
        </div>
        <div style="background:#F7FBFF;border-left:3px solid #2980B9;border-radius:8px;padding:14px 16px;">
            <div style="font-size:13px;font-weight:600;color:#1A5276;margin-bottom:6px;">↺ Retrain Pipeline</div>
            <div style="font-size:12px;color:#555;">Auto-retrain triggers at {DRIFT_THRESHOLD:.0%} drift. Last triggered: <b>{last_ts2} UTC</b>. Model backup saved before each retrain.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Drift Analysis ────────────────────────────────────────────────────────────
elif page == "Drift Analysis":
    st.markdown("### Drift Analysis")
    reports_df2 = get_monitoring_reports(100)
    if reports_df2.empty:
        st.info("No monitoring reports yet.")
    else:
        reports_df2["timestamp"] = pd.to_datetime(reports_df2["timestamp"])
        reports_df2 = reports_df2.sort_values("timestamp")
        scores = reports_df2["drift_score"].fillna(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=reports_df2["timestamp"], y=scores,
            mode="lines+markers", line=dict(color="#dc2626", width=2),
            fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
            hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Drift: %{y:.1%}<extra></extra>",
        ))
        fig.add_hline(y=DRIFT_THRESHOLD, line_dash="dash", line_color="#16a34a",
                      annotation_text=f"Threshold {DRIFT_THRESHOLD:.0%}",
                      annotation_font_color="#16a34a")
        fig.update_layout(**CHART_BASE, yaxis_tickformat=".0%", height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Drift Alert History</div>', unsafe_allow_html=True)
        alerts = reports_df2[reports_df2["drift_detected"]][["timestamp", "drift_score"]].copy()
        alerts.columns = ["Timestamp", "Drift Score"]
        if not alerts.empty:
            alerts["Drift Score"] = alerts["Drift Score"].apply(lambda x: f"{x:.2%}")
            st.dataframe(alerts, use_container_width=True, hide_index=True)
        else:
            st.success("No drift alerts triggered.")


# ── Predictions ───────────────────────────────────────────────────────────────
elif page == "Predictions":
    st.markdown("### Predictions")
    preds_df2 = get_prediction_logs(500)
    if preds_df2.empty:
        st.info("No predictions yet.")
    else:
        preds_df2["timestamp"] = pd.to_datetime(preds_df2["timestamp"])
        preds_df2 = preds_df2.sort_values("timestamp")

        c1, c2 = st.columns(2)
        with c1:
            labels = preds_df2["prediction"].map({0: "Legit", 1: "Fraud"})
            counts = labels.value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=counts.index, values=counts.values, hole=0.5,
                marker=dict(colors=["#16a34a", "#dc2626"],
                            line=dict(color="white", width=2)),
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            )])
            fig.update_layout(**CHART_BASE, height=300,
                              margin_t=30, margin_b=20, margin_l=20, margin_r=20)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=preds_df2["timestamp"], y=preds_df2["probability"],
                mode="lines", line=dict(color="#6366f1", width=1.5),
                hovertemplate="<b>%{x|%H:%M:%S}</b><br>Prob: %{y:.4f}<extra></extra>",
            ))
            fig2.add_hline(y=0.5, line_dash="dash", line_color="#dc2626",
                           annotation_text="Decision boundary")
            fig2.update_layout(**CHART_BASE, height=300, yaxis_title="Fraud Probability")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-title">All Predictions</div>', unsafe_allow_html=True)
        display = (
            preds_df2.sort_values("timestamp", ascending=False)
            .head(200)[["request_id", "timestamp", "probability", "prediction"]]
            .rename(columns={"request_id": "Txn ID", "timestamp": "Time",
                             "probability": "Fraud Prob", "prediction": "Label"})
        )
        display["Label"] = display["Label"].map({0: "Legit", 1: "Fraud"})
        st.dataframe(display, use_container_width=True, hide_index=True)


# ── Reports ───────────────────────────────────────────────────────────────────
elif page == "Reports":
    st.markdown("### Evidently Drift Reports")
    reports_dir = "monitoring/reports"
    if os.path.exists(reports_dir):
        html_files = sorted([f for f in os.listdir(reports_dir) if f.endswith(".html")], reverse=True)
        if html_files:
            selected = st.selectbox("Select report", html_files)
            with open(os.path.join(reports_dir, selected)) as f:
                st.components.v1.html(f.read(), height=800, scrolling=True)
        else:
            st.info("No HTML reports yet.")
    else:
        st.info("Reports directory not found.")


# ── Thresholds ────────────────────────────────────────────────────────────────
elif page == "Thresholds":
    st.markdown("### Thresholds")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Drift Threshold</div>
            <div class="metric-value">{DRIFT_THRESHOLD:.0%}</div>
            <div class="metric-delta">Triggers alert + auto-retrain when exceeded</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Performance Threshold</div>
            <div class="metric-value">{PERFORMANCE_THRESHOLD:.2f}</div>
            <div class="metric-delta">Minimum ROC-AUC before retraining is triggered</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-top:1rem;padding:1rem;background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;font-size:0.85rem;color:#475569;line-height:1.6">
    <b>How thresholds work:</b> Every monitoring cycle, Evidently computes the K-S statistic
    for each feature. If the share of drifted features exceeds the drift threshold, an alert fires
    and the auto-retrain pipeline runs automatically. If ROC-AUC drops below the performance
    threshold, retraining is also triggered. Configure these values in your <code>.env</code> file.
    </div>""", unsafe_allow_html=True)


# ── Auto-retrain ──────────────────────────────────────────────────────────────
elif page == "Auto-retrain":
    st.markdown("### Auto-retrain Pipeline")
    if st.button("▶  Force Retrain Now", type="primary"):
        try:
            from retrain.retrain import retrain
            with st.spinner("Retraining model..."):
                result = retrain(force=True)
            st.success(f"Retrain complete — New AUC: {result.get('new_auc', 0):.4f} | Backup: {result.get('backup_path', '—')}")
        except Exception as e:
            st.error(f"Retrain failed: {e}")
    st.markdown('<div class="section-title">Retrain History</div>', unsafe_allow_html=True)
    rpts = get_monitoring_reports(50)
    triggered = rpts[rpts["drift_detected"]][["timestamp", "drift_score"]].copy() if not rpts.empty else pd.DataFrame()
    if not triggered.empty:
        triggered.columns = ["Triggered At", "Drift Score"]
        triggered["Drift Score"] = triggered["Drift Score"].apply(lambda x: f"{x:.2%}")
        st.dataframe(triggered, use_container_width=True, hide_index=True)
    else:
        st.info("No auto-retrains triggered yet.")


# ── Alerts ────────────────────────────────────────────────────────────────────
elif page == "Alerts":
    st.markdown("### Alerts")
    rpts = get_monitoring_reports(100)
    drift_alerts_df = rpts[rpts["drift_detected"]].copy() if not rpts.empty else pd.DataFrame()
    if not drift_alerts_df.empty:
        st.error(f"**{len(drift_alerts_df)} drift alert(s) recorded.** Drift exceeded {DRIFT_THRESHOLD:.0%} threshold on these cycles:")
        drift_alerts_df["timestamp"] = pd.to_datetime(drift_alerts_df["timestamp"])
        drift_alerts_df = drift_alerts_df.sort_values("timestamp", ascending=False)
        for _, row in drift_alerts_df.iterrows():
            ts = row["timestamp"].strftime("%Y-%m-%d %H:%M UTC")
            st.markdown(f"""
            <div class="insight insight-warn" style="margin-bottom:8px">
                <div class="insight-title insight-title-warn">⚠ Drift Alert — {ts}</div>
                <div class="insight-body">Drift share: <b>{row['drift_score']:.2%}</b> — exceeded {DRIFT_THRESHOLD:.0%} threshold. Auto-retrain triggered.</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("No alerts. All monitoring cycles are within threshold.")
