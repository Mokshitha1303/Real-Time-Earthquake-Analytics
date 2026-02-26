from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from sqlalchemy import desc, select

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earthquake_analytics.config import Settings
from earthquake_analytics.db import build_engine, build_session_factory, init_db
from earthquake_analytics.models import ClusterSummary, IngestionRun, RegionalActivityScore
from earthquake_analytics.storage import load_events_frame

st.set_page_config(
    page_title="Earthquake Analytics Live Dashboard",
    page_icon="🌍",
    layout="wide",
)

settings = Settings.from_env()
st_autorefresh(interval=settings.refresh_interval_seconds * 1000, key="eq_refresh")

st.title("Real-Time Earthquake Analytics")
st.caption(
    f"Auto-refresh every {settings.refresh_interval_seconds}s | Source: {settings.source_url}"
)

mag_threshold = st.sidebar.slider("Magnitude Alert Threshold", 1.0, 8.0, float(settings.mag_alert_threshold), 0.1)
anomaly_threshold = st.sidebar.slider(
    "Regional Anomaly Z-Threshold", 1.0, 8.0, float(settings.anomaly_z_threshold), 0.1
)
window_hours = st.sidebar.slider("Map/Trend Window (Hours)", 6, 720, 72, 6)

engine = build_engine(settings.db_url)
init_db(engine)
SessionFactory = build_session_factory(engine)


@st.cache_data(ttl=30)
def load_summary_feed(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_dashboard_data(hours: int) -> tuple[pd.DataFrame, dict, IngestionRun | None, pd.DataFrame, pd.DataFrame]:
    with SessionFactory() as session:
        since = datetime.now(tz=UTC) - timedelta(hours=hours)
        events = load_events_frame(session, since_utc=since)
        latest_run = session.scalars(select(IngestionRun).order_by(desc(IngestionRun.run_id)).limit(1)).first()

        clusters_df = pd.DataFrame()
        activity_df = pd.DataFrame()
        if latest_run is not None:
            clusters_rows = session.execute(
                select(ClusterSummary).where(ClusterSummary.run_id == latest_run.run_id)
            ).scalars()
            clusters_df = pd.DataFrame(
                [
                    {
                        "cluster_id": row.cluster_id,
                        "event_count": row.event_count,
                        "start_time_utc": row.start_time_utc,
                        "end_time_utc": row.end_time_utc,
                        "centroid_lat": row.centroid_lat,
                        "centroid_lon": row.centroid_lon,
                        "mean_magnitude": row.mean_magnitude,
                        "max_magnitude": row.max_magnitude,
                        "region_hint": row.region_hint,
                    }
                    for row in clusters_rows
                ]
            )

            activity_rows = session.execute(
                select(RegionalActivityScore).where(RegionalActivityScore.run_id == latest_run.run_id)
            ).scalars()
            activity_df = pd.DataFrame(
                [
                    {
                        "region_hint": row.region_hint,
                        "hour_utc": row.hour_utc,
                        "observed_count": row.observed_count,
                        "expected_count": row.expected_count,
                        "z_score": row.z_score,
                        "is_anomalous": row.is_anomalous,
                    }
                    for row in activity_rows
                ]
            )
    return events, load_summary_feed(str(settings.summary_feed_path)), latest_run, clusters_df, activity_df


events_df, summary_feed, latest_run, clusters_df, activity_df = load_dashboard_data(window_hours)

if events_df.empty:
    st.warning("No events found in the selected window. Run `eq-pipeline` first.")
    st.stop()

events_df["time_utc"] = pd.to_datetime(events_df["time_utc"], utc=True)
events_24h = events_df[events_df["time_utc"] >= pd.Timestamp(datetime.now(tz=UTC) - timedelta(days=1))]
events_1h = events_df[events_df["time_utc"] >= pd.Timestamp(datetime.now(tz=UTC) - timedelta(hours=1))]
major_events = events_df[events_df["mag"].fillna(-999) >= mag_threshold].sort_values("mag", ascending=False)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Events (Last Hour)", int(len(events_1h)))
col2.metric("Events (Last 24h)", int(len(events_24h)))
col3.metric("Max Magnitude (24h)", f"{events_24h['mag'].max():.2f}" if events_24h["mag"].notna().any() else "N/A")
col4.metric("Tsunami-Flagged (24h)", int((events_24h["tsunami"] == 1).sum()))

map_col, alert_col = st.columns([2, 1])
with map_col:
    st.subheader("Interactive Event Map")
    map_df = events_df[events_df["latitude"].notna() & events_df["longitude"].notna()].copy()
    fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="mag",
        size="sig",
        size_max=24,
        zoom=1,
        hover_name="place",
        hover_data={
            "time_utc": True,
            "mag": ":.2f",
            "depth_km": ":.1f",
            "region_hint": True,
            "event_type": True,
            "latitude": False,
            "longitude": False,
        },
        color_continuous_scale="Turbo",
        height=520,
    )
    fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with alert_col:
    st.subheader("Alert Panel")
    st.write(f"**Magnitude alerts >= {mag_threshold:.1f}**")
    st.dataframe(
        major_events[["time_utc", "mag", "place", "region_hint", "tsunami"]].head(12),
        use_container_width=True,
        hide_index=True,
    )

    st.write(f"**Regional anomaly alerts z >= {anomaly_threshold:.1f}**")
    if activity_df.empty:
        st.info("No modeled regional scores found yet.")
    else:
        alerts_df = activity_df[
            (activity_df["z_score"] >= anomaly_threshold)
            & (activity_df["is_anomalous"] == True)
        ].sort_values("z_score", ascending=False)
        st.dataframe(
            alerts_df[["region_hint", "hour_utc", "observed_count", "expected_count", "z_score"]].head(12),
            use_container_width=True,
            hide_index=True,
        )

trend_col1, trend_col2 = st.columns(2)
with trend_col1:
    st.subheader("Hourly Trend")
    hourly = (
        events_df.assign(hour_utc=events_df["time_utc"].dt.floor("h"))
        .groupby("hour_utc")
        .size()
        .rename("count")
        .reset_index()
    )
    trend_fig = px.line(hourly, x="hour_utc", y="count", markers=True, height=340)
    trend_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(trend_fig, use_container_width=True)

with trend_col2:
    st.subheader("Top Active Regions")
    region_counts = (
        events_df.groupby("region_hint")
        .size()
        .sort_values(ascending=False)
        .head(15)
        .rename("count")
        .reset_index()
    )
    region_fig = px.bar(region_counts, x="region_hint", y="count", height=340, color="count")
    region_fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Region", yaxis_title="Events")
    st.plotly_chart(region_fig, use_container_width=True)

cluster_col, quality_col = st.columns([3, 2])
with cluster_col:
    st.subheader("Spatial-Temporal Clusters")
    if clusters_df.empty:
        st.info("No cluster summaries for the latest run.")
    else:
        st.dataframe(
            clusters_df.sort_values("event_count", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )

with quality_col:
    st.subheader("Pipeline Quality")
    if latest_run is None:
        st.info("No ingestion run metadata available.")
    else:
        st.write(f"Run ID: `{latest_run.run_id}`")
        st.write(f"Status: `{latest_run.status}`")
        st.write(f"Started: `{latest_run.run_started_at}`")
        st.write(f"Finished: `{latest_run.run_finished_at}`")
        st.write(f"Accepted: `{latest_run.accepted_count}`")
        st.write(f"Rejected: `{latest_run.rejected_count}`")
        st.write(f"Warnings: `{latest_run.warning_count}`")
        st.write(f"New: `{latest_run.new_count}`")
        st.write(f"Updated: `{latest_run.updated_count}`")

st.subheader("Model Performance Snapshot")
if summary_feed and isinstance(summary_feed.get("model"), dict):
    model = summary_feed["model"]
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("MAE", f"{model.get('mae', 0):.4f}" if model.get("mae") is not None else "N/A")
    mcol2.metric("RMSE", f"{model.get('rmse', 0):.4f}" if model.get("rmse") is not None else "N/A")
    mcol3.metric("Recall", f"{model.get('recall', 0):.3f}" if model.get("recall") is not None else "N/A")
    mcol4.metric("F1", f"{model.get('f1', 0):.3f}" if model.get("f1") is not None else "N/A")
    st.caption(
        f"Auto-selected anomaly threshold z={model.get('selected_z_threshold')} | "
        f"min_count={model.get('selected_min_anomaly_count')}"
    )
else:
    st.info("Model metrics are not available yet.")

if summary_feed:
    st.subheader("sysoJSON Summary Feed Snapshot")
    if st.checkbox("Show raw feed JSON", value=False):
        st.json(summary_feed)
