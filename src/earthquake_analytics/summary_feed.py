from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def build_summary_feed(
    *,
    now_utc: datetime,
    source_url: str,
    source_generated_utc: datetime | None,
    run_id: int,
    accepted_count: int,
    rejected_count: int,
    warning_count: int,
    new_count: int,
    updated_count: int,
    events: pd.DataFrame,
    clusters: list[dict[str, Any]],
    cluster_stats: dict[str, Any],
    model_summary: dict[str, Any],
    mag_alert_threshold: float,
    anomaly_z_threshold: float,
    refresh_interval_seconds: int,
) -> dict[str, Any]:
    selected_z_threshold = float(model_summary.get("selected_z_threshold", anomaly_z_threshold))
    selected_min_count = int(model_summary.get("selected_min_anomaly_count", 1))

    if events.empty:
        return {
            "sysoJSON_version": "1.0",
            "generated_at_utc": now_utc.isoformat(),
            "source_feed_url": source_url,
            "source_generated_at_utc": source_generated_utc.isoformat() if source_generated_utc else None,
            "refresh_interval_seconds": refresh_interval_seconds,
            "run_id": run_id,
            "kpis": {
                "events_last_hour": 0,
                "events_last_24h": 0,
                "max_magnitude_24h": None,
                "tsunami_flagged_24h": 0,
            },
            "alerts": {"magnitude": [], "anomalous_regions": []},
            "quality": {
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "warning_count": warning_count,
                "new_count": new_count,
                "updated_count": updated_count,
            },
            "clusters": [],
            "cluster_stats": cluster_stats,
            "model": model_summary,
            "trends": {"hourly_counts": [], "daily_counts": []},
        }

    frame = events.copy()
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    one_hour_ago = pd.Timestamp(now_utc - timedelta(hours=1))
    one_day_ago = pd.Timestamp(now_utc - timedelta(days=1))
    seven_days_ago = pd.Timestamp(now_utc - timedelta(days=7))

    events_last_hour = frame[frame["time_utc"] >= one_hour_ago]
    events_last_24h = frame[frame["time_utc"] >= one_day_ago]
    events_last_7d = frame[frame["time_utc"] >= seven_days_ago]

    major_events = (
        events_last_24h[events_last_24h["mag"].fillna(-999) >= mag_alert_threshold]
        .sort_values("mag", ascending=False)
        .head(20)
    )
    mag_alerts = [
        {
            "event_id": row.event_id,
            "time_utc": pd.Timestamp(row.time_utc).isoformat(),
            "mag": float(row.mag) if pd.notna(row.mag) else None,
            "place": row.place,
            "region_hint": row.region_hint,
            "tsunami": int(row.tsunami) if pd.notna(row.tsunami) else None,
        }
        for row in major_events.itertuples()
    ]

    top_clusters = sorted(clusters, key=lambda item: item["event_count"], reverse=True)[:10]
    for cluster in top_clusters:
        if isinstance(cluster.get("start_time_utc"), datetime):
            cluster["start_time_utc"] = cluster["start_time_utc"].isoformat()
        if isinstance(cluster.get("end_time_utc"), datetime):
            cluster["end_time_utc"] = cluster["end_time_utc"].isoformat()

    hourly = (
        events_last_7d.assign(hour_utc=events_last_7d["time_utc"].dt.floor("h"))
        .groupby("hour_utc")
        .size()
        .rename("count")
        .reset_index()
        .tail(168)
    )
    daily = (
        frame.assign(day_utc=frame["time_utc"].dt.floor("d"))
        .groupby("day_utc")
        .size()
        .rename("count")
        .reset_index()
        .tail(31)
    )

    anomalous_regions = [
        {
            "region_hint": row["region_hint"],
            "hour_utc": row["hour_utc"].isoformat()
            if isinstance(row.get("hour_utc"), datetime)
            else row.get("hour_utc"),
            "observed_count": row["observed_count"],
            "expected_count": row["expected_count"],
            "z_score": row["z_score"],
        }
        for row in model_summary.get("top_anomalies", [])
        if bool(row.get("is_anomalous"))
        or (
            row.get("z_score", 0) >= selected_z_threshold
            and row.get("observed_count", 0) >= selected_min_count
        )
    ]

    return {
        "sysoJSON_version": "1.0",
        "generated_at_utc": now_utc.isoformat(),
        "source_feed_url": source_url,
        "source_generated_at_utc": source_generated_utc.isoformat() if source_generated_utc else None,
        "refresh_interval_seconds": refresh_interval_seconds,
        "run_id": run_id,
        "kpis": {
            "events_last_hour": int(len(events_last_hour)),
            "events_last_24h": int(len(events_last_24h)),
            "max_magnitude_24h": float(events_last_24h["mag"].max())
            if events_last_24h["mag"].notna().any()
            else None,
            "tsunami_flagged_24h": int((events_last_24h["tsunami"] == 1).sum()),
            "quality_pass_rate_24h": float(events_last_24h["quality_passed"].mean() * 100)
            if not events_last_24h.empty
            else None,
        },
        "alerts": {
            "magnitude_threshold": mag_alert_threshold,
            "anomaly_z_threshold": selected_z_threshold,
            "anomaly_min_count": selected_min_count,
            "magnitude": mag_alerts,
            "anomalous_regions": anomalous_regions,
        },
        "quality": {
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "warning_count": warning_count,
            "new_count": new_count,
            "updated_count": updated_count,
        },
        "clusters": top_clusters,
        "cluster_stats": cluster_stats,
        "model": model_summary,
        "trends": {
            "hourly_counts": [
                {"hour_utc": row.hour_utc.isoformat(), "count": int(row.count)}
                for row in hourly.itertuples()
            ],
            "daily_counts": [
                {"day_utc": row.day_utc.isoformat(), "count": int(row.count)}
                for row in daily.itertuples()
            ],
        },
    }


def build_summary_geojson(events: pd.DataFrame, *, max_events: int = 500) -> dict[str, Any]:
    if events.empty:
        return {"type": "FeatureCollection", "features": []}

    frame = (
        events[events["latitude"].notna() & events["longitude"].notna()]
        .sort_values("time_utc", ascending=False)
        .head(max_events)
    )
    features = []
    for row in frame.itertuples():
        features.append(
            {
                "type": "Feature",
                "id": row.event_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row.longitude), float(row.latitude), float(row.depth_km or 0.0)],
                },
                "properties": {
                    "time_utc": pd.Timestamp(row.time_utc).isoformat(),
                    "mag": float(row.mag) if pd.notna(row.mag) else None,
                    "sig": int(row.sig) if pd.notna(row.sig) else None,
                    "place": row.place,
                    "region_hint": row.region_hint,
                    "event_type": row.event_type,
                    "alert": row.alert,
                    "tsunami": int(row.tsunami) if pd.notna(row.tsunami) else None,
                },
            }
        )
    return {"type": "FeatureCollection", "generated_at_utc": datetime.now(UTC).isoformat(), "features": features}


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
