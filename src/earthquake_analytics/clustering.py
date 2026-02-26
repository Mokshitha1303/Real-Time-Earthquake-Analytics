from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def cluster_spatiotemporal_events(
    events: pd.DataFrame,
    *,
    now_utc: datetime,
    space_km: float,
    time_window_hours: int,
    min_samples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if events.empty:
        return ([], [], {"window_event_count": 0, "cluster_count": 0, "noise_count": 0})

    filtered = events.copy()
    filtered = filtered[
        (filtered["event_type"] == "earthquake")
        & filtered["latitude"].notna()
        & filtered["longitude"].notna()
        & filtered["time_utc"].notna()
    ]
    window_start = now_utc - timedelta(hours=time_window_hours)
    filtered = filtered[filtered["time_utc"] >= pd.Timestamp(window_start)]

    if filtered.empty:
        return ([], [], {"window_event_count": 0, "cluster_count": 0, "noise_count": 0})

    lat = filtered["latitude"].to_numpy(dtype=float)
    lon = filtered["longitude"].to_numpy(dtype=float)
    time_hours = filtered["time_utc"].astype("int64").to_numpy(dtype=float) / 3_600_000_000_000.0

    lat_km = lat * 111.0
    lon_km = lon * 111.0 * np.cos(np.radians(lat))

    features = np.column_stack(
        [
            lat_km / max(space_km, 1.0),
            lon_km / max(space_km, 1.0),
            time_hours / max(float(time_window_hours), 1.0),
        ]
    )

    labels = DBSCAN(eps=1.0, min_samples=min_samples).fit_predict(features)
    filtered = filtered.reset_index(drop=True)
    filtered["cluster_id"] = labels
    filtered["is_noise"] = filtered["cluster_id"] == -1

    assignments = [
        {
            "event_id": row.event_id,
            "cluster_id": int(row.cluster_id),
            "is_noise": bool(row.is_noise),
        }
        for row in filtered.itertuples()
    ]

    summaries: list[dict[str, Any]] = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        members = filtered[filtered["cluster_id"] == cluster_id]
        if members.empty:
            continue
        mode_region = (
            members["region_hint"].mode().iat[0]
            if members["region_hint"].notna().any()
            else "unknown"
        )
        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "event_count": int(len(members)),
                "start_time_utc": members["time_utc"].min().to_pydatetime(),
                "end_time_utc": members["time_utc"].max().to_pydatetime(),
                "centroid_lat": float(members["latitude"].mean()),
                "centroid_lon": float(members["longitude"].mean()),
                "mean_magnitude": float(members["mag"].dropna().mean())
                if members["mag"].notna().any()
                else None,
                "max_magnitude": float(members["mag"].dropna().max())
                if members["mag"].notna().any()
                else None,
                "region_hint": str(mode_region) if mode_region is not None else None,
            }
        )

    stats = {
        "window_event_count": int(len(filtered)),
        "cluster_count": int(len(summaries)),
        "noise_count": int((labels == -1).sum()),
    }
    return (assignments, summaries, stats)

