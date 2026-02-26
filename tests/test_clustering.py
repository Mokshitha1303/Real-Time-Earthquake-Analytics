from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from earthquake_analytics.clustering import cluster_spatiotemporal_events


def test_cluster_spatiotemporal_events_returns_clusters() -> None:
    now = datetime(2026, 2, 26, tzinfo=UTC)
    rows = [
        {
            "event_id": f"ev{i}",
            "event_type": "earthquake",
            "latitude": 37.0 + i * 0.01,
            "longitude": -122.0 + i * 0.01,
            "time_utc": pd.Timestamp(now),
            "mag": 2.0,
            "region_hint": "CA",
        }
        for i in range(6)
    ]
    df = pd.DataFrame(rows)
    assignments, summaries, stats = cluster_spatiotemporal_events(
        df,
        now_utc=now,
        space_km=30,
        time_window_hours=48,
        min_samples=3,
    )
    assert len(assignments) == 6
    assert stats["window_event_count"] == 6
    assert stats["cluster_count"] >= 1
    assert len(summaries) >= 1

