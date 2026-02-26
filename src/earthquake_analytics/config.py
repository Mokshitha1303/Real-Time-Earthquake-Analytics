from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    source_url: str
    db_url: str
    summary_feed_path: Path
    summary_geojson_path: Path
    refresh_interval_seconds: int
    anomaly_z_threshold: float
    mag_alert_threshold: float
    cluster_time_window_hours: int
    cluster_space_km: float
    cluster_min_samples: int
    lookback_days: int
    max_event_age_days: int
    artifacts_dir: Path
    output_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        output_dir = Path(os.getenv("OUTPUT_DIR", "output"))
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))

        settings = cls(
            source_url=os.getenv(
                "EARTHQUAKE_SOURCE_URL",
                "data/all_month.geojson",
            ),
            db_url=os.getenv("EARTHQUAKE_DB_URL", "sqlite:///earthquakes.db"),
            summary_feed_path=Path(
                os.getenv("SUMMARY_FEED_PATH", str(output_dir / "syso_summary_feed.json"))
            ),
            summary_geojson_path=Path(
                os.getenv("SUMMARY_GEOJSON_PATH", str(output_dir / "syso_summary_feed.geojson"))
            ),
            refresh_interval_seconds=int(os.getenv("REFRESH_INTERVAL_SECONDS", "60")),
            anomaly_z_threshold=float(os.getenv("ANOMALY_Z_THRESHOLD", "3.0")),
            mag_alert_threshold=float(os.getenv("MAG_ALERT_THRESHOLD", "5.0")),
            cluster_time_window_hours=int(os.getenv("CLUSTER_TIME_WINDOW_HOURS", "48")),
            cluster_space_km=float(os.getenv("CLUSTER_SPACE_KM", "100")),
            cluster_min_samples=int(os.getenv("CLUSTER_MIN_SAMPLES", "4")),
            lookback_days=int(os.getenv("LOOKBACK_DAYS", "30")),
            max_event_age_days=int(os.getenv("MAX_EVENT_AGE_DAYS", "45")),
            artifacts_dir=artifacts_dir,
            output_dir=output_dir,
        )
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
        settings.summary_feed_path.parent.mkdir(parents=True, exist_ok=True)
        settings.summary_geojson_path.parent.mkdir(parents=True, exist_ok=True)
        return settings
