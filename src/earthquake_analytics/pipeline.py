from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from .clustering import cluster_spatiotemporal_events
from .config import Settings
from .db import build_engine, build_session_factory, init_db
from .eda import run_hypothesis_eda
from .ingest import fetch_geojson
from .logging_utils import get_logger
from .modeling import build_regional_activity_scores
from .storage import (
    finish_ingestion_run,
    load_events_frame,
    replace_activity_scores,
    replace_cluster_outputs,
    start_ingestion_run,
    upsert_events,
)
from .summary_feed import build_summary_feed, build_summary_geojson, write_json
from .utils import epoch_ms_to_dt, json_dumps
from .validation import normalize_feature, validate_feed_shape


def run_pipeline(settings: Settings | None = None) -> dict[str, Any]:
    settings = settings or Settings.from_env()
    logger = get_logger("earthquake_analytics.pipeline")
    run_started_at = datetime.now(tz=UTC)
    logger.info("Pipeline run started at %s", run_started_at.isoformat())

    payload = fetch_geojson(settings.source_url)
    feed_errors = validate_feed_shape(payload)
    if feed_errors:
        raise ValueError(f"Feed validation failed: {feed_errors}")

    metadata = payload.get("metadata") or {}
    source_generated_utc = epoch_ms_to_dt(metadata.get("generated"))
    source_event_count = int(metadata.get("count")) if metadata.get("count") is not None else None
    features = payload.get("features", [])

    engine = build_engine(settings.db_url)
    init_db(engine)
    session_factory = build_session_factory(engine)

    with session_factory() as session:
        return _run_pipeline_txn(
            session=session,
            settings=settings,
            run_started_at=run_started_at,
            source_generated_utc=source_generated_utc,
            source_event_count=source_event_count,
            features=features,
            metadata=metadata,
        )


def _run_pipeline_txn(
    *,
    session: Session,
    settings: Settings,
    run_started_at: datetime,
    source_generated_utc: datetime | None,
    source_event_count: int | None,
    features: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    logger = get_logger("earthquake_analytics.pipeline")
    run = start_ingestion_run(
        session,
        source_url=settings.source_url,
        run_started_at=run_started_at,
        source_generated_utc=source_generated_utc,
        source_event_count=source_event_count,
    )
    logger.info("Created ingestion run_id=%s", run.run_id)

    accepted_records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    warning_count = 0
    seen_ids: set[str] = set()

    try:
        now_utc = datetime.now(tz=UTC)
        for index, feature in enumerate(features):
            result = normalize_feature(
                feature,
                now_utc=now_utc,
                max_event_age_days=settings.max_event_age_days,
            )
            if result.record is None:
                rejected.append(
                    {
                        "index": index,
                        "id": feature.get("id") if isinstance(feature, dict) else None,
                        "errors": result.hard_errors,
                        "warnings": result.soft_warnings,
                    }
                )
                continue

            event_id = result.record["event_id"]
            if event_id in seen_ids:
                rejected.append(
                    {
                        "index": index,
                        "id": event_id,
                        "errors": ["duplicate_event_id_within_batch"],
                        "warnings": [],
                    }
                )
                continue
            seen_ids.add(event_id)

            warning_count += len(result.soft_warnings)
            result.record["quality_passed"] = len(result.soft_warnings) == 0
            result.record["quality_issues"] = (
                json_dumps(result.soft_warnings) if result.soft_warnings else None
            )
            result.record["ingested_at"] = now_utc
            accepted_records.append(result.record)

        new_count, updated_count = upsert_events(session, accepted_records)
        logger.info(
            "Upsert complete: accepted=%s rejected=%s new=%s updated=%s warnings=%s",
            len(accepted_records),
            len(rejected),
            new_count,
            updated_count,
            warning_count,
        )

        since = now_utc - timedelta(days=settings.lookback_days)
        events = load_events_frame(session, since_utc=since)

        assignments, cluster_summaries, cluster_stats = cluster_spatiotemporal_events(
            events,
            now_utc=now_utc,
            space_km=settings.cluster_space_km,
            time_window_hours=settings.cluster_time_window_hours,
            min_samples=settings.cluster_min_samples,
        )
        replace_cluster_outputs(
            session,
            run_id=run.run_id,
            assignments=[{**item, "run_id": run.run_id} for item in assignments],
            summaries=[{**item, "run_id": run.run_id} for item in cluster_summaries],
        )

        model_output = build_regional_activity_scores(
            events,
            now_utc=now_utc,
            z_threshold=settings.anomaly_z_threshold,
            artifacts_dir=settings.artifacts_dir,
        )
        replace_activity_scores(
            session,
            run_id=run.run_id,
            scores=[{**item, "run_id": run.run_id} for item in model_output.scores],
        )

        eda_output = run_hypothesis_eda(session, settings)

        summary = build_summary_feed(
            now_utc=now_utc,
            source_url=settings.source_url,
            source_generated_utc=source_generated_utc,
            run_id=run.run_id,
            accepted_count=len(accepted_records),
            rejected_count=len(rejected),
            warning_count=warning_count,
            new_count=new_count,
            updated_count=updated_count,
            events=events,
            clusters=cluster_summaries,
            cluster_stats=cluster_stats,
            model_summary=model_output.summary,
            mag_alert_threshold=settings.mag_alert_threshold,
            anomaly_z_threshold=settings.anomaly_z_threshold,
            refresh_interval_seconds=settings.refresh_interval_seconds,
        )
        write_json(summary, settings.summary_feed_path)

        geojson = build_summary_geojson(events, max_events=600)
        write_json(geojson, settings.summary_geojson_path)

        notes = {
            "source_title": metadata.get("title"),
            "source_status": metadata.get("status"),
            "rejected_examples": rejected[:20],
            "eda_artifact": eda_output.get("plot_path"),
            "hypothesis_results_path": str(settings.artifacts_dir / "hypothesis_results.json"),
        }
        finish_ingestion_run(
            session,
            run_id=run.run_id,
            run_finished_at=datetime.now(tz=UTC),
            accepted_count=len(accepted_records),
            rejected_count=len(rejected),
            new_count=new_count,
            updated_count=updated_count,
            warning_count=warning_count,
            summary_path=str(settings.summary_feed_path),
            status="success",
            notes_json=json_dumps(notes),
        )

        return {
            "run_id": run.run_id,
            "status": "success",
            "accepted_count": len(accepted_records),
            "rejected_count": len(rejected),
            "new_count": new_count,
            "updated_count": updated_count,
            "warning_count": warning_count,
            "summary_feed_path": str(settings.summary_feed_path),
            "summary_geojson_path": str(settings.summary_geojson_path),
            "model_validation_plot": model_output.summary.get("validation_plot"),
            "model_timeseries_plot": model_output.summary.get("timeseries_plot"),
            "eda_results_path": str(settings.artifacts_dir / "hypothesis_results.json"),
        }
    except Exception as exc:
        finish_ingestion_run(
            session,
            run_id=run.run_id,
            run_finished_at=datetime.now(tz=UTC),
            accepted_count=0,
            rejected_count=0,
            new_count=0,
            updated_count=0,
            warning_count=0,
            summary_path=None,
            status="failed",
            notes_json=json_dumps({"error": str(exc)}),
        )
        logger.exception("Pipeline run failed for run_id=%s", run.run_id)
        raise

