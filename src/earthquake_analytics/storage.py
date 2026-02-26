from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from .models import (
    ClusterSummary,
    EarthquakeEvent,
    EventClusterAssignment,
    IngestionRun,
    RegionalActivityScore,
)


def start_ingestion_run(
    session: Session,
    *,
    source_url: str,
    run_started_at: datetime,
    source_generated_utc: datetime | None,
    source_event_count: int | None,
) -> IngestionRun:
    run = IngestionRun(
        source_url=source_url,
        run_started_at=run_started_at,
        source_generated_utc=source_generated_utc,
        source_event_count=source_event_count,
        status="running",
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def finish_ingestion_run(
    session: Session,
    *,
    run_id: int,
    run_finished_at: datetime,
    accepted_count: int,
    rejected_count: int,
    new_count: int,
    updated_count: int,
    warning_count: int,
    summary_path: str | None,
    status: str,
    notes_json: str | None,
) -> None:
    run = session.get(IngestionRun, run_id)
    if run is None:
        return
    run.run_finished_at = run_finished_at
    run.accepted_count = accepted_count
    run.rejected_count = rejected_count
    run.new_count = new_count
    run.updated_count = updated_count
    run.warning_count = warning_count
    run.summary_path = summary_path
    run.status = status
    run.notes_json = notes_json
    session.commit()


def upsert_events(session: Session, records: list[dict[str, Any]]) -> tuple[int, int]:
    if not records:
        return (0, 0)

    ids = [record["event_id"] for record in records]
    existing_ids = set(
        session.scalars(select(EarthquakeEvent.event_id).where(EarthquakeEvent.event_id.in_(ids))).all()
    )
    new_count = sum(1 for event_id in ids if event_id not in existing_ids)
    updated_count = len(ids) - new_count

    dialect_name = session.bind.dialect.name if session.bind is not None else ""
    if dialect_name == "sqlite":
        # SQLite has a bound-variable limit, so large batches must be chunked.
        columns_per_row = len(EarthquakeEvent.__table__.columns)
        max_rows_per_chunk = max(1, 900 // max(columns_per_row, 1))
        for i in range(0, len(records), max_rows_per_chunk):
            chunk = records[i : i + max_rows_per_chunk]
            stmt = sqlite_insert(EarthquakeEvent).values(chunk)
            update_map = {
                col.name: getattr(stmt.excluded, col.name)
                for col in EarthquakeEvent.__table__.columns
                if col.name != "event_id"
            }
            stmt = stmt.on_conflict_do_update(index_elements=[EarthquakeEvent.event_id], set_=update_map)
            session.execute(stmt)
    else:
        for record in records:
            session.merge(EarthquakeEvent(**record))
    session.commit()
    return (new_count, updated_count)


def replace_cluster_outputs(
    session: Session,
    *,
    run_id: int,
    assignments: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    session.execute(delete(EventClusterAssignment).where(EventClusterAssignment.run_id == run_id))
    session.execute(delete(ClusterSummary).where(ClusterSummary.run_id == run_id))
    if assignments:
        session.bulk_insert_mappings(EventClusterAssignment, assignments)
    if summaries:
        session.bulk_insert_mappings(ClusterSummary, summaries)
    session.commit()


def replace_activity_scores(
    session: Session,
    *,
    run_id: int,
    scores: list[dict[str, Any]],
) -> None:
    session.execute(delete(RegionalActivityScore).where(RegionalActivityScore.run_id == run_id))
    if scores:
        session.bulk_insert_mappings(RegionalActivityScore, scores)
    session.commit()


def load_events_frame(session: Session, *, since_utc: datetime | None = None) -> pd.DataFrame:
    cols = [
        EarthquakeEvent.event_id,
        EarthquakeEvent.time_utc,
        EarthquakeEvent.updated_utc,
        EarthquakeEvent.longitude,
        EarthquakeEvent.latitude,
        EarthquakeEvent.depth_km,
        EarthquakeEvent.mag,
        EarthquakeEvent.sig,
        EarthquakeEvent.alert,
        EarthquakeEvent.tsunami,
        EarthquakeEvent.status,
        EarthquakeEvent.net,
        EarthquakeEvent.place,
        EarthquakeEvent.region_hint,
        EarthquakeEvent.geo_bucket,
        EarthquakeEvent.event_type,
        EarthquakeEvent.quality_passed,
        EarthquakeEvent.quality_issues,
    ]
    stmt = select(*cols)
    if since_utc is not None:
        stmt = stmt.where(EarthquakeEvent.time_utc >= since_utc)
    rows = session.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(columns=[col.key for col in cols])
    frame = pd.DataFrame(rows)
    frame["time_utc"] = pd.to_datetime(frame["time_utc"], utc=True)
    frame["updated_utc"] = pd.to_datetime(frame["updated_utc"], utc=True)
    return frame
