from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class EarthquakeEvent(Base):
    __tablename__ = "earthquake_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    code: Mapped[str | None] = mapped_column(String(64))
    net: Mapped[str | None] = mapped_column(String(16), index=True)
    place: Mapped[str | None] = mapped_column(String(256), index=True)
    region_hint: Mapped[str | None] = mapped_column(String(128), index=True)
    geo_bucket: Mapped[str | None] = mapped_column(String(64), index=True)
    event_type: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[str | None] = mapped_column(String(32), index=True)
    mag: Mapped[float | None] = mapped_column(Float, index=True)
    mag_type: Mapped[str | None] = mapped_column(String(16))
    sig: Mapped[int | None] = mapped_column(Integer, index=True)
    tsunami: Mapped[int | None] = mapped_column(Integer, index=True)
    alert: Mapped[str | None] = mapped_column(String(32), index=True)
    time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    updated_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    longitude: Mapped[float | None] = mapped_column(Float, index=True)
    latitude: Mapped[float | None] = mapped_column(Float, index=True)
    depth_km: Mapped[float | None] = mapped_column(Float, index=True)
    dmin: Mapped[float | None] = mapped_column(Float)
    rms: Mapped[float | None] = mapped_column(Float)
    gap: Mapped[int | None] = mapped_column(Integer)
    nst: Mapped[int | None] = mapped_column(Integer)
    cdi: Mapped[float | None] = mapped_column(Float)
    mmi: Mapped[float | None] = mapped_column(Float)
    felt: Mapped[int | None] = mapped_column(Integer)
    sources: Mapped[str | None] = mapped_column(Text)
    ids: Mapped[str | None] = mapped_column(Text)
    types: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text)
    quality_passed: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    quality_issues: Mapped[str | None] = mapped_column(Text)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    run_finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    source_generated_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_event_count: Mapped[int | None] = mapped_column(Integer)
    accepted_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rejected_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    new_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    warning_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="running", nullable=False, index=True)
    notes_json: Mapped[str | None] = mapped_column(Text)


class EventClusterAssignment(Base):
    __tablename__ = "event_cluster_assignments"

    assignment_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ingestion_runs.run_id"), index=True, nullable=False)
    event_id: Mapped[str] = mapped_column(
        ForeignKey("earthquake_events.event_id"), index=True, nullable=False
    )
    cluster_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    is_noise: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class ClusterSummary(Base):
    __tablename__ = "cluster_summaries"

    summary_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ingestion_runs.run_id"), index=True, nullable=False)
    cluster_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    event_count: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_time_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    centroid_lat: Mapped[float | None] = mapped_column(Float)
    centroid_lon: Mapped[float | None] = mapped_column(Float)
    mean_magnitude: Mapped[float | None] = mapped_column(Float)
    max_magnitude: Mapped[float | None] = mapped_column(Float)
    region_hint: Mapped[str | None] = mapped_column(String(128), index=True)


class RegionalActivityScore(Base):
    __tablename__ = "regional_activity_scores"

    score_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ingestion_runs.run_id"), index=True, nullable=False)
    region_hint: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    hour_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
    observed_count: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_count: Mapped[float] = mapped_column(Float, nullable=False)
    z_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomalous: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


ModelDict = dict[str, Any]

