from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any


def epoch_ms_to_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except (TypeError, ValueError, OSError):
        return None


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_region_hint(place: str | None) -> str:
    if not place:
        return "unknown"
    parts = [part.strip() for part in place.split(",") if part.strip()]
    return parts[-1] if parts else "unknown"


def coarse_geo_bucket(lat: float | None, lon: float | None, cell_size_deg: float = 2.0) -> str:
    if lat is None or lon is None:
        return "unknown"
    lat_bucket = math.floor(lat / cell_size_deg) * cell_size_deg
    lon_bucket = math.floor(lon / cell_size_deg) * cell_size_deg
    return f"lat{lat_bucket:.1f}_lon{lon_bucket:.1f}"


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c

