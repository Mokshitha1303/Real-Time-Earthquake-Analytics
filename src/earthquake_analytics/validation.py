from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from .utils import coarse_geo_bucket, epoch_ms_to_dt, extract_region_hint, to_float, to_int


REQUIRED_PROPERTIES = {
    "code",
    "detail",
    "net",
    "place",
    "status",
    "time",
    "title",
    "type",
    "updated",
    "url",
}


@dataclass
class NormalizationResult:
    record: dict[str, Any] | None
    hard_errors: list[str]
    soft_warnings: list[str]


def validate_feed_shape(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("type") != "FeatureCollection":
        errors.append("payload.type is not FeatureCollection")
    if not isinstance(payload.get("features"), list):
        errors.append("payload.features is not a list")
    if "metadata" not in payload:
        errors.append("payload.metadata missing")
    return errors


def normalize_feature(
    feature: dict[str, Any],
    *,
    now_utc: datetime | None = None,
    max_event_age_days: int = 45,
) -> NormalizationResult:
    now_utc = now_utc or datetime.now(tz=UTC)
    hard_errors: list[str] = []
    soft_warnings: list[str] = []

    if not isinstance(feature, dict):
        return NormalizationResult(record=None, hard_errors=["feature_not_object"], soft_warnings=[])

    event_id = feature.get("id")
    if not event_id:
        hard_errors.append("missing_event_id")

    props = feature.get("properties")
    if not isinstance(props, dict):
        hard_errors.append("missing_properties")
        props = {}

    missing_props = [key for key in REQUIRED_PROPERTIES if props.get(key) is None]
    if missing_props:
        hard_errors.append(f"missing_required_properties:{','.join(sorted(missing_props))}")

    geom = feature.get("geometry")
    coords: list[Any] = []
    if not isinstance(geom, dict):
        hard_errors.append("missing_geometry")
    else:
        if geom.get("type") != "Point":
            hard_errors.append("geometry_not_point")
        raw_coords = geom.get("coordinates")
        if not isinstance(raw_coords, list) or len(raw_coords) < 3:
            hard_errors.append("invalid_coordinates")
        else:
            coords = raw_coords

    lon = to_float(coords[0]) if len(coords) >= 1 else None
    lat = to_float(coords[1]) if len(coords) >= 2 else None
    depth = to_float(coords[2]) if len(coords) >= 3 else None
    if lon is None or lat is None or depth is None:
        hard_errors.append("coordinate_type_error")
    else:
        if not (-180 <= lon <= 180):
            hard_errors.append("longitude_out_of_range")
        if not (-90 <= lat <= 90):
            hard_errors.append("latitude_out_of_range")
        if not (-20 <= depth <= 800):
            soft_warnings.append("depth_outside_typical_range")

    event_time = epoch_ms_to_dt(props.get("time"))
    updated_time = epoch_ms_to_dt(props.get("updated"))
    if event_time is None:
        hard_errors.append("invalid_event_time")
    else:
        if event_time > now_utc + timedelta(minutes=10):
            hard_errors.append("event_time_far_in_future")
        if event_time < now_utc - timedelta(days=max_event_age_days):
            soft_warnings.append("event_time_older_than_window")
    if updated_time is None:
        hard_errors.append("invalid_updated_time")

    mag = to_float(props.get("mag"))
    if mag is None:
        soft_warnings.append("missing_magnitude")
    elif not (-3.0 <= mag <= 10.0):
        soft_warnings.append("magnitude_outside_expected_range")

    gap = to_int(props.get("gap"))
    if gap is not None and gap > 300:
        soft_warnings.append("high_azimuthal_gap")
    rms = to_float(props.get("rms"))
    if rms is not None and rms > 1.0:
        soft_warnings.append("high_rms")

    event_type = props.get("type")
    if event_type and event_type != "earthquake":
        soft_warnings.append(f"non_earthquake_event:{event_type}")

    if hard_errors:
        return NormalizationResult(record=None, hard_errors=hard_errors, soft_warnings=soft_warnings)

    place = str(props.get("place")) if props.get("place") is not None else None
    region_hint = extract_region_hint(place)
    record = {
        "event_id": str(event_id),
        "code": str(props.get("code")) if props.get("code") is not None else None,
        "net": str(props.get("net")) if props.get("net") is not None else None,
        "place": place,
        "region_hint": region_hint,
        "geo_bucket": coarse_geo_bucket(lat, lon),
        "event_type": str(event_type) if event_type is not None else None,
        "status": str(props.get("status")) if props.get("status") is not None else None,
        "mag": mag,
        "mag_type": str(props.get("magType")) if props.get("magType") is not None else None,
        "sig": to_int(props.get("sig")),
        "tsunami": to_int(props.get("tsunami")),
        "alert": str(props.get("alert")) if props.get("alert") is not None else None,
        "time_utc": event_time,
        "updated_utc": updated_time,
        "longitude": lon,
        "latitude": lat,
        "depth_km": depth,
        "dmin": to_float(props.get("dmin")),
        "rms": rms,
        "gap": gap,
        "nst": to_int(props.get("nst")),
        "cdi": to_float(props.get("cdi")),
        "mmi": to_float(props.get("mmi")),
        "felt": to_int(props.get("felt")),
        "sources": str(props.get("sources")) if props.get("sources") is not None else None,
        "ids": str(props.get("ids")) if props.get("ids") is not None else None,
        "types": str(props.get("types")) if props.get("types") is not None else None,
        "title": str(props.get("title")) if props.get("title") is not None else None,
        "url": str(props.get("url")) if props.get("url") is not None else None,
        "detail": str(props.get("detail")) if props.get("detail") is not None else None,
    }
    return NormalizationResult(record=record, hard_errors=hard_errors, soft_warnings=soft_warnings)

