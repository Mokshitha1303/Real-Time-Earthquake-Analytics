from __future__ import annotations

from datetime import UTC, datetime

from earthquake_analytics.validation import normalize_feature


def _sample_feature() -> dict:
    return {
        "type": "Feature",
        "id": "test123",
        "properties": {
            "code": "123",
            "detail": "https://example.com/detail/test123.geojson",
            "dmin": 0.1,
            "felt": None,
            "gap": 55,
            "ids": ",test123,",
            "mag": 2.8,
            "magType": "ml",
            "mmi": None,
            "net": "us",
            "nst": 30,
            "place": "10 km NE of Example City, CA",
            "rms": 0.2,
            "sig": 121,
            "sources": ",us,",
            "status": "reviewed",
            "time": 1772000000000,
            "title": "M 2.8 - 10 km NE of Example City, CA",
            "tsunami": 0,
            "type": "earthquake",
            "types": ",origin,phase-data,",
            "tz": None,
            "updated": 1772000100000,
            "url": "https://example.com/event/test123",
            "alert": None,
            "cdi": None,
        },
        "geometry": {"type": "Point", "coordinates": [-121.1, 37.5, 8.2]},
    }


def test_normalize_feature_valid() -> None:
    result = normalize_feature(_sample_feature(), now_utc=datetime(2026, 2, 26, tzinfo=UTC))
    assert result.record is not None
    assert result.hard_errors == []
    assert result.record["event_id"] == "test123"
    assert result.record["region_hint"] == "CA"


def test_normalize_feature_rejects_bad_coordinates() -> None:
    feature = _sample_feature()
    feature["geometry"]["coordinates"] = [-500, 95, "bad"]
    result = normalize_feature(feature, now_utc=datetime(2026, 2, 26, tzinfo=UTC))
    assert result.record is None
    assert any("longitude_out_of_range" in err or "coordinate_type_error" in err for err in result.hard_errors)

