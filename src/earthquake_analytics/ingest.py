from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import requests


def _load_geojson_from_file(path_value: str) -> dict[str, Any]:
    parsed = urlparse(path_value)
    if parsed.scheme == "file":
        file_path = Path(unquote(parsed.path.lstrip("/")))
    else:
        file_path = Path(path_value)

    if not file_path.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("GeoJSON file payload is not a JSON object.")
    return payload


def fetch_geojson(source: str, timeout_seconds: int = 30) -> dict[str, Any]:
    lower = source.lower()
    if lower.startswith("http://") or lower.startswith("https://"):
        response = requests.get(source, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Feed payload is not a JSON object.")
        return payload

    payload = _load_geojson_from_file(source)
    if not isinstance(payload, dict):
        raise ValueError("Feed payload is not a JSON object.")
    return payload
