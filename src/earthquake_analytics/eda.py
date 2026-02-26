from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from .config import Settings
from .storage import load_events_frame
from .utils import haversine_km


@dataclass
class HypothesisResult:
    hypothesis_id: str
    statement: str
    metric: str
    value: float | None
    decision_rule: str
    status: str
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "metric": self.metric,
            "value": self.value,
            "decision_rule": self.decision_rule,
            "status": self.status,
            "notes": self.notes,
        }


def run_hypothesis_eda(session: Session, settings: Settings) -> dict[str, Any]:
    frame = load_events_frame(session)
    frame = frame[frame["event_type"] == "earthquake"].copy()
    now_utc = datetime.now(tz=UTC)
    frame = frame[frame["time_utc"] >= pd.Timestamp(now_utc - timedelta(days=settings.lookback_days))]

    if frame.empty:
        output = {
            "generated_at_utc": now_utc.isoformat(),
            "rows_used": 0,
            "results": [],
            "plot_path": None,
        }
        _write_hypothesis_results(output, settings.artifacts_dir / "hypothesis_results.json")
        return output

    results: list[HypothesisResult] = []

    # H1: Magnitude and significance should be positively correlated.
    h1_df = frame[frame["mag"].notna() & frame["sig"].notna()]
    corr = float(np.corrcoef(h1_df["mag"], h1_df["sig"])[0, 1]) if len(h1_df) >= 10 else None
    if corr is None:
        status = "inconclusive"
    elif corr >= 0.5:
        status = "supported"
    elif corr >= 0.2:
        status = "partially_supported"
    else:
        status = "not_supported"
    results.append(
        HypothesisResult(
            hypothesis_id="H1",
            statement="Higher magnitude events tend to have higher significance scores.",
            metric="pearson_corr(mag,sig)",
            value=corr,
            decision_rule="supported if correlation >= 0.50",
            status=status,
            notes="Uses all earthquake rows with non-null mag and sig.",
        )
    )

    # H2: Shallow events should have higher felt/significance signal than deeper events.
    shallow = frame[frame["depth_km"].notna() & (frame["depth_km"] < 30)]
    deep = frame[frame["depth_km"].notna() & (frame["depth_km"] >= 30)]
    shallow_sig = float(shallow["sig"].median()) if not shallow.empty else None
    deep_sig = float(deep["sig"].median()) if not deep.empty else None
    lift = (shallow_sig / deep_sig) if (shallow_sig is not None and deep_sig not in (None, 0)) else None
    if lift is None:
        status = "inconclusive"
    elif lift >= 1.2:
        status = "supported"
    elif lift >= 1.05:
        status = "partially_supported"
    else:
        status = "not_supported"
    results.append(
        HypothesisResult(
            hypothesis_id="H2",
            statement="Shallow earthquakes (<30 km) show stronger impact signal than deeper earthquakes.",
            metric="median_sig_ratio(shallow/deep)",
            value=float(lift) if lift is not None else None,
            decision_rule="supported if ratio >= 1.20",
            status=status,
            notes=f"shallow_median_sig={shallow_sig}, deep_median_sig={deep_sig}",
        )
    )

    # H3: Mainshocks (M>=5) are followed by a local surge in nearby event counts.
    surge_ratio = _aftershock_surge_ratio(frame)
    if surge_ratio is None:
        status = "inconclusive"
    elif surge_ratio >= 1.2:
        status = "supported"
    elif surge_ratio >= 1.0:
        status = "partially_supported"
    else:
        status = "not_supported"
    results.append(
        HypothesisResult(
            hypothesis_id="H3",
            statement="Local activity within 200 km increases in 48h after M>=5 events.",
            metric="mean((after_48h+1)/(before_48h+1))",
            value=float(surge_ratio) if surge_ratio is not None else None,
            decision_rule="supported if mean ratio >= 1.20",
            status=status,
            notes="Mainshock windows compared against pre-shock baseline in same radius.",
        )
    )

    plot_path = settings.artifacts_dir / "eda_hypothesis_plots.png"
    _save_eda_plot(frame, plot_path)

    output = {
        "generated_at_utc": now_utc.isoformat(),
        "rows_used": int(len(frame)),
        "results": [result.as_dict() for result in results],
        "plot_path": str(plot_path),
    }
    _write_hypothesis_results(output, settings.artifacts_dir / "hypothesis_results.json")
    return output


def _aftershock_surge_ratio(frame: pd.DataFrame) -> float | None:
    mainshocks = frame[
        frame["mag"].notna() & (frame["mag"] >= 5.0) & frame["latitude"].notna() & frame["longitude"].notna()
    ].copy()
    if mainshocks.empty:
        return None

    all_rows = frame[
        frame["latitude"].notna() & frame["longitude"].notna() & frame["time_utc"].notna()
    ].copy()
    if all_rows.empty:
        return None

    ratios: list[float] = []
    for mainshock in mainshocks.itertuples():
        before_start = mainshock.time_utc - pd.Timedelta(hours=48)
        before_end = mainshock.time_utc
        after_start = mainshock.time_utc
        after_end = mainshock.time_utc + pd.Timedelta(hours=48)

        window = all_rows[
            (all_rows["time_utc"] >= before_start) & (all_rows["time_utc"] <= after_end)
        ].copy()
        if window.empty:
            continue
        dists = window.apply(
            lambda row: haversine_km(
                float(mainshock.latitude),
                float(mainshock.longitude),
                float(row["latitude"]),
                float(row["longitude"]),
            ),
            axis=1,
        )
        window = window[dists <= 200.0]
        if window.empty:
            continue

        before_count = int(
            ((window["time_utc"] >= before_start) & (window["time_utc"] < before_end)).sum()
        )
        after_count = int(
            ((window["time_utc"] >= after_start) & (window["time_utc"] <= after_end)).sum()
        )
        ratios.append((after_count + 1.0) / (before_count + 1.0))

    if not ratios:
        return None
    return float(np.mean(ratios))


def _save_eda_plot(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    subset = frame[frame["mag"].notna() & frame["sig"].notna()].copy()
    sample = subset.sample(min(2500, len(subset)), random_state=7) if not subset.empty else subset
    axes[0].scatter(sample["mag"], sample["sig"], alpha=0.2, s=10)
    axes[0].set_title("Magnitude vs Significance")
    axes[0].set_xlabel("Magnitude")
    axes[0].set_ylabel("Significance")

    depth_subset = frame[frame["depth_km"].notna()]
    axes[1].hist(depth_subset["depth_km"], bins=40, color="#ff8c42", alpha=0.9)
    axes[1].set_title("Depth Distribution")
    axes[1].set_xlabel("Depth (km)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)


def _write_hypothesis_results(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
