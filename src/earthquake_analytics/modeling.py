from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_COMPONENT_WEIGHTS: dict[str, float] = {
    "recent_mean_6": 0.2938388519,
    "recent_mean_24": 0.1642107105,
    "recent_mean_48": 0.0317200260,
    "ewma_24": 0.0669754899,
    "seasonal_mean": 0.0991426172,
    "seasonal_median": 0.3441123045,
}

SIGMA_COMPONENT_WEIGHTS: dict[str, float] = {
    "recent_std_24": 0.2406044693,
    "recent_std_48": 0.5542963997,
    "seasonal_std": 0.2050991310,
}


@dataclass
class ModelOutput:
    scores: list[dict[str, Any]]
    summary: dict[str, Any]


def _empty_summary() -> dict[str, Any]:
    return {
        "regions_modeled": 0,
        "mae": None,
        "rmse": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "balanced_accuracy": None,
        "specificity": None,
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "selected_z_threshold": None,
        "selected_min_anomaly_count": None,
        "threshold_selection_strategy": None,
        "metrics_latest_path": None,
        "metrics_history_path": None,
        "explainability_path": None,
        "model_log_path": None,
        "top_anomalies": [],
        "validation_plot": None,
        "timeseries_plot": None,
    }


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _classification_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def _weighted_nan_mean(components: list[tuple[pd.Series, float]]) -> pd.Series:
    if not components:
        return pd.Series(dtype=float)
    index = components[0][0].index
    numerator = pd.Series(0.0, index=index)
    denominator = pd.Series(0.0, index=index)
    for series, weight in components:
        valid = series.notna()
        numerator = numerator + series.fillna(0.0) * weight
        denominator = denominator + valid.astype(float) * weight
    out = numerator / denominator.replace(0, np.nan)
    return out


def _weighted_components_with_contributions(
    component_series: dict[str, pd.Series],
    weights: dict[str, float],
) -> tuple[pd.Series, pd.DataFrame]:
    if not component_series:
        return pd.Series(dtype=float), pd.DataFrame()
    index = next(iter(component_series.values())).index

    denominator = pd.Series(0.0, index=index)
    for name, weight in weights.items():
        series = component_series[name]
        denominator = denominator + series.notna().astype(float) * weight

    total = pd.Series(0.0, index=index)
    contrib_df = pd.DataFrame(index=index)
    for name, weight in weights.items():
        series = component_series[name]
        normalized_weight = (
            (series.notna().astype(float) * weight) / denominator.replace(0.0, np.nan)
        ).fillna(0.0)
        contribution = series.fillna(0.0) * normalized_weight
        contrib_df[f"contrib_{name}"] = contribution
        total = total + contribution
    return total, contrib_df


def _build_expected_and_sigma(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    lag_1 = series.shift(1)
    expected_components = {
        "recent_mean_6": lag_1.rolling(window=6, min_periods=1).mean(),
        "recent_mean_24": lag_1.rolling(window=24, min_periods=6).mean(),
        "recent_mean_48": lag_1.rolling(window=48, min_periods=12).mean(),
        "ewma_24": lag_1.ewm(span=24, adjust=False, min_periods=6).mean(),
    }
    seasonal_stack_mean = pd.concat(
        [series.shift(hours) for hours in (24, 48, 72, 96, 120, 144, 168)],
        axis=1,
    )
    expected_components["seasonal_mean"] = seasonal_stack_mean.mean(axis=1, skipna=True)
    expected_components["seasonal_median"] = seasonal_stack_mean.median(axis=1, skipna=True)
    seasonal_std = seasonal_stack_mean.std(axis=1, skipna=True)

    expected, contribution_df = _weighted_components_with_contributions(
        expected_components,
        EXPECTED_COMPONENT_WEIGHTS,
    )
    fallback_expected = lag_1.rolling(window=12, min_periods=1).mean()
    fallback_mask = expected.isna()
    expected = expected.fillna(fallback_expected)
    expected = expected.fillna(0.0).clip(lower=0.0)

    sigma_components = {
        "recent_std_24": lag_1.rolling(window=24, min_periods=6).std(),
        "recent_std_48": lag_1.rolling(window=48, min_periods=12).std(),
        "seasonal_std": seasonal_std,
    }
    sigma, _ = _weighted_components_with_contributions(
        sigma_components,
        SIGMA_COMPONENT_WEIGHTS,
    )

    poisson_floor = np.sqrt(expected + 1.0)
    sigma = sigma.fillna(poisson_floor).clip(lower=0.35)
    sigma = np.maximum(sigma.to_numpy(dtype=float), poisson_floor.to_numpy(dtype=float))
    sigma = pd.Series(sigma, index=series.index)

    explain_df = pd.DataFrame(index=series.index)
    for name, values in expected_components.items():
        explain_df[name] = values
    explain_df = pd.concat([explain_df, contribution_df], axis=1)
    explain_df["expected_fallback"] = np.where(fallback_mask, expected, 0.0)
    explain_df["poisson_floor"] = poisson_floor
    for name, values in sigma_components.items():
        explain_df[name] = values
    return expected, sigma, explain_df


def _build_target_labels(valid_pred_df: pd.DataFrame, *, min_anomaly_count: int) -> np.ndarray:
    q95 = valid_pred_df.groupby("region_hint")["observed_count"].transform("quantile", 0.95)
    y_true = (
        (valid_pred_df["observed_count"] >= q95)
        & (valid_pred_df["observed_count"] >= min_anomaly_count)
    ).astype(int)
    return y_true.to_numpy()


def _find_best_threshold(
    valid_pred_df: pd.DataFrame,
    y_true: np.ndarray,
    *,
    default_z_threshold: float,
    default_min_anomaly_count: int,
) -> dict[str, float | int]:
    z_grid = np.round(np.arange(1.0, 4.01, 0.1), 2)
    min_count_grid = [1, 2, 3, 4, 5]
    best: dict[str, float | int] = {
        "z_threshold": float(default_z_threshold),
        "min_anomaly_count": int(default_min_anomaly_count),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    observed = valid_pred_df["observed_count"].to_numpy()
    z_scores = valid_pred_df["z_score"].to_numpy()
    for min_count in min_count_grid:
        for z_thr in z_grid:
            y_pred = ((z_scores >= z_thr) & (observed >= min_count)).astype(int)
            precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
            score = f1 * 0.70 + recall * 0.30
            best_score = float(best["f1"]) * 0.70 + float(best["recall"]) * 0.30
            if score > best_score:
                best = {
                    "z_threshold": float(z_thr),
                    "min_anomaly_count": int(min_count),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                }
    return best


def _find_precision_guarded_threshold(
    valid_pred_df: pd.DataFrame,
    y_true: np.ndarray,
    *,
    baseline_threshold: dict[str, float | int],
    min_precision_gain: float = 0.0005,
) -> dict[str, float | int] | None:
    z_grid = np.round(np.arange(1.0, 4.01, 0.1), 2)
    min_count_grid = [1, 2, 3, 4, 5]

    baseline_precision = float(baseline_threshold["precision"])
    recall_floor = float(baseline_threshold["recall"])
    f1_floor = float(baseline_threshold["f1"])
    observed = valid_pred_df["observed_count"].to_numpy()
    z_scores = valid_pred_df["z_score"].to_numpy()

    best: dict[str, float | int] | None = None
    for min_count in min_count_grid:
        for z_thr in z_grid:
            y_pred = ((z_scores >= z_thr) & (observed >= min_count)).astype(int)
            stats = _classification_stats(y_true, y_pred)
            precision = float(stats["precision"])
            recall = float(stats["recall"])
            f1 = float(stats["f1"])
            if recall + 1e-12 < recall_floor:
                continue
            if f1 + 1e-12 < f1_floor:
                continue
            if precision + 1e-12 < baseline_precision + min_precision_gain:
                continue

            candidate = {
                "z_threshold": float(z_thr),
                "min_anomaly_count": int(min_count),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            if best is None:
                best = candidate
                continue
            if precision > float(best["precision"]) + 1e-12:
                best = candidate
                continue
            if abs(precision - float(best["precision"])) <= 1e-12 and f1 > float(best["f1"]) + 1e-12:
                best = candidate
                continue
            if (
                abs(precision - float(best["precision"])) <= 1e-12
                and abs(f1 - float(best["f1"])) <= 1e-12
                and recall > float(best["recall"]) + 1e-12
            ):
                best = candidate
    return best


def _calibrate_region_thresholds(
    model_df: pd.DataFrame,
    *,
    global_z_threshold: float,
    global_min_anomaly_count: int,
    base_min_anomaly_count: int,
) -> tuple[dict[str, dict[str, float | int]], dict[str, Any]]:
    region_thresholds: dict[str, dict[str, float | int]] = {}
    tested_regions = 0
    accepted_regions = 0
    f1_gains: list[float] = []
    recall_gains: list[float] = []

    for region, region_df in model_df.groupby("region_hint"):
        region_df = region_df.sort_values("hour_utc").reset_index(drop=True)
        if len(region_df) < 180:
            continue

        split_idx = int(len(region_df) * 0.80)
        if split_idx < 72 or split_idx >= len(region_df) - 24:
            continue

        train_df = region_df.iloc[:split_idx].copy()
        valid_df = region_df.iloc[split_idx:].copy()

        train_q95 = float(train_df["observed_count"].quantile(0.95))
        y_train = (
            (train_df["observed_count"] >= train_q95)
            & (train_df["observed_count"] >= base_min_anomaly_count)
        ).astype(int)
        y_valid = (
            (valid_df["observed_count"] >= train_q95)
            & (valid_df["observed_count"] >= base_min_anomaly_count)
        ).astype(int)
        if int(y_train.sum()) < 6 or int(y_valid.sum()) < 2:
            continue

        tested_regions += 1

        tuned = _find_best_threshold(
            train_df,
            y_train.to_numpy(),
            default_z_threshold=global_z_threshold,
            default_min_anomaly_count=global_min_anomaly_count,
        )
        region_z = float(tuned["z_threshold"])
        region_min = int(tuned["min_anomaly_count"])

        y_pred_region = (
            (valid_df["z_score"] >= region_z) & (valid_df["observed_count"] >= region_min)
        ).astype(int)
        y_pred_global = (
            (valid_df["z_score"] >= global_z_threshold)
            & (valid_df["observed_count"] >= global_min_anomaly_count)
        ).astype(int)
        region_stats = _classification_stats(y_valid.to_numpy(), y_pred_region.to_numpy())
        global_stats = _classification_stats(y_valid.to_numpy(), y_pred_global.to_numpy())

        f1_gain = float(region_stats["f1"]) - float(global_stats["f1"])
        recall_gain = float(region_stats["recall"]) - float(global_stats["recall"])
        precision_drop = float(global_stats["precision"]) - float(region_stats["precision"])

        # Holdout-safe acceptance rule tuned for production alert quality.
        accept = False
        if f1_gain >= 0.02:
            accept = True
        elif recall_gain >= 0.05 and precision_drop <= 0.08:
            accept = True
        elif f1_gain >= 0.0 and recall_gain >= 0.03:
            accept = True

        if accept:
            accepted_regions += 1
            f1_gains.append(f1_gain)
            recall_gains.append(recall_gain)
            region_thresholds[str(region)] = {
                "z_threshold": region_z,
                "min_anomaly_count": region_min,
                "holdout_f1_gain": f1_gain,
                "holdout_recall_gain": recall_gain,
                "holdout_precision": float(region_stats["precision"]),
                "holdout_f1": float(region_stats["f1"]),
                "holdout_recall": float(region_stats["recall"]),
            }

    summary = {
        "regions_tested": tested_regions,
        "regions_accepted": accepted_regions,
        "regions_with_custom_threshold": len(region_thresholds),
        "mean_holdout_f1_gain": float(np.mean(f1_gains)) if f1_gains else 0.0,
        "mean_holdout_recall_gain": float(np.mean(recall_gains)) if recall_gains else 0.0,
    }
    return region_thresholds, summary


def _build_row_explanation(row: Any) -> dict[str, Any]:
    component_items: list[dict[str, Any]] = []
    for component, weight in EXPECTED_COMPONENT_WEIGHTS.items():
        raw_val = getattr(row, component, None)
        contrib_val = getattr(row, f"contrib_{component}", None)
        component_items.append(
            {
                "component": component,
                "weight": float(weight),
                "raw_value": float(raw_val) if raw_val is not None and pd.notna(raw_val) else None,
                "contribution": float(contrib_val)
                if contrib_val is not None and pd.notna(contrib_val)
                else 0.0,
            }
        )
    component_items.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    top_components = component_items[:3]

    reason_flags: list[str] = []
    if getattr(row, "z_score", 0.0) >= getattr(row, "region_z_threshold", np.inf):
        reason_flags.append("z_score_above_region_threshold")
    if getattr(row, "observed_count", 0) >= getattr(row, "region_min_anomaly_count", np.inf):
        reason_flags.append("count_above_region_min")
    if getattr(row, "expected_fallback", 0.0) > 0:
        reason_flags.append("fallback_expected_used")
    if getattr(row, "is_anomalous", False):
        reason_flags.append("flagged_anomalous")

    return {
        "top_expected_components": top_components,
        "reason_flags": reason_flags,
    }


def _json_default_encoder(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _write_model_artifacts(
    *,
    summary: dict[str, Any],
    latest_scores: list[dict[str, Any]],
    artifacts_dir: Path,
    now_utc: datetime,
) -> dict[str, str]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_latest_path = artifacts_dir / "model_metrics_latest.json"
    metrics_history_path = artifacts_dir / "model_metrics_history.jsonl"
    explainability_path = artifacts_dir / "model_explainability_latest.json"
    model_log_path = artifacts_dir / "model_run.log"

    metrics_latest_payload = {
        "generated_at_utc": now_utc.isoformat(),
        "metrics": {
            "precision": summary.get("precision"),
            "recall": summary.get("recall"),
            "f1": summary.get("f1"),
            "accuracy": summary.get("accuracy"),
            "balanced_accuracy": summary.get("balanced_accuracy"),
            "mae": summary.get("mae"),
            "rmse": summary.get("rmse"),
            "tp": summary.get("tp"),
            "tn": summary.get("tn"),
            "fp": summary.get("fp"),
            "fn": summary.get("fn"),
        },
        "thresholds": {
            "selected_z_threshold": summary.get("selected_z_threshold"),
            "selected_min_anomaly_count": summary.get("selected_min_anomaly_count"),
            "region_thresholds_count": summary.get("region_thresholds_count"),
        },
        "summary": summary,
    }
    with metrics_latest_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_latest_payload, file, indent=2, default=_json_default_encoder)

    history_entry = {
        "generated_at_utc": now_utc.isoformat(),
        "precision": summary.get("precision"),
        "recall": summary.get("recall"),
        "f1": summary.get("f1"),
        "accuracy": summary.get("accuracy"),
        "balanced_accuracy": summary.get("balanced_accuracy"),
        "mae": summary.get("mae"),
        "rmse": summary.get("rmse"),
        "tp": summary.get("tp"),
        "tn": summary.get("tn"),
        "fp": summary.get("fp"),
        "fn": summary.get("fn"),
        "selected_z_threshold": summary.get("selected_z_threshold"),
        "selected_min_anomaly_count": summary.get("selected_min_anomaly_count"),
        "region_thresholds_count": summary.get("region_thresholds_count"),
    }
    with metrics_history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(history_entry, default=_json_default_encoder) + "\n")

    explainability_payload = {
        "generated_at_utc": now_utc.isoformat(),
        "top_anomalies": [score for score in latest_scores if score.get("is_anomalous")][:25],
        "top_z_regions": sorted(latest_scores, key=lambda item: float(item.get("z_score", 0.0)), reverse=True)[
            :25
        ],
    }
    with explainability_path.open("w", encoding="utf-8") as file:
        json.dump(explainability_payload, file, indent=2, default=_json_default_encoder)

    log_line = (
        f"{now_utc.isoformat()} | "
        f"precision={summary.get('precision')} recall={summary.get('recall')} "
        f"f1={summary.get('f1')} accuracy={summary.get('accuracy')} "
        f"mae={summary.get('mae')} rmse={summary.get('rmse')} "
        f"z_threshold={summary.get('selected_z_threshold')} "
        f"min_count={summary.get('selected_min_anomaly_count')} "
        f"region_thresholds={summary.get('region_thresholds_count')}"
    )
    with model_log_path.open("a", encoding="utf-8") as file:
        file.write(log_line + "\n")

    return {
        "metrics_latest_path": str(metrics_latest_path),
        "metrics_history_path": str(metrics_history_path),
        "explainability_path": str(explainability_path),
        "model_log_path": str(model_log_path),
    }


def build_regional_activity_scores(
    events: pd.DataFrame,
    *,
    now_utc: datetime,
    z_threshold: float,
    artifacts_dir: Path,
    min_anomaly_count: int = 3,
) -> ModelOutput:
    if events.empty:
        return ModelOutput(scores=[], summary=_empty_summary())

    filtered = events.copy()
    filtered = filtered[
        (filtered["event_type"] == "earthquake")
        & filtered["time_utc"].notna()
        & filtered["region_hint"].notna()
    ]
    if filtered.empty:
        return ModelOutput(scores=[], summary=_empty_summary())

    filtered["hour_utc"] = filtered["time_utc"].dt.floor("h")
    counts = (
        filtered.groupby(["region_hint", "hour_utc"])
        .size()
        .rename("observed_count")
        .reset_index()
        .sort_values(["region_hint", "hour_utc"])
    )

    all_eval_rows: list[pd.DataFrame] = []
    latest_scores: list[dict[str, Any]] = []
    top_region_series: tuple[str, pd.DataFrame] | None = None
    latest_hour = counts["hour_utc"].max()

    for region, region_df in counts.groupby("region_hint"):
        series = region_df.set_index("hour_utc")["observed_count"].sort_index()
        full_idx = pd.date_range(series.index.min(), series.index.max(), freq="h", tz="UTC")
        series = series.reindex(full_idx, fill_value=0)

        expected, sigma, explain_df = _build_expected_and_sigma(series)
        z_score = (series - expected) / sigma.replace(0, np.nan)
        z_score = z_score.fillna(0.0)

        eval_df = pd.DataFrame(
            {
                "region_hint": region,
                "hour_utc": series.index,
                "observed_count": series.values,
                "expected_count": expected.values,
                "sigma": sigma.values,
                "z_score": z_score.values,
            }
        )
        eval_df = pd.concat([eval_df, explain_df.reset_index(drop=True)], axis=1)
        all_eval_rows.append(eval_df)

        if top_region_series is None or int(series.sum()) > int(top_region_series[1]["observed_count"].sum()):
            top_region_series = (region, eval_df)

    model_df = pd.concat(all_eval_rows, ignore_index=True) if all_eval_rows else pd.DataFrame()
    valid_pred_df = model_df[model_df["expected_count"].notna()].copy()
    selected_z_threshold = float(z_threshold)
    selected_min_anomaly_count = int(min_anomaly_count)
    metrics: dict[str, float | int | None] = {
        "mae": None,
        "rmse": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "accuracy": None,
        "specificity": None,
        "balanced_accuracy": None,
        "tp": None,
        "tn": None,
        "fp": None,
        "fn": None,
    }
    tuned_threshold_result: dict[str, float | int] | None = None
    base_tuned_threshold_result: dict[str, float | int] | None = None
    region_thresholds: dict[str, dict[str, float | int]] = {}
    region_calibration_summary: dict[str, Any] = {}

    if not valid_pred_df.empty:
        abs_err = np.abs(valid_pred_df["observed_count"] - valid_pred_df["expected_count"])
        metrics["mae"] = float(abs_err.mean())
        metrics["rmse"] = float(
            np.sqrt(np.mean((valid_pred_df["observed_count"] - valid_pred_df["expected_count"]) ** 2))
        )

        y_true = _build_target_labels(valid_pred_df, min_anomaly_count=min_anomaly_count)
        base_tuned_threshold_result = _find_best_threshold(
            valid_pred_df,
            y_true,
            default_z_threshold=z_threshold,
            default_min_anomaly_count=min_anomaly_count,
        )
        precision_guarded = _find_precision_guarded_threshold(
            valid_pred_df,
            y_true,
            baseline_threshold=base_tuned_threshold_result,
        )
        tuned_threshold_result = precision_guarded or base_tuned_threshold_result

        selected_z_threshold = float(tuned_threshold_result["z_threshold"])
        selected_min_anomaly_count = int(tuned_threshold_result["min_anomaly_count"])

        region_thresholds, region_calibration_summary = _calibrate_region_thresholds(
            valid_pred_df,
            global_z_threshold=selected_z_threshold,
            global_min_anomaly_count=selected_min_anomaly_count,
            base_min_anomaly_count=min_anomaly_count,
        )

    region_z_map = {
        region: float(config["z_threshold"])
        for region, config in region_thresholds.items()
        if config.get("z_threshold") is not None
    }
    region_min_map = {
        region: int(config["min_anomaly_count"])
        for region, config in region_thresholds.items()
        if config.get("min_anomaly_count") is not None
    }
    model_df["region_z_threshold"] = model_df["region_hint"].map(region_z_map).fillna(selected_z_threshold)
    model_df["region_min_anomaly_count"] = (
        model_df["region_hint"].map(region_min_map).fillna(selected_min_anomaly_count)
    )
    model_df["is_anomalous"] = (model_df["z_score"] >= model_df["region_z_threshold"]) & (
        model_df["observed_count"] >= model_df["region_min_anomaly_count"]
    )
    valid_pred_df = model_df[model_df["expected_count"].notna()].copy()

    if not valid_pred_df.empty:
        y_true = _build_target_labels(valid_pred_df, min_anomaly_count=min_anomaly_count)
        y_pred = valid_pred_df["is_anomalous"].astype(int).to_numpy()
        cls_stats = _classification_stats(y_true, y_pred)
        metrics.update(cls_stats)

    if pd.notna(latest_hour):
        latest_rows = model_df[model_df["hour_utc"] == latest_hour].copy()
        for row in latest_rows.itertuples():
            row_explanation = _build_row_explanation(row)
            latest_scores.append(
                {
                    "region_hint": str(row.region_hint),
                    "hour_utc": pd.Timestamp(row.hour_utc).to_pydatetime(),
                    "observed_count": int(row.observed_count),
                    "expected_count": float(row.expected_count) if pd.notna(row.expected_count) else 0.0,
                    "z_score": float(row.z_score),
                    "region_z_threshold": float(row.region_z_threshold),
                    "region_min_anomaly_count": int(row.region_min_anomaly_count),
                    "is_anomalous": bool(row.is_anomalous),
                    "explanation": row_explanation,
                }
            )

    latest_scores.sort(key=lambda row: row["z_score"], reverse=True)
    top_anomalies = [score for score in latest_scores if score["is_anomalous"]][:10]
    serializable_top_anomalies: list[dict[str, Any]] = []
    for score in top_anomalies:
        score_copy = dict(score)
        hour = score_copy.get("hour_utc")
        if isinstance(hour, (pd.Timestamp, datetime)):
            score_copy["hour_utc"] = pd.Timestamp(hour).isoformat()
        serializable_top_anomalies.append(score_copy)

    validation_plot_path = artifacts_dir / "model_validation_scatter.png"
    timeseries_plot_path = artifacts_dir / "model_validation_timeseries.png"
    _save_validation_plots(
        valid_pred_df=valid_pred_df,
        top_region_series=top_region_series,
        now_utc=now_utc,
        validation_plot_path=validation_plot_path,
        timeseries_plot_path=timeseries_plot_path,
    )

    summary = {
        "regions_modeled": int(model_df["region_hint"].nunique()) if not model_df.empty else 0,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "accuracy": metrics["accuracy"],
        "specificity": metrics["specificity"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "selected_z_threshold": selected_z_threshold,
        "selected_min_anomaly_count": selected_min_anomaly_count,
        "threshold_selection_strategy": "precision_guarded",
        "base_tuned_threshold_metrics": base_tuned_threshold_result,
        "region_thresholds_count": int(len(region_thresholds)),
        "region_calibration_summary": region_calibration_summary,
        "tuned_threshold_metrics": tuned_threshold_result,
        "top_anomalies": serializable_top_anomalies,
        "latest_hour_utc": latest_hour.isoformat() if pd.notna(latest_hour) else None,
        "validation_plot": str(validation_plot_path),
        "timeseries_plot": str(timeseries_plot_path),
    }
    artifact_paths = _write_model_artifacts(
        summary=summary,
        latest_scores=latest_scores,
        artifacts_dir=artifacts_dir,
        now_utc=now_utc,
    )
    summary.update(artifact_paths)
    return ModelOutput(scores=latest_scores, summary=summary)


def _save_validation_plots(
    *,
    valid_pred_df: pd.DataFrame,
    top_region_series: tuple[str, pd.DataFrame] | None,
    now_utc: datetime,
    validation_plot_path: Path,
    timeseries_plot_path: Path,
) -> None:
    validation_plot_path.parent.mkdir(parents=True, exist_ok=True)

    if valid_pred_df.empty:
        plt.figure(figsize=(7, 4))
        plt.title("Model Validation Scatter")
        plt.text(0.5, 0.5, "Not enough history for validation.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(validation_plot_path, dpi=130)
        plt.close()
    else:
        sample = valid_pred_df.sample(min(3000, len(valid_pred_df)), random_state=42)
        plt.figure(figsize=(7, 5))
        plt.scatter(sample["expected_count"], sample["observed_count"], alpha=0.2, s=10)
        max_v = max(float(sample["expected_count"].max()), float(sample["observed_count"].max()), 1.0)
        plt.plot([0, max_v], [0, max_v], color="red", linestyle="--", linewidth=1)
        plt.xlabel("Expected count")
        plt.ylabel("Observed count")
        plt.title("Observed vs Expected Regional Activity")
        plt.tight_layout()
        plt.savefig(validation_plot_path, dpi=130)
        plt.close()

    if top_region_series is None:
        plt.figure(figsize=(9, 4))
        plt.title("Regional Activity Trend")
        plt.text(0.5, 0.5, "No region data available.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(timeseries_plot_path, dpi=130)
        plt.close()
        return

    region, region_df = top_region_series
    cutoff = pd.Timestamp(now_utc - timedelta(days=7))
    window = region_df[region_df["hour_utc"] >= cutoff]
    if window.empty:
        window = region_df.tail(168)

    plt.figure(figsize=(10, 4))
    plt.plot(window["hour_utc"], window["observed_count"], label="Observed", linewidth=1.2)
    plt.plot(window["hour_utc"], window["expected_count"], label="Expected", linewidth=1.2, alpha=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Hourly event count")
    plt.title(f"Top Region Trend: {region}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(timeseries_plot_path, dpi=130)
    plt.close()
