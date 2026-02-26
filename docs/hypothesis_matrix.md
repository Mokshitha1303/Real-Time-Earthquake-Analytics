# Hypothesis Matrix (Earthquake Analytics)

## Objective
Use hypothesis-driven EDA to guide exploratory and modeling decisions for real-time earthquake monitoring.

| Hypothesis ID | Hypothesis Statement | Metric / Test | Decision Rule | Pipeline Artifact | Current Status |
|---|---|---|---|---|---|
| H1 | Higher magnitude events tend to have higher impact significance. | Pearson correlation between `mag` and `sig`. | Supported if corr >= 0.50 | `artifacts/hypothesis_results.json` | Auto-updated |
| H2 | Shallow events (<30 km) produce stronger impact signal than deep events. | Ratio of median `sig` (shallow/deep). | Supported if ratio >= 1.20 | `artifacts/hypothesis_results.json` | Auto-updated |
| H3 | Mainshocks (M>=5) are followed by local surge activity (aftershock behavior). | Mean ratio of nearby events after vs before (48h, 200 km). | Supported if ratio >= 1.20 | `artifacts/hypothesis_results.json` | Auto-updated |

## Usage
1. Run ingestion pipeline: `eq-pipeline`
2. Run hypothesis refresh (optional standalone): `eq-eda`
3. Inspect:
   - `artifacts/hypothesis_results.json`
   - `artifacts/eda_hypothesis_plots.png`

## Notes
- Hypothesis outcomes are indicative and designed for intern-level production analytics workflows.
- Thresholds can be tuned based on regional operational goals and false-alert tolerance.

