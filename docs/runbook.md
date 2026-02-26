# Runbook

## 1. Bootstrap
```bash
pip install -e .
```

## 2. Configure
Copy `.env.example` to `.env` and adjust if needed:
- `EARTHQUAKE_SOURCE_URL`
- `EARTHQUAKE_DB_URL`
- `REFRESH_INTERVAL_SECONDS`
- Alert/clustering/model thresholds

Example local source:
- `EARTHQUAKE_SOURCE_URL=data/all_month.geojson`

## 3. One-Time Pipeline Run
```bash
python -m earthquake_analytics
```

## 4. Continuous Every-Minute Mode
```bash
python -m earthquake_analytics.run_scheduler
```

## 5. Dashboard
```bash
python -m streamlit run dashboard/app.py
```

## 6. Hypothesis EDA Refresh
```bash
python -m earthquake_analytics.run_eda
```

## 7. Explainability And Metrics Logs
- Latest model metrics JSON: `artifacts/model_metrics_latest.json`
- Append-only model metrics history: `artifacts/model_metrics_history.jsonl`
- Latest explainability output: `artifacts/model_explainability_latest.json`
- Append-only model run log: `artifacts/model_run.log`

## 8. Troubleshooting
- No dashboard data:
  - Ensure `eq-pipeline` has run at least once.
- No anomalies detected:
  - Lower `ANOMALY_Z_THRESHOLD` or increase lookback depth.
- Too many alert events:
  - Raise `MAG_ALERT_THRESHOLD`.
- Scheduler overlap risk:
  - Prevented by APScheduler `max_instances=1` and `coalesce=True`.

## 9. GitHub Actions Scheduled Runs
- Workflow file: `.github/workflows/earthquake_pipeline.yml`
- Add repository secret:
  - `EARTHQUAKE_DB_URL=postgresql+psycopg2://...`
- Trigger options:
  - Automatic: every 5 minutes
  - Manual: GitHub Actions "Run workflow"
