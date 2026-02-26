from __future__ import annotations

from datetime import UTC, datetime

from apscheduler.schedulers.blocking import BlockingScheduler

from .config import Settings
from .logging_utils import get_logger
from .pipeline import run_pipeline


def run_scheduler(settings: Settings | None = None) -> None:
    settings = settings or Settings.from_env()
    logger = get_logger("earthquake_analytics.scheduler")
    scheduler = BlockingScheduler(timezone="UTC")

    def _job() -> None:
        try:
            result = run_pipeline(settings)
            logger.info(
                "Run %s completed | accepted=%s rejected=%s new=%s updated=%s",
                result["run_id"],
                result["accepted_count"],
                result["rejected_count"],
                result["new_count"],
                result["updated_count"],
            )
        except Exception:  # noqa: BLE001
            logger.exception("Scheduled pipeline job failed")

    scheduler.add_job(
        _job,
        trigger="interval",
        seconds=settings.refresh_interval_seconds,
        next_run_time=datetime.now(tz=UTC),
        max_instances=1,
        coalesce=True,
    )
    logger.info("Scheduler started | refresh interval = %s sec", settings.refresh_interval_seconds)
    scheduler.start()

