from __future__ import annotations

import json

from .config import Settings
from .db import build_engine, build_session_factory, init_db
from .eda import run_hypothesis_eda
from .pipeline import run_pipeline
from .scheduler import run_scheduler


def run_pipeline_command() -> None:
    settings = Settings.from_env()
    result = run_pipeline(settings)
    print(json.dumps(result, indent=2))


def run_scheduler_command() -> None:
    settings = Settings.from_env()
    run_scheduler(settings)


def run_eda_command() -> None:
    settings = Settings.from_env()
    engine = build_engine(settings.db_url)
    init_db(engine)
    session_factory = build_session_factory(engine)
    with session_factory() as session:
        results = run_hypothesis_eda(session, settings)
    print(json.dumps(results, indent=2))

