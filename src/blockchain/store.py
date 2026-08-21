"""Historical feature store. SQLite, standard library only.

This is the piece the risk engine has been waiting for. Anomaly detection is a
comparison against a baseline, and a baseline is history — so until something
persisted readings over time, `src.risk` was a correct implementation with
nothing to run on.

Two design points.

**Idempotent writes.** The primary key is (protocol, metric, subject,
observed_at), so recording the same observation twice is a no-op rather than a
duplicate. Collection will be scheduled, schedules overlap and retry, and a
duplicated reading is not harmless here: it doubles that value's weight in the
baseline it later forms part of. This is the same reasoning that made evidence
ids content-addressed.

**`prior_history` is the only read used for scoring, and its name is the API.**
The risk engine requires a baseline that excludes the point being tested — an
outlier included in its own baseline suppresses its own score. Rather than
documenting that and hoping, the query that callers reach for cannot include the
current observation, because it takes the current timestamp as an exclusive
bound.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from src.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    protocol     TEXT NOT NULL,
    metric       TEXT NOT NULL,
    subject      TEXT NOT NULL DEFAULT '',
    value        REAL NOT NULL,
    observed_at  TEXT NOT NULL,
    collected_at TEXT NOT NULL,
    PRIMARY KEY (protocol, metric, subject, observed_at)
);
CREATE INDEX IF NOT EXISTS observations_series
    ON observations (protocol, metric, subject, observed_at);
"""


class Observation(BaseModel):
    model_config = ConfigDict(frozen=True)

    protocol: str
    metric: str
    subject: str = ""
    value: float
    observed_at: datetime
    collected_at: datetime


def _iso(moment: datetime) -> str:
    """UTC ISO-8601, normalised so lexical order equals chronological order.

    The column is TEXT and comparisons are string comparisons, so a mixture of
    offsets ("+00:00" and "Z") or of naive and aware values would silently break
    ordering — and a series in the wrong order produces a baseline that is not
    wrong in any way a test of the statistics would catch.
    """
    if moment.tzinfo is None:
        raise ValueError("observation timestamps must be timezone-aware")
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    target = Path(path or settings.feature_store)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def record(observations: list[Observation], path: Path | None = None) -> int:
    """Persist readings. Returns how many were newly stored.

    `INSERT OR IGNORE` rather than `REPLACE`: if a reading already exists for
    this exact (metric, subject, time), the stored one is authoritative. A later
    collection run reporting a different value for a moment already recorded is a
    source inconsistency, and quietly overwriting history to match the newest
    answer would erase the evidence of it.
    """
    if not observations:
        return 0
    rows = [
        (o.protocol, o.metric, o.subject, o.value, _iso(o.observed_at), _iso(o.collected_at))
        for o in observations
    ]
    with connect(path) as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO observations "
            "(protocol, metric, subject, value, observed_at, collected_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        return conn.total_changes - before


def prior_history(
    protocol: str,
    metric: str,
    subject: str = "",
    before: datetime | None = None,
    limit: int = 120,
    path: Path | None = None,
) -> list[float]:
    """Readings strictly BEFORE `before`, oldest first.

    The exclusive bound is the point of this function; see the module docstring.
    `limit` caps how far back a baseline reaches, so a metric that changed regime
    six months ago is not still being compared against its old behaviour.
    """
    cutoff = _iso(before) if before else _iso(datetime.now(timezone.utc))
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT value FROM observations "
            "WHERE protocol = ? AND metric = ? AND subject = ? AND observed_at < ? "
            "ORDER BY observed_at DESC LIMIT ?",
            (protocol, metric, subject, cutoff, limit),
        ).fetchall()
    return [row[0] for row in reversed(rows)]


def series_length(
    protocol: str, metric: str, subject: str = "", path: Path | None = None
) -> int:
    """How many readings exist. Drives the honest 'not enough history yet' copy."""
    with connect(path) as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM observations "
            "WHERE protocol = ? AND metric = ? AND subject = ?",
            (protocol, metric, subject),
        ).fetchone()[0]


def coverage(path: Path | None = None) -> list[dict]:
    """One row per stored series: what exists, how much, and how current."""
    with connect(path) as conn:
        rows = conn.execute(
            "SELECT protocol, metric, subject, COUNT(*), MIN(observed_at), MAX(observed_at) "
            "FROM observations GROUP BY protocol, metric, subject "
            "ORDER BY protocol, metric, subject"
        ).fetchall()
    return [
        {
            "protocol": r[0],
            "metric": r[1],
            "subject": r[2],
            "observations": r[3],
            "first": r[4],
            "latest": r[5],
        }
        for r in rows
    ]
