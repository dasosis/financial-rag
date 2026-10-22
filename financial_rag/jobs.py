"""
jobs.py — Slow work (fetching and ingesting a filing) off the request path.

A request queues a job and gets its id straight away; GET /jobs/{id} reports
its progress. Jobs run one at a time on a single worker thread: ingestion
writes to the vector store, and one 300-page filing shouldn't run alongside
another anyway. Job records live in SQLite next to the vector store, so a
status survives the request that started it. A job still queued or running
when the process stops is marked failed at the next start, never left hanging.

  queued -> running -> done | failed
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from financial_rag import config

JOBS_DB = Path("chroma_db") / "jobs.sqlite3"

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    kind TEXT NOT NULL,
    subject TEXT NOT NULL,          -- what the job is about: a ticker, a file name
    state TEXT NOT NULL,
    step TEXT NOT NULL DEFAULT '',  -- what a running job is doing now
    result TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created REAL NOT NULL,
    updated REAL NOT NULL
)
"""

_db_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jobs")
_recovered = False


@dataclass(frozen=True)
class Job:
    id: str
    owner: str
    kind: str
    subject: str
    state: str
    step: str
    result: str         # JSON written by the job's work when it's done
    error: str
    created: float
    updated: float


def _connect() -> sqlite3.Connection:
    JOBS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(JOBS_DB)
    conn.execute(_SCHEMA)
    return conn


def _write(sql: str, args: tuple) -> None:
    with _db_lock:
        conn = _connect()
        try:
            with conn:
                conn.execute(sql, args)
        finally:
            conn.close()


def recover() -> int:
    """Fail jobs a previous process left queued or running. Returns how many."""
    with _db_lock:
        conn = _connect()
        try:
            with conn:
                cursor = conn.execute(
                    "UPDATE jobs SET state = ?, error = ?, updated = ? WHERE state IN (?, ?)",
                    (FAILED, "Interrupted by a server restart. Please try again.",
                     time.time(), QUEUED, RUNNING))
            return cursor.rowcount
        finally:
            conn.close()


def get(job_id: str) -> Job | None:
    with _db_lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT id, owner, kind, subject, state, step, result, error, created, updated "
                "FROM jobs WHERE id = ?", (job_id,)).fetchone()
        finally:
            conn.close()
    return Job(*row) if row else None


def set_step(job_id: str, step: str) -> None:
    _write("UPDATE jobs SET step = ?, updated = ? WHERE id = ?", (step, time.time(), job_id))


def submit(owner: str, kind: str, subject: str,
           work: Callable[[Callable[[str], None]], str],
           on_failure: Callable[[], None] | None = None) -> Job:
    """Queue `work` and return its job at once.

    `work(step)` runs on the worker thread; it reports progress by calling
    step("Fetching the filing …") and returns the result as a JSON string. If it
    raises, the job fails with the exception's message and `on_failure` runs
    (to refund usage, say).
    """
    global _recovered
    if not _recovered:
        recover()
        _recovered = True
    now = time.time()
    job_id = uuid.uuid4().hex
    _write("INSERT INTO jobs (id, owner, kind, subject, state, created, updated) "
           "VALUES (?, ?, ?, ?, ?, ?, ?)", (job_id, owner, kind, subject, QUEUED, now, now))

    def run() -> None:
        _write("UPDATE jobs SET state = ?, updated = ? WHERE id = ?",
               (RUNNING, time.time(), job_id))
        try:
            # The worker thread doesn't inherit the request's context: name the
            # job's owner as the consumer of its model calls here
            with config.as_consumer(config.user_consumer(owner)):
                result = work(lambda step: set_step(job_id, step))
        except Exception as exc:
            _write("UPDATE jobs SET state = ?, error = ?, step = '', updated = ? WHERE id = ?",
                   (FAILED, str(exc) or exc.__class__.__name__, time.time(), job_id))
            if on_failure is not None:
                on_failure()
            return
        _write("UPDATE jobs SET state = ?, result = ?, step = '', updated = ? WHERE id = ?",
               (DONE, result, time.time(), job_id))

    _executor.submit(run)
    return get(job_id)
