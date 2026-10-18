#!/usr/bin/env python3
"""SQLite ledger for the job-based matrix runner.

Layout: ``<runs_dir>/<universe_hash>/ledger.sqlite3``.  The database uses WAL so
that the scheduler/submitter and every job wrapper can update their own rows
concurrently.  All state transitions are guarded ``UPDATE ... WHERE status IN
(...)`` statements, so duplicate launches are harmless.
"""

import json
import os
import random
import sqlite3
import time
from pathlib import Path

from matrix_common import CASE_FIELDS

TERMINAL_STATUSES = ("success", "failed", "timeout", "error", "cancelled", "lost", "skipped")
ACTIVE_STATUSES = ("scheduling", "scheduled", "running")
BAD_STATUSES = ("failed", "timeout", "error", "cancelled", "lost")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    universe_hash TEXT NOT NULL,
    matrix_version INTEGER,
    parent_run_id TEXT,
    created_at REAL,
    started_at REAL,
    finished_at REAL,
    status TEXT DEFAULT 'pending',
    n_jobs INTEGER,
    k INTEGER,
    timeout_s REAL,
    cpus_per_job INTEGER,
    cpus_per_job_cpu INTEGER,
    cpus_per_job_gpu INTEGER,
    executor TEXT,
    partition_cpu TEXT,
    account_cpu TEXT,
    partition_gpu TEXT,
    account_gpu TEXT,
    gres_gpu TEXT,
    time_limit TEXT,
    sbatch_extra TEXT,
    filters_json TEXT,
    build_json TEXT,
    scorep_mpp TEXT,
    analysis_status TEXT DEFAULT 'pending',
    analysis_started_at REAL,
    analysis_finished_at REAL
);

CREATE TABLE IF NOT EXISTS jobs (
    run_id TEXT NOT NULL,
    job_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    params_hash TEXT,
    params_json TEXT,
    provider TEXT,
    device TEXT,
    model TEXT,
    effective_model TEXT,
    api_mode TEXT,
    api_lang TEXT,
    dl_mode TEXT,
    layout TEXT,
    scorep TEXT,
    steps INTEGER,
    clients INTEGER,
    batch_size INTEGER,
    status TEXT DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    slurm_job_id TEXT,
    slurm_state TEXT,
    scheduled_at REAL,
    started_at REAL,
    finished_at REAL,
    duration_s REAL,
    exit_code INTEGER,
    pid INTEGER,
    wrapper_pid INTEGER,
    node TEXT,
    log_path TEXT,
    slurm_log_path TEXT,
    result_json TEXT,
    result_valid INTEGER DEFAULT 0,
    cpu_solver_mb REAL,
    cpu_ml_mb REAL,
    cpu_other_mb REAL,
    cpu_total_mb REAL,
    gpu_mb REAL,
    gpu_procs_json TEXT,
    error_kind TEXT,
    error_excerpt TEXT,
    PRIMARY KEY (run_id, job_id),
    UNIQUE (run_id, seq)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    job_id INTEGER,
    ts REAL NOT NULL,
    event TEXT NOT NULL,
    detail_json TEXT
);

CREATE TABLE IF NOT EXISTS analysis (
    run_id TEXT PRIMARY KEY,
    created_at REAL,
    status TEXT,
    groups_total INTEGER,
    anomaly_count INTEGER,
    summary_json TEXT,
    report_path TEXT
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON jobs (run_id, status, seq);
CREATE INDEX IF NOT EXISTS jobs_seq_idx ON jobs (run_id, seq);
CREATE INDEX IF NOT EXISTS jobs_hash_idx ON jobs (run_id, params_hash);
CREATE INDEX IF NOT EXISTS events_run_idx ON events (run_id, ts);
"""


def universe_dir(runs_dir, universe_hash):
    return Path(runs_dir) / universe_hash


def db_path(runs_dir, universe_hash):
    return universe_dir(runs_dir, universe_hash) / "ledger.sqlite3"


def connect(path, readonly=False):
    path = Path(path)
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), timeout=30.0)
        # Do NOT set journal_mode on every connection.  Switching journal
        # modes requires an exclusive lock, and hundreds of job wrappers doing
        # that concurrently on a shared filesystem produces
        # "sqlite3.OperationalError: locking protocol".
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _retry(conn, action, attempts=12, base_delay=0.05, max_delay=2.0):
    """Retry short write transactions on transient lock errors.

    SQLite on shared filesystems can return SQLITE_BUSY or SQLITE_PROTOCOL
    while another process is committing.  The busy timeout handles most cases;
    this adds bounded exponential backoff for the rest.
    """
    retryable = ("locked", "busy", "locking protocol")
    for attempt in range(attempts):
        try:
            return action()
        except sqlite3.OperationalError as exc:
            try:
                conn.rollback()
            except Exception:
                pass
            message = str(exc).lower()
            if attempt == attempts - 1 or not any(token in message for token in retryable):
                raise
            delay = min(max_delay, base_delay * (2 ** attempt))
            time.sleep(delay + random.random() * 0.05)


def init_db(path, universe_hash=None, matrix_version=None):
    path = Path(path)
    new_db = not path.exists()
    conn = connect(path)
    if new_db:
        # Rollback journal is the default: WAL's shared-memory locks can hang
        # in Lustre's DLM under concurrent writers (SQLITE_PROTOCOL / ldlm).
        journal = os.environ.get("MATRIX_SQLITE_JOURNAL", "DELETE").strip().upper()
        if journal:
            conn.execute(f"PRAGMA journal_mode={journal}")
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "wrapper_pid" not in columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN wrapper_pid INTEGER")
    run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    for column in ("cpus_per_job_cpu", "cpus_per_job_gpu"):
        if column not in run_columns:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column} INTEGER")
    if universe_hash is not None:
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('universe_hash', ?)",
            (universe_hash,),
        )
    if matrix_version is not None:
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('matrix_version', ?)",
            (str(matrix_version),),
        )
    conn.commit()
    return conn


def create_run(conn, run):
    columns = [
        "run_id", "universe_hash", "matrix_version", "parent_run_id",
        "created_at", "started_at", "finished_at", "status", "n_jobs", "k",
        "timeout_s", "cpus_per_job", "cpus_per_job_cpu", "cpus_per_job_gpu",
        "executor",
        "partition_cpu", "account_cpu", "partition_gpu", "account_gpu",
        "gres_gpu", "time_limit", "sbatch_extra",
        "filters_json", "build_json", "scorep_mpp",
        "analysis_status", "analysis_started_at", "analysis_finished_at",
    ]
    values = [run.get(column) for column in columns]
    placeholders = ",".join("?" for _ in columns)
    _retry(conn, lambda: (
        conn.execute(
            f"INSERT INTO runs ({','.join(columns)}) VALUES ({placeholders})",
            values,
        ),
        conn.commit(),
    ))


def insert_jobs(conn, run_id, cases, seq_start=1):
    from matrix_common import effective_model

    rows = []
    for seq, case in enumerate(cases, start=seq_start):
        params = {field: case[field] for field in CASE_FIELDS}
        rows.append((
            run_id,
            int(case["job_id"]),
            int(seq),
            case["params_hash"],
            json.dumps(params, sort_keys=True),
            case["provider"],
            case["device"],
            case["model"],
            effective_model(case["provider"], case["model"], case["api_mode"]),
            case["api_mode"],
            case["api_lang"],
            case["dl_mode"],
            case["layout"],
            case["scorep"],
            int(case["steps"]),
            int(case["clients"]),
            int(case["batch_size"]),
            "pending",
        ))
    _retry(conn, lambda: (
        conn.executemany(
            """
            INSERT INTO jobs (
                run_id, job_id, seq, params_hash, params_json,
                provider, device, model, effective_model, api_mode, api_lang,
                dl_mode, layout, scorep, steps, clients, batch_size, status
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows,
        ),
        conn.commit(),
    ))
    return len(rows)


def get_run(conn, run_id):
    row = conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_job(conn, run_id, job_id):
    row = conn.execute(
        "SELECT * FROM jobs WHERE run_id=? AND job_id=?", (run_id, job_id)
    ).fetchone()
    return dict(row) if row else None


def get_job_by_seq(conn, run_id, seq):
    row = conn.execute(
        "SELECT * FROM jobs WHERE run_id=? AND seq=?", (run_id, seq)
    ).fetchone()
    return dict(row) if row else None


def get_jobs(conn, run_id, statuses=None, order_by="seq"):
    query = "SELECT * FROM jobs WHERE run_id=?"
    params = [run_id]
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        query += f" AND status IN ({placeholders})"
        params.extend(statuses)
    query += f" ORDER BY {order_by}"
    return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_counts(conn, run_id):
    counts = {}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM jobs WHERE run_id=? GROUP BY status",
        (run_id,),
    ):
        counts[row["status"]] = row["n"]
    counts["total"] = sum(counts.values())
    return counts


def count_status(conn, run_id, statuses):
    placeholders = ",".join("?" for _ in statuses)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM jobs WHERE run_id=? AND status IN ({placeholders})",
        [run_id, *statuses],
    ).fetchone()
    return row["n"]


def next_pending(conn, run_id):
    row = conn.execute(
        "SELECT * FROM jobs WHERE run_id=? AND status='pending' ORDER BY seq LIMIT 1",
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def pending_jobs(conn, run_id):
    return get_jobs(conn, run_id, statuses=("pending",))


def active_jobs(conn, run_id):
    return get_jobs(conn, run_id, statuses=ACTIVE_STATUSES)


def all_terminal(conn, run_id):
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM jobs WHERE run_id=? AND status NOT IN (?,?,?,?,?,?,?)",
        (run_id, *TERMINAL_STATUSES),
    ).fetchone()
    return row["n"] == 0


def _transition(conn, run_id, job_id, from_statuses, to_status, fields=None):
    fields = dict(fields or {})
    fields["status"] = to_status
    assignments = ", ".join(f"{column}=?" for column in fields)
    values = list(fields.values())
    placeholders = ",".join("?" for _ in from_statuses)
    values.extend([run_id, job_id])
    values.extend(from_statuses)
    def action():
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            f"UPDATE jobs SET {assignments} WHERE run_id=? AND job_id=? "
            f"AND status IN ({placeholders})",
            values,
        )
        conn.commit()
        return cursor.rowcount == 1

    return _retry(conn, action)


def mark_scheduled(conn, run_id, job_id, slurm_job_id=None, from_statuses=("pending", "scheduling")):
    return _transition(
        conn, run_id, job_id, from_statuses, "scheduled",
        {
            "scheduled_at": time.time(),
            "slurm_job_id": slurm_job_id,
        },
    )


def claim_for_submission(conn, run_id, job_id):
    return _transition(
        conn, run_id, job_id, ("pending",), "scheduling",
        {"scheduled_at": time.time()},
    )


def release_submission(conn, run_id, job_id, error=None):
    if error:
        insert_event(conn, run_id, job_id, "submit_failed", {"error": error})
    return _transition(
        conn, run_id, job_id, ("scheduling",), "pending", {"scheduled_at": None}
    )


def mark_running(conn, run_id, job_id, pid=None, node=None, from_statuses=("pending", "scheduling", "scheduled")):
    ok = _transition(
        conn, run_id, job_id, from_statuses, "running",
        {
            "started_at": time.time(),
            "pid": pid,
            "node": node,
        },
    )
    if ok:
        _retry(conn, lambda: (
            conn.execute(
                "UPDATE jobs SET attempts=attempts+1 WHERE run_id=? AND job_id=?",
                (run_id, job_id),
            ),
            conn.commit(),
        ))
    return ok


def mark_terminal(conn, run_id, job_id, status, duration_s=None, exit_code=None,
                  result_json=None, result_valid=0, error_kind=None,
                  error_excerpt=None, metrics=None):
    fields = {
        "finished_at": time.time(),
        "duration_s": duration_s,
        "exit_code": exit_code,
        "result_json": result_json,
        "result_valid": 1 if result_valid else 0,
        "error_kind": error_kind,
        "error_excerpt": error_excerpt,
        "pid": None,
        "wrapper_pid": None,
    }
    for key, value in (metrics or {}).items():
        fields[key] = value
    return _transition(conn, run_id, job_id, ACTIVE_STATUSES + ("pending",), status, fields)


def reset_to_pending(conn, run_id, job_id, reason=None):
    if reason:
        insert_event(conn, run_id, job_id, "requeued", {"reason": reason})
    return _transition(
        conn, run_id, job_id, ACTIVE_STATUSES + ("pending",), "pending",
        {
            "scheduled_at": None,
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "wrapper_pid": None,
            "slurm_job_id": None,
            "slurm_state": None,
        },
    )


def set_job_paths(conn, run_id, job_id, log_path, slurm_log_path=None):
    _retry(conn, lambda: (
        conn.execute(
            "UPDATE jobs SET log_path=?, slurm_log_path=? WHERE run_id=? AND job_id=?",
            (str(log_path) if log_path else None,
             str(slurm_log_path) if slurm_log_path else None,
             run_id, job_id),
        ),
        conn.commit(),
    ))


def set_pid(conn, run_id, job_id, pid):
    _retry(conn, lambda: (
        conn.execute(
            "UPDATE jobs SET pid=? WHERE run_id=? AND job_id=?",
            (pid, run_id, job_id),
        ),
        conn.commit(),
    ))


def set_wrapper_pid(conn, run_id, job_id, pid):
    _retry(conn, lambda: (
        conn.execute(
            "UPDATE jobs SET wrapper_pid=? WHERE run_id=? AND job_id=?",
            (pid, run_id, job_id),
        ),
        conn.commit(),
    ))


def set_slurm_state(conn, run_id, job_id, slurm_state, slurm_job_id=None):
    _retry(conn, lambda: (
        conn.execute(
            "UPDATE jobs SET slurm_state=?, slurm_job_id=COALESCE(?, slurm_job_id) "
            "WHERE run_id=? AND job_id=?",
            (slurm_state, slurm_job_id, run_id, job_id),
        ),
        conn.commit(),
    ))


def insert_event(conn, run_id, job_id, event, detail=None):
    _retry(conn, lambda: (
        conn.execute(
            "INSERT INTO events (run_id, job_id, ts, event, detail_json) VALUES (?,?,?,?,?)",
            (run_id, job_id, time.time(), event,
             json.dumps(detail, sort_keys=True) if detail is not None else None),
        ),
        conn.commit(),
    ))


def update_run_status(conn, run_id, status, finished_at=None):
    _retry(conn, lambda: (
        conn.execute(
            "UPDATE runs SET status=?, finished_at=? WHERE run_id=?",
            (status, finished_at if finished_at is not None else time.time(), run_id),
        ),
        conn.commit(),
    ))


def mark_run_started(conn, run_id):
    _retry(conn, lambda: (
        conn.execute(
            "UPDATE runs SET status='running', started_at=COALESCE(started_at, ?) WHERE run_id=?",
            (time.time(), run_id),
        ),
        conn.commit(),
    ))


def finalize_run(conn, run_id):
    run = get_run(conn, run_id)
    if run and run.get("status") == "cancelled":
        return "cancelled"
    if not all_terminal(conn, run_id):
        return None
    counts = get_counts(conn, run_id)
    bad = sum(counts.get(status, 0) for status in BAD_STATUSES)
    status = "complete" if bad == 0 else "complete_with_failures"
    update_run_status(conn, run_id, status)
    return status


def claim_analysis(conn, run_id):
    if not all_terminal(conn, run_id):
        return False
    now = time.time()
    def action():
        conn.execute("BEGIN IMMEDIATE")
        cursor = conn.execute(
            """
            UPDATE runs
               SET analysis_status='running', analysis_started_at=?
             WHERE run_id=?
               AND (analysis_status IS NULL OR analysis_status='pending')
            """,
            (now, run_id),
        )
        conn.commit()
        return cursor.rowcount == 1

    return _retry(conn, action)


def finish_analysis(conn, run_id, status, groups_total, anomaly_count,
                    summary_json, report_path):
    now = time.time()

    def action():
        conn.execute(
            "UPDATE runs SET analysis_status=?, analysis_finished_at=? WHERE run_id=?",
            (status, now, run_id),
        )
        conn.execute(
            """
            INSERT INTO analysis (
                run_id, created_at, status, groups_total, anomaly_count,
                summary_json, report_path
            ) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(run_id) DO UPDATE SET
                created_at=excluded.created_at,
                status=excluded.status,
                groups_total=excluded.groups_total,
                anomaly_count=excluded.anomaly_count,
                summary_json=excluded.summary_json,
                report_path=excluded.report_path
            """,
            (run_id, now, status, groups_total, anomaly_count,
             summary_json, str(report_path)),
        )
        conn.commit()

    _retry(conn, action)


def get_analysis(conn, run_id):
    row = conn.execute("SELECT * FROM analysis WHERE run_id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(conn):
    return [dict(row) for row in conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC"
    ).fetchall()]


def list_universe_dirs(runs_dir):
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(
        path for path in runs_dir.iterdir()
        if path.is_dir() and (path / "ledger.sqlite3").is_file()
    )


def cancel_pending(conn, run_id):
    _retry(conn, lambda: (
        conn.execute(
            "UPDATE jobs SET status='cancelled', finished_at=? "
            "WHERE run_id=? AND status='pending'",
            (time.time(), run_id),
        ),
        conn.commit(),
    ))


def stale_active(conn, run_id):
    return [
        job for job in get_jobs(conn, run_id, statuses=("scheduled", "running"))
        if job.get("slurm_job_id")
    ]


def reset_orphans(conn, run_id, active_slurm_ids):
    """Mark scheduled/running rows whose SLURM job is no longer queued.

    Returns the list of rows that were reset to pending and the list that were
    left alone because their SLURM id is still active.
    """
    reset = []
    alive = []
    for job in stale_active(conn, run_id):
        if job["slurm_job_id"] in active_slurm_ids:
            alive.append(job)
            continue
        ok = _transition(
            conn, run_id, job["job_id"], ("scheduled", "running"), "pending",
            {"started_at": None, "pid": None, "slurm_job_id": None},
        )
        if ok:
            reset.append(job)
    return reset, alive
