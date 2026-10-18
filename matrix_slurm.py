#!/usr/bin/env python3
"""Small SLURM helpers used by the scheduler and by the job wrappers."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from matrix_common import MODULE_TEST_DIR, cores_for_job, safe_token

TERMINAL_SLURM_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "SPECIAL_EXIT",
    "REVOKED",
}
NON_TERMINAL_SLURM_STATES = {
    "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "RESIZING",
    "SUSPENDED", "REQUEUED", "REQUEUE_FED", "REQUEUE_HOLD",
}


def have_sbatch():
    return shutil.which("sbatch") is not None


def _format_time(seconds):
    seconds = max(60, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def default_time_limit(timeout_s, margin_min=10):
    return _format_time(int(timeout_s) + int(margin_min) * 60)


def run_dir(runs_dir, universe_hash, run_id):
    return Path(runs_dir) / universe_hash / str(run_id)


def submit_job(run, job, runs_dir, dependency=None, extra_sbatch=None):
    """Submit one matrix job and return the SLURM job id."""
    universe_hash = run["universe_hash"]
    run_id = run["run_id"]
    job_dir = run_dir(runs_dir, universe_hash, run_id)
    log_dir = job_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    device = job["device"]
    partition = run.get("partition_gpu") if device == "GPU" else run.get("partition_cpu")
    account = run.get("account_gpu") if device == "GPU" else run.get("account_cpu")
    if not partition:
        raise RuntimeError(f"No SLURM partition configured for {device} jobs")

    cpus = cores_for_job(run, job["device"])
    time_limit = run.get("time_limit") or default_time_limit(run.get("timeout_s") or 900)

    command = [
        sys.executable,
        str(MODULE_TEST_DIR / "matrix_job.py"),
        "--universe", universe_hash,
        "--run-id", run_id,
        "--job-id", str(job["job_id"]),
        "--runs-dir", str(Path(runs_dir).resolve()),
    ]

    log_file = log_dir / f"{job['job_id']:05d}.log"
    inner = shlex.join(command)
    redirected = f"exec {inner} >> {shlex.quote(str(log_file))} 2>&1"

    sbatch = [
        "sbatch", "--parsable",
        "--job-name", f"mt-{safe_token(run_id)[-12:]}-{job['job_id']:05d}",
        "--nodes", "1",
        "--ntasks", "1",
        "--cpus-per-task", str(max(cpus, int(job["clients"]))),
        "--time", str(time_limit),
        "--chdir", str(MODULE_TEST_DIR),
        # One log file per job: the wrapper appends its own output and the
        # shell redirect captures Python tracebacks/import errors too.
        "--output", "/dev/null",
        "--error", "/dev/null",
        "--export", "ALL",
        "--partition", partition,
    ]
    if account:
        sbatch += ["--account", account]
    if device == "GPU":
        sbatch += ["--gres", run.get("gres_gpu") or "gpu:1"]
    if dependency:
        sbatch += ["--dependency", f"afterany:{dependency}"]
    if extra_sbatch:
        sbatch += list(extra_sbatch)
    sbatch += ["--wrap", shlex.join(["bash", "-c", redirected])]

    proc = subprocess.run(sbatch, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed ({proc.returncode}): {' '.join(sbatch)}\n{proc.stderr.strip()}"
        )
    slurm_id = proc.stdout.strip().split(";")[0]
    if not slurm_id:
        raise RuntimeError(f"sbatch returned no job id: {proc.stdout!r}")
    return slurm_id


def squeue_ids(ids):
    ids = [str(value) for value in ids if value]
    if not ids:
        return set()
    proc = subprocess.run(
        ["squeue", "-h", "-o", "%i", "-j", ",".join(ids)],
        text=True, capture_output=True,
    )
    if proc.returncode != 0:
        return set()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def sacct_states(ids):
    ids = [str(value) for value in ids if value]
    if not ids:
        return {}
    proc = subprocess.run(
        ["sacct", "-n", "-X", "-P", "-j", ",".join(ids),
         "-o", "JobID,State,ExitCode,Elapsed"],
        text=True, capture_output=True,
    )
    if proc.returncode != 0:
        return {}
    states = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 4:
            continue
        job_id, state, exit_code, elapsed = parts[:4]
        job_id = job_id.split(".")[0]
        state = state.split()[0] if state else ""
        states[job_id] = {
            "state": state,
            "exit_code": exit_code,
            "elapsed": elapsed,
        }
    return states


def scancel(ids):
    ids = [str(value) for value in ids if value]
    if not ids:
        return
    subprocess.run(["scancel", *ids], text=True, capture_output=True)


def classify_state(state):
    """Map a SLURM state to (matrix_status, error_kind)."""
    state = (state or "").upper()
    if state == "COMPLETED":
        return "success", None
    if state in ("TIMEOUT", "DEADLINE"):
        return "timeout", "slurm_timeout"
    if state in ("CANCELLED", "REVOKED"):
        return "cancelled", "slurm_cancelled"
    if state in ("OUT_OF_MEMORY",):
        return "failed", "out_of_memory"
    if state in ("NODE_FAIL", "BOOT_FAIL", "PREEMPTED", "SPECIAL_EXIT", "FAILED"):
        return "failed", "slurm_failed"
    return "failed", "slurm_unknown"


def elapsed_to_seconds(text):
    text = (text or "").strip()
    if not text:
        return None
    if "-" in text:
        days, rest = text.split("-", 1)
        base = int(days) * 86400
    else:
        rest = text
        base = 0
    parts = rest.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(p) for p in parts)
            return base + hours * 3600 + minutes * 60 + seconds
        if len(parts) == 2:
            minutes, seconds = (int(p) for p in parts)
            return base + minutes * 60 + seconds
    except ValueError:
        return None
    return None
