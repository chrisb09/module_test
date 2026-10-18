#!/usr/bin/env python3
"""Execute exactly one matrix case.

This is the process that runs inside a local scheduler slot or a SLURM job.  It
isolates the case (config copy, state directory, Score-P experiment, log file),
runs ``run.sh`` with a timeout, records the result in the ledger, submits the
next job in the SLURM chain, and triggers the consistency analysis when it is
the last job to finish.
"""

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from matrix_common import (
    CMI_DIR,
    MODULE_TEST_DIR,
    make_job_config,
    config_template_name,
    cores_for_job,
    parse_summary,
    safe_token,
)
from matrix_db import (
    connect,
    db_path,
    get_job,
    get_job_by_seq,
    get_run,
    mark_running,
    mark_terminal,
    mark_scheduled,
    claim_for_submission,
    release_submission,
    insert_event,
    set_pid,
    set_job_paths,
    all_terminal,
    finalize_run,
)
from matrix_slurm import submit_job


class ResourceMonitor:
    """Cheap per-job RSS/GPU monitor.

    It walks /proc/<pid>/task/<pid>/children instead of scanning every process,
    so hundreds of concurrent jobs do not flood the node with `ps` calls.
    """

    def __init__(self, root_pid=None, enable_gpu=False):
        self.root_pid = root_pid
        self.enable_gpu = enable_gpu
        self.max_total_kb = 0
        self.max_gpu_mb = 0
        self.running = False
        self.thread = None
        self.pids = set()
        self.page_size = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

    @staticmethod
    def _children(pid):
        try:
            with open(f"/proc/{pid}/task/{pid}/children", "r", encoding="utf-8") as handle:
                return [value for value in handle.read().split() if value]
        except Exception:
            return None

    def _rss_kb(self, pid):
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
                fields = handle.read().split()
            return int(fields[23]) * (self.page_size // 1024)
        except Exception:
            return 0

    def _tree_pids_ps(self):
        try:
            out = subprocess.check_output(
                ["ps", "-e", "-o", "pid=,ppid=,rss="], text=True
            )
        except Exception:
            return set()
        parent = {}
        rss = {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            pid, ppid, rss_kb = parts[0], parts[1], parts[2]
            parent[pid] = ppid
            rss[pid] = int(rss_kb)
        root = str(self.root_pid)
        children = {}
        for pid, ppid in parent.items():
            children.setdefault(ppid, []).append(pid)
        stack = [root]
        seen = set()
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            stack.extend(children.get(pid, []))
        self.pids = seen
        return sum(rss.get(pid, 0) for pid in seen)

    def _tree_pids(self):
        if not self.root_pid:
            return 0
        root = str(self.root_pid)
        stack = [root]
        seen = set()
        total = 0
        while stack:
            pid = stack.pop()
            if pid in seen:
                continue
            seen.add(pid)
            total += self._rss_kb(pid)
            children = self._children(pid)
            if children is None:
                # /proc/<pid>/task/<pid>/children is not available; fall back
                # to one ps scan for this sample.
                return self._tree_pids_ps()
            stack.extend(children)
        self.pids = seen
        return total

    def _gpu_memory(self):
        if not self.enable_gpu or not self.pids:
            return 0
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
        except Exception:
            return 0
        total = 0
        for line in out.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2 and parts[0] in self.pids:
                try:
                    total += int(parts[1])
                except ValueError:
                    pass
        return total

    def _loop(self):
        tick = 0
        while self.running:
            total_kb = self._tree_pids()
            if total_kb > self.max_total_kb:
                self.max_total_kb = total_kb
            if self.enable_gpu and tick % 3 == 0:
                gpu_mb = self._gpu_memory()
                if gpu_mb > self.max_gpu_mb:
                    self.max_gpu_mb = gpu_mb
            tick += 1
            time.sleep(2.0)

    def start(self, root_pid):
        self.root_pid = root_pid
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=3)


def choose_port(job_id):
    base = 7200 + (int(job_id) * 7) % 2000
    for offset in range(0, 400):
        port = base + offset
        if port > 65535:
            break
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free localhost port found")


def kill_process_group(process, pgid):
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.2)
    if process.poll() is None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def build_env(run, job, config_path, state_dir, log_path):
    env = os.environ.copy()
    env["PROVIDER"] = job["provider"]
    env["DEVICE"] = job["device"]
    env["API_MODE"] = job["api_mode"]
    env["API_LANG"] = job["api_lang"]
    env["STEPS"] = str(job["steps"])
    env["CLIENTS"] = str(job["clients"])
    env["BATCH_SIZE"] = str(job["batch_size"])
    env["COMPILE"] = "0"
    env["MODEL"] = job["effective_model"]
    env["CONFIG_FILE"] = str(config_path)
    env["USE_SCOREP"] = "1" if job["scorep"] == "on" else "0"
    env["SCOREP_MPP"] = run.get("scorep_mpp") or "none"
    if job["scorep"] == "on":
        run_dir = log_path.parent.parent
        env["SCOREP_EXPERIMENT_DIRECTORY"] = str(
            run_dir / "scorep" / f"{job['job_id']:05d}"
        )
        env["SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY"] = "true"
    env["MODULE_TEST_RUN_ID"] = f"{run['run_id']}_{job['job_id']}"
    env["MODULE_TEST_STATE_DIR"] = str(state_dir)
    env["SMARTSIM_EXP_DIR"] = str(state_dir / "smartsim")
    env["PHYDLL_REBUILD_DL_CLIENT"] = "0"
    env["PYTHONUNBUFFERED"] = "1"

    if "mmcp" in job["effective_model"]:
        env["MERGE_STRATEGY"] = "AUTO"
    elif job["provider"] == "SMARTSIM" and "MULTI" in job["api_mode"]:
        env["MERGE_STRATEGY"] = "NONE"
    else:
        env["MERGE_STRATEGY"] = "LIST"

    if job["provider"] == "PHYDLL":
        env["USE_PYTHON_DL_CLIENT"] = "1" if job["dl_mode"] == "python" else "0"
        env["PHYDLL_PY_SCOREP_WRAPPER"] = (
            "1" if job["scorep"] == "on" and job["dl_mode"] == "python" else "0"
        )
        dl_build = CMI_DIR / "dl_clients" / (
            "build-scorep-none" if job["scorep"] == "on" else "build-module-test"
        )
        env["PHYDLL_DL_BUILD_DIR"] = str(dl_build)
        if job["layout"] == "uniform_chunks" and "mmcp" in job["effective_model"]:
            field_count = 2
        else:
            field_count = 1
        env["PHYDLL_DL_FIELD_COUNT"] = str(field_count)
        env["PHYDLL_DL_COUNT"] = str(field_count)

    if job["provider"] == "SMARTSIM":
        env["MLCOUPLING_SMARTSIM_NUM_GPUS"] = "1" if job["device"] == "GPU" else "0"
        env["MLCOUPLING_SMARTSIM_FIRST_GPU"] = "0"
        env["SS_PORT"] = str(choose_port(job["job_id"]))

    if job["device"] == "GPU":
        if not env.get("CUDA_VISIBLE_DEVICES") and not os.environ.get("SLURM_JOB_ID"):
            env["CUDA_VISIBLE_DEVICES"] = "1"

    cpus = cores_for_job(run, job["device"])
    if cpus > 0:
        env["OMP_NUM_THREADS"] = str(cpus)
        env["MKL_NUM_THREADS"] = str(cpus)
        env["TORCH_NUM_THREADS"] = str(cpus)
        env["TF_NUM_INTRAOP_THREADS"] = str(cpus)
        env["TF_NUM_INTEROP_THREADS"] = "1"

    if os.environ.get("SLURM_JOB_ID"):
        env["OMPI_MCA_rmaps_base_oversubscribe"] = "1"

    env["MATRIX_LOG_PATH"] = str(log_path)
    return env


def run_case(run, job, config_path, state_dir, log_path, on_start=None):
    env = build_env(run, job, config_path, state_dir, log_path)
    timeout_s = float(run.get("timeout_s") or 900)
    header = (
        f"\n# matrix job {job['job_id']:05d} seq={job['seq']} "
        f"run={run['run_id']} universe={run['universe_hash']}\n"
        f"# provider={job['provider']} device={job['device']} "
        f"model={job['effective_model']} api={job['api_mode']}/{job['api_lang']} "
        f"dl={job['dl_mode']} layout={job['layout']} scorep={job['scorep']} "
        f"steps={job['steps']} clients={job['clients']} batch={job['batch_size']}\n"
        f"# timeout={timeout_s:.0f}s log={log_path}\n"
        f"# command=./run.sh\n\n"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(header)

    monitor = ResourceMonitor(enable_gpu=(job["device"] == "GPU"))
    start = time.time()
    timed_out = False
    output = ""
    returncode = None
    pid = None
    try:
        process = subprocess.Popen(
            ["./run.sh"],
            cwd=str(MODULE_TEST_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        pid = process.pid
        if on_start is not None:
            on_start(pid)
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = pid
        monitor.start(pid)
        try:
            output, _ = process.communicate(timeout=timeout_s)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            kill_process_group(process, pgid)
            try:
                output, _ = process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                output = ""
            returncode = 124
    finally:
        monitor.stop()
        duration = time.time() - start
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(output or "")
            if timed_out:
                handle.write(f"\n[matrix] TIMEOUT after {timeout_s:.0f}s\n")

    metrics = {
        "cpu_solver_mb": 0.0,
        "cpu_ml_mb": 0.0,
        "cpu_other_mb": 0.0,
        "cpu_total_mb": monitor.max_total_kb / 1024.0,
        "gpu_mb": float(monitor.max_gpu_mb),
        "gpu_procs_json": None,
    }
    return {
        "returncode": returncode,
        "duration": duration,
        "timed_out": timed_out,
        "output": output or "",
        "pid": pid,
        "metrics": metrics,
    }


def ensure_templates(run_dir):
    templates = run_dir / "configs" / "templates"
    templates.mkdir(parents=True, exist_ok=True)
    return templates


def schedule_successor(conn, run, job, runs_dir):
    """Submit seq+k at job start (and retry at the end). Return True if done."""
    if run.get("executor") != "slurm":
        return True
    seq = int(job["seq"])
    k = int(run.get("k") or 1)
    n = int(run.get("n_jobs") or 0)
    if seq + k > n:
        return True
    successor = get_job_by_seq(conn, run["run_id"], seq + k)
    if not successor or successor["status"] != "pending":
        return True
    if not claim_for_submission(conn, run["run_id"], successor["job_id"]):
        return True
    parent = os.environ.get("SLURM_JOB_ID")
    if not parent:
        release_submission(conn, run["run_id"], successor["job_id"],
                           error="no SLURM_JOB_ID in job environment")
        return False
    extra = shlex.split(run.get("sbatch_extra") or "")
    try:
        slurm_id = submit_job(run, successor, runs_dir, dependency=parent,
                              extra_sbatch=extra)
    except Exception as exc:
        release_submission(conn, run["run_id"], successor["job_id"], error=str(exc))
        return False
    mark_scheduled(conn, run["run_id"], successor["job_id"], slurm_id,
                   from_statuses=("scheduling",))
    insert_event(conn, run["run_id"], successor["job_id"], "submitted_by_predecessor",
                 {"predecessor_job_id": job["job_id"], "slurm_job_id": slurm_id})
    return True


def maybe_analyze(conn, db_file, run_id):
    run = get_run(conn, run_id)
    if run and run.get("status") == "cancelled":
        return
    finalize_run(conn, run_id)
    if not all_terminal(conn, run_id):
        return
    try:
        from matrix_analyze import analyze_run
        analyze_run(db_file, run_id, claim=True)
    except Exception as exc:
        insert_event(conn, run_id, None, "analysis_error", {"error": str(exc)})


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one matrix case.")
    parser.add_argument("--universe", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir).resolve()
    db_file = db_path(runs_dir, args.universe)
    conn = connect(db_file)
    run = get_run(conn, args.run_id)
    if not run:
        print(f"Unknown run: {args.run_id}", file=sys.stderr)
        return 2
    job = get_job(conn, args.run_id, args.job_id)
    if not job:
        print(f"Unknown job {args.job_id} in run {args.run_id}", file=sys.stderr)
        return 2

    if job["status"] in ("success", "failed", "timeout", "error", "cancelled", "lost", "skipped"):
        print(f"Job {args.job_id} already terminal ({job['status']})")
        return 0

    run_dir = runs_dir / args.universe / args.run_id
    log_path = run_dir / "logs" / f"{args.job_id:05d}.log"
    state_dir = run_dir / "state" / f"{args.job_id:05d}"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "smartsim").mkdir(parents=True, exist_ok=True)

    templates = ensure_templates(run_dir)
    template_name = config_template_name(job["provider"], job["device"], job["api_mode"])
    template_path = templates / template_name
    if not template_path.is_file():
        fallback = MODULE_TEST_DIR / template_name
        if fallback.is_file():
            template_path = fallback
        else:
            print(f"Missing config template: {template_path}", file=sys.stderr)
            return 2

    config_path = run_dir / "configs" / f"{args.job_id:05d}.toml"
    try:
        make_job_config(template_path, config_path, job)
    except Exception as exc:
        print(f"Failed to prepare config: {exc}", file=sys.stderr)
        mark_terminal(conn, args.run_id, args.job_id, "error",
                      error_kind="config", error_excerpt=str(exc))
        maybe_analyze(conn, db_file, args.run_id)
        return 2

    if not mark_running(conn, args.run_id, args.job_id,
                        node=socket.gethostname()):
        print(f"Job {args.job_id} was claimed by another process")
        return 0

    set_job_paths(conn, args.run_id, args.job_id, log_path, log_path)

    insert_event(conn, args.run_id, args.job_id, "started", {
        "node": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    if args.verbose:
        print(f"[matrix] job {args.job_id:05d} seq={job['seq']} "
              f"{job['provider']}/{job['device']}/{job['effective_model']} "
              f"{job['api_mode']}/{job['api_lang']} scorep={job['scorep']}")

    try:
        successor_submitted = schedule_successor(conn, run, job, runs_dir)
        result = run_case(
            run, job, config_path, state_dir, log_path,
            on_start=lambda child_pid: set_pid(conn, args.run_id, args.job_id, child_pid),
        )
        if not successor_submitted:
            # The submit QOS may only free a slot once a sibling finishes.
            for delay in (5, 15, 30):
                time.sleep(delay)
                if schedule_successor(conn, run, job, runs_dir):
                    break
        if result["pid"]:
            set_pid(conn, args.run_id, args.job_id, result["pid"])

        output = result["output"]
        summary = parse_summary(output)
        if result["timed_out"]:
            status = "timeout"
            error_kind = "timeout"
            exit_code = 124
        elif result["returncode"] == 0:
            status = "success"
            error_kind = None
            exit_code = 0
        else:
            status = "failed"
            exit_code = result["returncode"]
            error_kind = "signal" if exit_code is not None and exit_code < 0 else "nonzero_exit"
        result_valid = summary != "N/A"
        error_excerpt = None
        if status != "success":
            error_excerpt = output[-4000:] if output else None

        mark_terminal(
            conn, args.run_id, args.job_id, status,
            duration_s=result["duration"],
            exit_code=exit_code,
            result_json=summary,
            result_valid=result_valid,
            error_kind=error_kind,
            error_excerpt=error_excerpt,
            metrics=result["metrics"],
        )
        insert_event(conn, args.run_id, args.job_id, "finished", {
            "status": status,
            "duration_s": result["duration"],
            "exit_code": exit_code,
        })
        maybe_analyze(conn, db_file, args.run_id)
        return 0 if status == "success" else (124 if status == "timeout" else 1)
    except Exception:
        excerpt = traceback.format_exc()
        mark_terminal(conn, args.run_id, args.job_id, "error",
                      error_kind="exception", error_excerpt=excerpt[-4000:])
        insert_event(conn, args.run_id, args.job_id, "exception", {"error": excerpt})
        maybe_analyze(conn, db_file, args.run_id)
        return 2
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
