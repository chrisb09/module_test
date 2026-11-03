#!/usr/bin/env python3
"""Plan, submit, repair and cancel job-based matrix runs.

The scheduler is deliberately split into two layers:

* ``matrix_jobs.py submit --executor local`` keeps ``k`` wrapper processes alive.
* ``matrix_jobs.py submit --executor slurm`` seeds the first ``k`` SLURM jobs;
  every job then submits ``seq + k`` with ``--dependency=afterany:<its id>`` at
  startup.  This is exactly a sliding window of size ``k`` and survives the
  submitter exiting.  ``watch``/``resume`` repair a chain if a job dies before
  it can submit its successor.
"""

import argparse
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from matrix_common import (
    API_LANGS_ALL,
    API_MODES,
    BATCH_SIZES,
    CMI_DIR,
    DEFAULT_GPU_CORES,
    DEVICES,
    MATRIX_VERSION,
    MODELS,
    MODULE_TEST_DIR,
    PHYDLL_DL_MODES,
    PHYDLL_TRANSPORT_LAYOUTS,
    PROVIDERS,
    UNIVERSE_HASH,
    config_template_name,
    build_provenance,
    format_duration,
    resolve_cores,
    parse_workload,
    prepare_phydll_dl_client,
    run_build_dir,
    select_cases,
    solver_binary,
)
from matrix_db import (
    ACTIVE_STATUSES,
    BAD_STATUSES,
    TERMINAL_STATUSES,
    active_jobs,
    all_terminal,
    claim_for_submission,
    connect,
    count_status,
    create_run,
    db_path,
    finalize_run,
    get_counts,
    get_job,
    get_job_by_seq,
    get_jobs,
    get_run,
    init_db,
    insert_event,
    insert_jobs,
    list_universe_dirs,
    mark_run_started,
    mark_scheduled,
    mark_terminal,
    next_pending,
    pending_jobs,
    release_submission,
    reset_to_pending,
    set_slurm_state,
    set_wrapper_pid,
    stale_active,
    update_run_status,
)
from matrix_slurm import (
    NON_TERMINAL_SLURM_STATES,
    classify_state,
    default_time_limit,
    elapsed_to_seconds,
    have_sbatch,
    sacct_states,
    scancel,
    squeue_ids,
    submit_job,
)


class PreflightError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_filter_args(parser):
    parser.add_argument("--providers", nargs="+", default=None, choices=PROVIDERS,
                        help="Providers to include (default: all)")
    parser.add_argument("--devices", nargs="+", default=None, choices=DEVICES,
                        help="Devices to include (default: all)")
    parser.add_argument("--models", nargs="+", default=None, choices=MODELS,
                        help="Models to include (default: all)")
    parser.add_argument("--api-modes", nargs="+", default=None, choices=API_MODES,
                        help="API modes to include (default: all)")
    parser.add_argument("--api-langs", nargs="+", default=None, choices=API_LANGS_ALL,
                        help="Solver languages to include for all API modes (default: all)")
    parser.add_argument("--workloads", nargs="+", default=None,
                        help="Workloads as steps/clients, e.g. 1/1 5/2")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None,
                        help="Batch sizes to include (default: 1 7)")
    parser.add_argument("--scorep", nargs="+", default=None, choices=["off", "on"],
                        help="Score-P modes to include (default: off on)")
    parser.add_argument("--transport-layouts", nargs="+", default=None,
                        choices=PHYDLL_TRANSPORT_LAYOUTS,
                        help="PhyDLL transport layouts (default: packed uniform_chunks)")
    parser.add_argument("--dl-modes", nargs="+", default=None,
                        choices=PHYDLL_DL_MODES,
                        help="PhyDLL DL client modes (default: cpp python)")


def add_selection_args(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--max-cases", type=int, default=None,
                       help="Run only the first N selected cases in canonical order")
    group.add_argument("--sample", type=int, default=None,
                       help="Run a random sample of N selected cases")
    group.add_argument("--job-ids", nargs="+", type=int, default=None,
                       help="Run exactly these global job IDs (see plan --csv)")
    parser.add_argument("--ids-file", default="",
                        help="File with one global job ID per line (combines with --job-ids)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for --sample (default: 0)")


def _read_ids_file(path):
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        ids.append(int(line))
    return ids


def apply_selection(selected, args):
    """Apply --max-cases / --sample / --job-ids to the filtered cases."""
    ids = []
    if getattr(args, "job_ids", None):
        ids.extend(args.job_ids)
    if getattr(args, "ids_file", ""):
        ids.extend(_read_ids_file(args.ids_file))
    if ids:
        if (getattr(args, "max_cases", None) is not None
                or getattr(args, "sample", None) is not None):
            raise SystemExit("--job-ids/--ids-file cannot be combined with "
                             "--max-cases/--sample")
        by_id = {case["job_id"]: case for case in selected}
        missing = [job_id for job_id in ids if job_id not in by_id]
        if missing:
            raise SystemExit(f"job IDs not in the filtered selection: {missing}")
        seen = set()
        ordered = []
        for job_id in ids:
            if job_id in seen:
                continue
            seen.add(job_id)
            ordered.append(by_id[job_id])
        return ordered, {"mode": "job_ids",
                         "job_ids": [case["job_id"] for case in ordered]}

    if getattr(args, "sample", None) is not None:
        if args.sample <= 0:
            raise SystemExit("--sample must be positive")
        rng = random.Random(args.seed)
        chosen = rng.sample(selected, min(args.sample, len(selected)))
        return chosen, {"mode": "sample", "sample": args.sample,
                        "seed": args.seed,
                        "job_ids": [case["job_id"] for case in chosen]}

    if getattr(args, "max_cases", None) is not None:
        if args.max_cases <= 0:
            raise SystemExit("--max-cases must be positive")
        chosen = selected[: args.max_cases]
        return chosen, {"mode": "max_cases", "max_cases": args.max_cases}

    return selected, {"mode": "all"}


def add_resource_args(parser, rerun=False):
    parser.add_argument("--executor", choices=["local", "slurm"], default=None,
                        help="Execution backend (default: local)")
    parser.add_argument("-k", "--max-jobs", dest="k", type=int, default=None,
                        help="Maximum simultaneously scheduled jobs (default: 4)")
    parser.add_argument("--cpus-per-job", type=int, default=None,
                        help="Default CPU cores / thread cap per job (default: 4)")
    parser.add_argument("--cpus-per-job-cpu", type=int, default=None,
                        help="Cores / thread cap for CPU jobs (overrides --cpus-per-job)")
    parser.add_argument("--cpus-per-job-gpu", type=int, default=None,
                        help=f"Cores / thread cap for GPU jobs (default: {DEFAULT_GPU_CORES}; "
                             "matches a 96c/4g GPU node)")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Per-job timeout in seconds (default: 900)")
    parser.add_argument("--time", default="",
                        help="SLURM walltime (default: timeout + 10 min)")
    parser.add_argument("--partition-cpu", default="",
                        help="SLURM partition for CPU jobs")
    parser.add_argument("--account-cpu", default="",
                        help="SLURM account for CPU jobs")
    parser.add_argument("--partition-gpu", default="",
                        help="SLURM partition for GPU jobs")
    parser.add_argument("--account-gpu", default="",
                        help="SLURM account for GPU jobs")
    parser.add_argument("--gres-gpu", default="gpu:1",
                        help="SLURM GPU request per GPU job (default: gpu:1)")
    parser.add_argument("--sbatch-extra", default="",
                        help="Extra sbatch arguments (shell-quoted)")
    parser.add_argument("--scorep-mpp", default=None if rerun else os.environ.get("SCOREP_MPP", "none"),
                        help=("Score-P MPI mode (default: source run, then SCOREP_MPP or none)"
                              if rerun else "Score-P MPI mode (default: SCOREP_MPP or none)"))
    parser.add_argument("--skip-build", action="store_true",
                        help="Do not rebuild PhyDLL DL clients during preflight")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-job command/start details")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress local progress output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the selection and exit without creating a run")


def filters_from_args(args):
    workloads = None
    if args.workloads:
        workloads = [parse_workload(text) for text in args.workloads]
    return {
        "providers": args.providers,
        "devices": args.devices,
        "models": args.models,
        "api_modes": args.api_modes,
        "api_langs": args.api_langs,
        "workloads": workloads,
        "batch_sizes": args.batch_sizes,
        "scorep": args.scorep,
        "transport_layouts": args.transport_layouts,
        "dl_modes": args.dl_modes,
    }


def resolve_db(runs_dir, run_id, universe=""):
    runs_dir = Path(runs_dir)
    if universe:
        candidate = db_path(runs_dir, universe)
        return candidate if candidate.is_file() else None
    for universe_dir in list_universe_dirs(runs_dir):
        candidate = universe_dir / "ledger.sqlite3"
        conn = connect(candidate, readonly=True)
        try:
            row = conn.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        except Exception:
            row = None
        finally:
            conn.close()
        if row:
            return candidate
    return None


def load_run(args):
    runs_dir = Path(args.runs_dir).resolve()
    db_file = resolve_db(runs_dir, args.run_id, getattr(args, "universe", ""))
    if not db_file:
        raise SystemExit(f"Could not find ledger for run {args.run_id}")
    conn = connect(db_file)
    run = get_run(conn, args.run_id)
    if not run:
        conn.close()
        raise SystemExit(f"Unknown run {args.run_id}")
    return conn, run, db_file, runs_dir


def make_run_id(prefix=""):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:6]
    return f"{prefix}{stamp}-{token}" if prefix else f"{stamp}-{token}"


# ---------------------------------------------------------------------------
# Planning / templates
# ---------------------------------------------------------------------------
def print_plan(selected, show_all=False, limit=100, as_csv=False, selection=None):
    print(f"Universe: {UNIVERSE_HASH} (matrix v{MATRIX_VERSION})")
    suffix = ""
    if selection and selection.get("mode") not in (None, "all"):
        suffix = f"  [selection: {selection.get('mode')}]"
    print(f"Selected cases: {len(selected)}{suffix}")
    by_provider = {}
    for case in selected:
        by_provider.setdefault(case["provider"], []).append(case["job_id"])
    for provider in PROVIDERS:
        ids = by_provider.get(provider)
        if ids:
            print(f"  {provider:<9} {len(ids):>6} jobs  ids {min(ids):05d}..{max(ids):05d}")
    rows = selected if show_all else selected[:limit]
    if as_csv:
        print("job_id,provider,device,model,effective_model,dl_mode,layout,scorep,"
              "api_mode,api_lang,steps,clients,batch_size")
        for case in rows:
            effective = case["model"]
            if case["provider"] == "SMARTSIM" and "MULTI" in case["api_mode"] and "mmcp" not in case["model"]:
                effective = f"{case['model']}_split_flat"
            print(",".join(str(v) for v in [
                case["job_id"], case["provider"], case["device"], case["model"],
                effective, case["dl_mode"], case["layout"], case["scorep"],
                case["api_mode"], case["api_lang"], case["steps"],
                case["clients"], case["batch_size"],
            ]))
        return
    print(f"{'Job':>5} {'Seq':>5} {'Provider':<9} {'Dev':<4} {'Model':<15} "
          f"{'DL':<6} {'Layout':<14} {'SP':<3} {'API':<12} {'Lang':<7} "
          f"{'St/Cl/B':<8}")
    for seq, case in enumerate(rows, start=1):
        print(f"{case['job_id']:>5} {seq:>5} {case['provider']:<9} {case['device']:<4} "
              f"{case['model']:<15} {case['dl_mode']:<6} {case['layout']:<14} "
              f"{case['scorep']:<3} {case['api_mode']:<12} {case['api_lang']:<7} "
              f"{case['steps']}/{case['clients']}/{case['batch_size']}")
    if not show_all and len(selected) > limit:
        print(f"... {len(selected) - limit} more (use --all)")


def copy_templates(selected, run_dir):
    template_dir = Path(run_dir) / "configs" / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    names = sorted({
        config_template_name(case["provider"], case["device"], case["api_mode"])
        for case in selected
    })
    for name in names:
        source = MODULE_TEST_DIR / name
        if not source.is_file():
            raise PreflightError(f"Missing config template: {source}")
        shutil.copy2(source, template_dir / name)
    return names


def write_run_snapshot(run, selected, run_dir):
    payload = dict(run)
    payload["job_ids"] = [case["job_id"] for case in selected]
    with open(Path(run_dir) / "run.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def scorep_elf(path):
    try:
        return subprocess.check_output(["readelf", "-d", str(path)], text=True,
                                       stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreflightError(f"Cannot inspect Score-P ELF dependencies for {path}: {exc}") from exc


def preflight(selected, args, run_dir):
    scorep_cases = [case for case in selected if case["scorep"] == "on"]
    if (os.environ.get("MODULE_TEST_BUILD_DIR") and scorep_cases
            and any(case["scorep"] == "off" for case in selected)):
        raise PreflightError("MODULE_TEST_BUILD_DIR cannot be used with mixed Score-P on/off cases; "
                             "select one --scorep mode per run or unset the override")
    if scorep_cases and (args.scorep_mpp or "none") != "none":
        raise PreflightError("Score-P matrix profiling requires --scorep-mpp none: "
                             "per-rank validation is unsupported for MPI mode")
    missing_solvers = []
    pairs = {(case["api_lang"], case["scorep"]) for case in selected}
    for api_lang, scorep_mode in sorted(pairs):
        binary = solver_binary(api_lang, scorep_mode, args.scorep_mpp)
        if not binary.is_file():
            missing_solvers.append(str(binary))
    if missing_solvers:
        raise PreflightError(
            "Missing solver binaries:\n  " + "\n  ".join(missing_solvers) +
            "\nRun ./build.sh and ./build.sh --scorep first (or set MODULE_TEST_BUILD_DIR)."
        )

    for api_lang in sorted({case["api_lang"] for case in scorep_cases}):
        binary = solver_binary(api_lang, "on", args.scorep_mpp)
        dynamic = scorep_elf(binary)
        if "libscorep_measurement.so" not in dynamic:
            raise PreflightError(f"Score-P solver is not linked to the measurement runtime: {binary}; rebuild it")
        if any(case["device"] == "CPU" and case["api_lang"] == api_lang
               for case in scorep_cases):
            # Inspect embedded search paths and the standard installation without
            # relying on the submit node having compute-node driver libraries.
            import re
            paths = []
            for value in re.findall(r"(?:RPATH|RUNPATH).*?\[(.*?)\]", dynamic):
                paths.extend(value.split(":"))
            paths.extend(os.environ.get("LD_LIBRARY_PATH", "").split(":"))
            candidates = []
            for directory in paths:
                if not directory:
                    continue
                directory = directory.replace("${ORIGIN}", str(binary.parent)).replace("$ORIGIN", str(binary.parent))
                library = Path(directory) / "libAIxeleratorService.so"
                if library.is_file():
                    candidates.append(library)
            if not candidates:
                candidates.append(CMI_DIR / "extern" / "AIxeleratorService" / "INSTALL-SCOREP" / "lib" / "libAIxeleratorService.so")
            for library in candidates:
                if library.is_file() and "[libnvidia-ml.so.1]" in scorep_elf(library):
                    raise PreflightError(f"CPU Score-P may load {library}, which requires libnvidia-ml.so.1; rebuild the stale AIX installation without that dependency")

    if any(case["provider"] == "PHYDLL" for case in selected):
        for scorep_mode in sorted({case["scorep"] for case in selected
                                   if case["provider"] == "PHYDLL"}):
            if not any(case["provider"] == "PHYDLL" and case["dl_mode"] == "cpp"
                       and case["scorep"] == scorep_mode for case in selected):
                continue
            build_dir = CMI_DIR / "dl_clients" / (
                "build-module-test-scorep-none" if scorep_mode == "on" else "build-module-test"
            )
            if not args.skip_build:
                try:
                    prepare_phydll_dl_client(scorep_mode)
                except RuntimeError as exc:
                    raise PreflightError(str(exc)) from exc
            client = Path(os.environ.get("PHYDLL_DL_CLIENT") or build_dir / "phydll_dl_client")
            if not client.is_file():
                raise PreflightError(
                    f"Missing PhyDLL C++ DL client: {client}; run without --skip-build."
                )
            if scorep_mode == "on" and "libscorep_measurement.so" not in scorep_elf(client):
                raise PreflightError(f"Score-P PhyDLL C++ client is not linked to the measurement runtime: {client}; rebuild it")

    if args.executor == "slurm":
        if not have_sbatch():
            raise PreflightError("sbatch is not available")
        if any(case["device"] == "CPU" for case in selected) and not args.partition_cpu:
            raise PreflightError("CPU jobs selected but --partition-cpu is empty")
        if any(case["device"] == "GPU" for case in selected) and not args.partition_gpu:
            raise PreflightError("GPU jobs selected but --partition-gpu is empty")
    else:
        cores = os.cpu_count() or 1
        max_per_job = max(args.cpus_per_job_cpu or 0, args.cpus_per_job_gpu or 0)
        if args.k * max_per_job > cores:
            print(f"[preflight] warning: k*max(cpus_cpu,cpus_gpu)="
                  f"{args.k * max_per_job} exceeds {cores} local cores",
                  file=sys.stderr)
    if (any(case["device"] == "GPU" for case in selected)
            and (args.cpus_per_job_gpu or 0) < DEFAULT_GPU_CORES):
        print(f"[preflight] warning: a 96c/4g GPU node bills one GPU as "
              f"{DEFAULT_GPU_CORES} cores, but --cpus-per-job-gpu is "
              f"{args.cpus_per_job_gpu}.", file=sys.stderr)


def build_run_record(args, selected, run_id, parent_run_id=None, selection=None):
    timeout_s = float(args.timeout)
    time_limit = args.time or default_time_limit(timeout_s)
    return {
        "run_id": run_id,
        "universe_hash": UNIVERSE_HASH,
        "matrix_version": MATRIX_VERSION,
        "parent_run_id": parent_run_id,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "status": "pending",
        "n_jobs": len(selected),
        "k": int(args.k),
        "timeout_s": timeout_s,
        "cpus_per_job": int(args.cpus_per_job),
        "cpus_per_job_cpu": int(args.cpus_per_job_cpu),
        "cpus_per_job_gpu": int(args.cpus_per_job_gpu),
        "executor": args.executor,
        "partition_cpu": args.partition_cpu or None,
        "account_cpu": args.account_cpu or None,
        "partition_gpu": args.partition_gpu or None,
        "account_gpu": args.account_gpu or None,
        "gres_gpu": args.gres_gpu or "gpu:1",
        "time_limit": time_limit,
        "sbatch_extra": args.sbatch_extra or "",
        "filters_json": json.dumps(
            {**filters_from_args(args), "selection": selection or {"mode": "all"}},
            sort_keys=True,
        ),
        "build_json": json.dumps(build_provenance(), sort_keys=True),
        "scorep_mpp": args.scorep_mpp,
        "analysis_status": "pending",
        "analysis_started_at": None,
        "analysis_finished_at": None,
    }


def create_run_and_jobs(run, selected, runs_dir):
    run_dir = Path(runs_dir) / run["universe_hash"] / run["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    copy_templates(selected, run_dir)
    universe_root = Path(runs_dir) / run["universe_hash"]
    db_file = db_path(runs_dir, run["universe_hash"])
    conn = init_db(db_file, run["universe_hash"], run["matrix_version"])
    if get_run(conn, run["run_id"]):
        conn.close()
        raise SystemExit(f"Run {run['run_id']} already exists")
    create_run(conn, run)
    insert_jobs(conn, run["run_id"], selected)
    write_run_snapshot(run, selected, run_dir)
    return conn, run_dir, db_file


# ---------------------------------------------------------------------------
# Local scheduler
# ---------------------------------------------------------------------------
def launch_local(job, run, runs_dir, verbose):
    command = [
        sys.executable,
        str(MODULE_TEST_DIR / "matrix_job.py"),
        "--universe", run["universe_hash"],
        "--run-id", run["run_id"],
        "--job-id", str(job["job_id"]),
        "--runs-dir", str(Path(runs_dir).resolve()),
    ]
    if verbose:
        command.append("--verbose")
    return subprocess.Popen(
        command, cwd=str(MODULE_TEST_DIR), env=os.environ.copy(),
        start_new_session=True,
    )


def run_local_scheduler(conn, run, runs_dir, db_file, verbose=False,
                        quiet=False, poll=0.5):
    run_id = run["run_id"]
    k = max(1, int(run["k"] or 1))
    mark_run_started(conn, run_id)
    procs = {}
    interrupted = False
    start_ts = time.time()
    last_heartbeat = start_ts
    if not quiet:
        print(f"[local] {run_id}: launching up to {k} concurrent job(s)", flush=True)
    try:
        while True:
            pending = pending_jobs(conn, run_id)
            while len(procs) < k and pending:
                job = pending.pop(0)
                try:
                    procs[job["job_id"]] = launch_local(job, run, runs_dir, verbose)
                    set_wrapper_pid(conn, run_id, job["job_id"], procs[job["job_id"]].pid)
                except Exception as exc:
                    mark_terminal(conn, run_id, job["job_id"], "error",
                                  error_kind="launch", error_excerpt=str(exc))
                    insert_event(conn, run_id, job["job_id"], "launch_failed",
                                 {"error": str(exc)})

            for job_id, process in list(procs.items()):
                returncode = process.poll()
                if returncode is None:
                    continue
                del procs[job_id]
                row = get_job(conn, run_id, job_id)
                if row and row["status"] not in TERMINAL_STATUSES:
                    mark_terminal(
                        conn, run_id, job_id, "error",
                        error_kind="wrapper_exit",
                        error_excerpt=f"wrapper exited with {returncode} "
                                      f"without recording a terminal state",
                    )
                    row = get_job(conn, run_id, job_id)
                if not quiet:
                    counts = get_counts(conn, run_id)
                    done = sum(counts.get(status, 0) for status in TERMINAL_STATUSES)
                    status = (row or {}).get("status", "?")
                    duration = (row or {}).get("duration_s")
                    duration_txt = f" {duration:.1f}s" if duration is not None else ""
                    print(f"[{done:>5}/{counts['total']:<5}] job {job_id:05d} "
                          f"{status}{duration_txt}  "
                          f"running={counts.get('running', 0)} "
                          f"pending={counts.get('pending', 0)}", flush=True)
                    last_heartbeat = time.time()

            if all_terminal(conn, run_id):
                break
            if not procs and not pending_jobs(conn, run_id):
                break

            now = time.time()
            if not quiet and now - last_heartbeat >= 20:
                counts = get_counts(conn, run_id)
                done = sum(counts.get(status, 0) for status in TERMINAL_STATUSES)
                print(f"[{done:>5}/{counts['total']:<5}] "
                      f"running={counts.get('running', 0)} "
                      f"pending={counts.get('pending', 0)}  "
                      f"elapsed={format_duration(now - start_ts)}", flush=True)
                last_heartbeat = now
            time.sleep(poll)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[local] interrupted; waiting for active jobs "
              "(Ctrl-C again to abandon)...", flush=True)
        for process in procs.values():
            try:
                process.wait()
            except KeyboardInterrupt:
                for active in procs.values():
                    try:
                        os.killpg(os.getpgid(active.pid), signal.SIGTERM)
                    except Exception:
                        pass
                break
    finally:
        finalize_run(conn, run_id)
        current = get_run(conn, run_id)
        if not interrupted and (not current or current.get("status") != "cancelled"):
            try:
                from matrix_analyze import analyze_run
                analyze_run(db_file, run_id, claim=True)
            except Exception as exc:
                insert_event(conn, run_id, None, "analysis_error", {"error": str(exc)})
    return get_counts(conn, run_id)

# ---------------------------------------------------------------------------
# SLURM scheduler
# ---------------------------------------------------------------------------
def _submit_one(conn, run, job, runs_dir, dependency=None):
    if not claim_for_submission(conn, run["run_id"], job["job_id"]):
        return False
    extra = shlex.split(run.get("sbatch_extra") or "")
    try:
        slurm_id = submit_job(run, job, runs_dir, dependency=dependency,
                              extra_sbatch=extra)
    except Exception as exc:
        release_submission(conn, run["run_id"], job["job_id"], error=str(exc))
        return False
    mark_scheduled(conn, run["run_id"], job["job_id"], slurm_id,
                   from_statuses=("scheduling",))
    insert_event(conn, run["run_id"], job["job_id"], "submitted",
                 {"slurm_job_id": slurm_id, "dependency": dependency})
    return True


def seed_slurm(conn, run, runs_dir):
    run_id = run["run_id"]
    k = max(1, int(run["k"] or 1))
    submitted = 0
    for job in pending_jobs(conn, run_id):
        if job["seq"] > k:
            continue
        if _submit_one(conn, run, job, runs_dir):
            submitted += 1
    return submitted


def repair_chain(conn, run, runs_dir):
    run_id = run["run_id"]
    k = max(1, int(run["k"] or 1))
    active = count_status(conn, run_id, ACTIVE_STATUSES)
    slots = k - active
    if slots <= 0:
        return 0
    submitted = 0
    for job in pending_jobs(conn, run_id):
        if job["seq"] <= k:
            dependency = None
        else:
            predecessor = get_job_by_seq(conn, run_id, job["seq"] - k)
            if not predecessor or predecessor["status"] not in TERMINAL_STATUSES:
                continue
            dependency = predecessor.get("slurm_job_id")
        if _submit_one(conn, run, job, runs_dir, dependency=dependency):
            submitted += 1
            slots -= 1
            if slots <= 0:
                break
    return submitted


def _exit_code(value):
    if not value:
        return None
    try:
        return int(str(value).split(":")[0])
    except (ValueError, TypeError):
        return None


def recover_scheduling(conn, run, force=False, stale_after=300.0):
    """Reset 'scheduling' rows whose submitter disappeared before sbatch."""
    reset = 0
    now = time.time()
    for job in get_jobs(conn, run["run_id"], statuses=("scheduling",)):
        if job.get("slurm_job_id"):
            continue
        scheduled_at = job.get("scheduled_at") or 0
        if force or (scheduled_at and now - scheduled_at > stale_after):
            if reset_to_pending(conn, run["run_id"], job["job_id"],
                                reason="stale scheduling claim"):
                reset += 1
    return reset


def reconcile_slurm(conn, run, missing_counts, force_missing=False):
    run_id = run["run_id"]
    stale = stale_active(conn, run_id)
    ids = [job["slurm_job_id"] for job in stale]
    queued = squeue_ids(ids)
    missing = [job["slurm_job_id"] for job in stale if job["slurm_job_id"] not in queued]
    states = sacct_states(missing)
    changed = 0
    for job in stale:
        slurm_id = job["slurm_job_id"]
        if slurm_id in queued:
            missing_counts.pop(slurm_id, None)
            continue
        info = states.get(slurm_id)
        state = (info or {}).get("state", "")
        if state:
            set_state = state.upper()
            set_slurm_state(conn, run_id, job["job_id"], set_state, slurm_id)
            missing_counts.pop(slurm_id, None)
            if set_state not in NON_TERMINAL_SLURM_STATES:
                status, kind = classify_state(set_state)
                mark_terminal(
                    conn, run_id, job["job_id"], status,
                    duration_s=elapsed_to_seconds(info.get("elapsed")),
                    exit_code=_exit_code(info.get("exit_code")),
                    error_kind=kind,
                    error_excerpt=f"SLURM state {set_state}",
                )
                insert_event(conn, run_id, job["job_id"], "slurm_terminal",
                             {"state": set_state, "slurm_job_id": slurm_id})
                changed += 1
            continue
        missing_counts[slurm_id] = missing_counts.get(slurm_id, 0) + 1
        if missing_counts[slurm_id] >= (1 if force_missing else 3):
            mark_terminal(
                conn, run_id, job["job_id"], "lost",
                error_kind="slurm_missing",
                error_excerpt=f"SLURM job {slurm_id} disappeared from squeue and sacct",
            )
            insert_event(conn, run_id, job["job_id"], "slurm_lost",
                         {"slurm_job_id": slurm_id})
            missing_counts.pop(slurm_id, None)
            changed += 1
    if changed:
        finalize_run(conn, run_id)
    return changed


def watch_loop(conn, run, runs_dir, db_file, interval=60.0, verbose=False, once=False):
    run_id = run["run_id"]
    missing_counts = {}
    mark_run_started(conn, run_id)
    iteration = 0
    while True:
        current = get_run(conn, run_id)
        if current and current.get("status") == "cancelled":
            break
        recover_scheduling(conn, run)
        repair_chain(conn, run, runs_dir)
        reconcile_slurm(conn, run, missing_counts)
        finalize_run(conn, run_id)
        counts = get_counts(conn, run_id)
        if verbose or iteration == 0:
            done = sum(counts.get(status, 0) for status in TERMINAL_STATUSES)
            print(f"[{done:>5}/{counts['total']:<5}] "
                  f"ok={counts.get('success', 0)} "
                  f"bad={sum(counts.get(status, 0) for status in BAD_STATUSES)} "
                  f"active={sum(counts.get(status, 0) for status in ACTIVE_STATUSES)} "
                  f"pending={counts.get('pending', 0)}", flush=True)
        if all_terminal(conn, run_id):
            break
        if once:
            return counts
        iteration += 1
        time.sleep(interval)

    current = get_run(conn, run_id)
    if not current or current.get("status") != "cancelled":
        try:
            from matrix_analyze import analyze_run
            analyze_run(db_file, run_id, claim=True)
        except Exception as exc:
            insert_event(conn, run_id, None, "analysis_error", {"error": str(exc)})
    return get_counts(conn, run_id)


def compact_run_directory(run_dir, archive_scorep=False):
    """Reclaim filesystem inodes by removing transient state, configs, and Score-P boilerplate."""
    run_dir = Path(run_dir).resolve()
    removed_files = 0
    removed_dirs = 0
    scorep_dir = run_dir / "scorep"
    tar_path = run_dir / "scorep.tar"
    if archive_scorep and scorep_dir.is_dir() and any(scorep_dir.iterdir()) and tar_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing archive: {tar_path}")

    # 1. State directory: remove transient socket/endpoint/telemetry files and empty dirs
    state_dir = run_dir / "state"
    if state_dir.is_dir():
        for job_dir in list(state_dir.iterdir()):
            if job_dir.is_dir():
                shutil.rmtree(job_dir, ignore_errors=True)
                removed_dirs += 1
        try:
            state_dir.rmdir()
            removed_dirs += 1
        except OSError:
            pass

    # 2. Configs: remove per-job TOML files (keeping templates/ directory)
    configs_dir = run_dir / "configs"
    if configs_dir.is_dir():
        for toml in configs_dir.glob("*.toml"):
            try:
                toml.unlink()
                removed_files += 1
            except OSError:
                pass

    # 3. Score-P boilerplate: remove MANIFEST.md and scorep.cfg from rank dirs that have profile.cubex
    if scorep_dir.is_dir():
        for rank_dir in scorep_dir.glob("*_rank_*"):
            if rank_dir.is_dir() and (rank_dir / "profile.cubex").is_file():
                for name in ("MANIFEST.md", "scorep.cfg"):
                    p = rank_dir / name
                    if p.is_file():
                        try:
                            p.unlink()
                            removed_files += 1
                        except OSError:
                            pass
        if archive_scorep and any(scorep_dir.iterdir()):
            try:
                subprocess.run(
                    ["tar", "-cf", str(tar_path), "-C", str(run_dir), "scorep"],
                    check=True, capture_output=True, text=True,
                )
                shutil.rmtree(scorep_dir, ignore_errors=True)
                removed_dirs += 1
            except Exception as exc:
                raise RuntimeError(f"archiving scorep failed: {exc}") from exc

    return removed_files, removed_dirs


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_plan(args):
    selected = select_cases(**filters_from_args(args))
    selected, selection = apply_selection(selected, args)
    print_plan(selected, show_all=args.all, limit=args.display_limit,
               as_csv=args.csv, selection=selection)
    return 0


def cmd_submit(args):
    args.executor = args.executor or "local"
    args.k = 4 if args.k is None else args.k
    (args.cpus_per_job, args.cpus_per_job_cpu,
     args.cpus_per_job_gpu) = resolve_cores(
        args.cpus_per_job, args.cpus_per_job_cpu, args.cpus_per_job_gpu
    )
    args.timeout = 900.0 if args.timeout is None else args.timeout
    selected = select_cases(**filters_from_args(args))
    selected, selection = apply_selection(selected, args)
    if not selected:
        print("No cases matched the filters/selection", file=sys.stderr)
        return 2
    if args.dry_run:
        print_plan(selected, show_all=True, selection=selection)
        return 0
    if args.k < 1:
        print("k must be >= 1", file=sys.stderr)
        return 2
    if args.timeout <= 0 or min(args.cpus_per_job_cpu, args.cpus_per_job_gpu) <= 0:
        print("timeout and cpus-per-job values must be positive", file=sys.stderr)
        return 2

    run_id = args.run_id or make_run_id()
    if "/" in run_id or "\\" in run_id or run_id in (".", ".."):
        print(f"Invalid run id: {run_id!r}", file=sys.stderr)
        return 2
    run = build_run_record(args, selected, run_id, selection=selection)
    runs_dir = Path(args.runs_dir).resolve()
    conn, run_dir, db_file = create_run_and_jobs(run, selected, runs_dir)
    print(f"Run: {run_id}")
    print(f"Universe: {run['universe_hash']}  jobs: {len(selected)}  "
          f"selection: {selection.get('mode')}  executor: {run['executor']}  "
          f"k={run['k']}  cores cpu={run['cpus_per_job_cpu']} "
          f"gpu={run['cpus_per_job_gpu']} (base={run['cpus_per_job']})")
    print(f"Ledger: {db_file}")
    try:
        preflight(selected, args, run_dir)
    except PreflightError as exc:
        update_run_status(conn, run_id, "error")
        print(f"Preflight failed: {exc}", file=sys.stderr)
        conn.close()
        return 2

    mark_run_started(conn, run_id)
    if args.executor == "slurm":
        submitted = seed_slurm(conn, run, runs_dir)
        print(f"Seeded {submitted} SLURM job(s); each will queue seq+k with "
              f"--dependency=afterany.")
        remaining_seeds = [
            job for job in pending_jobs(conn, run["run_id"])
            if job["seq"] <= args.k
        ]
        if remaining_seeds:
            print(f"warning: {len(remaining_seeds)} seed job(s) could not be "
                  f"submitted yet (queue/QOS limit). Run "
                  f"'python3 matrix_jobs.py watch {run['run_id']}' to repair.",
                  file=sys.stderr)
        if args.watch:
            watch_loop(conn, run, runs_dir, db_file, interval=args.interval,
                       verbose=args.verbose)
        else:
            print(f"Watch: python3 matrix_status.py --run-id {run_id}")
            print(f"Repair: python3 matrix_jobs.py watch {run_id}")
        conn.close()
        return 0

    counts = run_local_scheduler(conn, run, runs_dir, db_file,
                                 verbose=args.verbose, quiet=args.quiet)
    print(f"Finished: {counts.get('success', 0)} success, "
          f"{counts.get('failed', 0)} failed, {counts.get('timeout', 0)} timeout, "
          f"{counts.get('error', 0)} error")
    conn.close()
    return 0


def cmd_watch(args):
    conn, run, db_file, runs_dir = load_run(args)
    if run["executor"] != "slurm":
        print("watch is for SLURM runs; use the local scheduler/resume instead",
              file=sys.stderr)
        conn.close()
        return 2
    counts = watch_loop(conn, run, runs_dir, db_file, interval=args.interval,
                        verbose=True)
    print(f"Finished: {counts.get('success', 0)} success, "
          f"{sum(counts.get(status, 0) for status in BAD_STATUSES)} unsuccessful")
    conn.close()
    return 0


def cmd_resume(args):
    conn, run, db_file, runs_dir = load_run(args)
    if run.get("status") == "cancelled":
        print(f"Run {run['run_id']} was cancelled; nothing to resume")
        conn.close()
        return 0
    if run["executor"] == "local" and run.get("status") in ("complete", "complete_with_failures"):
        from matrix_analyze import analyze_run
        analyze_run(db_file, run["run_id"], claim=True, force=args.force)
        print(f"Run {run['run_id']} is already {run.get('status')}")
        conn.close()
        return 0
    if run["executor"] == "slurm":
        if run.get("status") in ("complete", "complete_with_failures"):
            from matrix_analyze import analyze_run
            result = analyze_run(db_file, run["run_id"], claim=True, force=args.force)
            print(f"Run {run['run_id']} is already {run.get('status')}"
                  + ("" if result is None else "; analysis refreshed"))
            conn.close()
            return 0
        mark_run_started(conn, run["run_id"])
        recover_scheduling(conn, run, force=True)
        reconcile_slurm(conn, run, {}, force_missing=True)
        repair_chain(conn, run, runs_dir)
        finalize_run(conn, run["run_id"])
        if args.watch:
            counts = watch_loop(conn, run, runs_dir, db_file,
                                interval=args.interval, verbose=True)
            print(f"Finished: {counts.get('success', 0)} success")
        else:
            counts = get_counts(conn, run["run_id"])
            print(f"Resumed; status: {counts}")
        conn.close()
        return 0

    active = active_jobs(conn, run["run_id"])
    live = []
    for job in active:
        alive = False
        for key in ("wrapper_pid", "pid"):
            pid = job.get(key)
            if not pid:
                continue
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                pass
        if alive:
            live.append(job)
    if live and not args.force:
        print(f"{len(live)} local job(s) still have a live pid; use --force to "
              f"abandon them", file=sys.stderr)
        conn.close()
        return 2
    for job in active:
        for key in ("wrapper_pid", "pid"):
            if job.get(key):
                try:
                    os.killpg(int(job[key]), signal.SIGTERM)
                except Exception:
                    pass
        reset_to_pending(conn, run["run_id"], job["job_id"],
                         reason="requeued by resume")
    counts = run_local_scheduler(conn, run, runs_dir, db_file,
                                 verbose=args.verbose, quiet=args.quiet)
    print(f"Finished: {counts.get('success', 0)} success")
    conn.close()
    return 0


def cmd_rerun(args):
    conn, source_run, db_file, runs_dir = load_run(args)
    statuses = args.status or ["failed", "timeout"]
    source_jobs = get_jobs(conn, source_run["run_id"], statuses=statuses)
    if not source_jobs:
        print(f"No jobs with status {statuses} in run {source_run['run_id']}",
              file=sys.stderr)
        conn.close()
        return 2
    selected = []
    for job in source_jobs:
        case = {field: job[field] for field in (
            "provider", "device", "model", "dl_mode", "layout", "scorep",
            "api_mode", "api_lang", "steps", "clients", "batch_size",
        )}
        case["job_id"] = job["job_id"]
        case["params_hash"] = job["params_hash"]
        selected.append(case)

    if args.dry_run:
        print(f"Rerun of {source_run['run_id']} would select {len(selected)} job(s):")
        for job in selected:
            print(f"  {job['job_id']:05d} {job['provider']}/{job['device']} "
                  f"{job['model']} {job['api_mode']}/{job['api_lang']} "
                  f"scorep={job['scorep']}")
        conn.close()
        return 0
    base, cpu_cores, gpu_cores = resolve_cores(
        args.cpus_per_job, args.cpus_per_job_cpu, args.cpus_per_job_gpu,
        source=source_run,
    )
    new_run_id = args.new_run_id or make_run_id(prefix="rerun-")
    new_run = dict(source_run)
    new_run.update({
        "run_id": new_run_id,
        "parent_run_id": source_run["run_id"],
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "status": "pending",
        "n_jobs": len(selected),
        "k": args.k if args.k is not None else source_run["k"],
        "timeout_s": args.timeout if args.timeout is not None else source_run["timeout_s"],
        "cpus_per_job": int(base),
        "cpus_per_job_cpu": int(cpu_cores),
        "cpus_per_job_gpu": int(gpu_cores),
        "executor": args.executor or source_run["executor"],
        "partition_cpu": args.partition_cpu or source_run["partition_cpu"],
        "account_cpu": args.account_cpu or source_run["account_cpu"],
        "partition_gpu": args.partition_gpu or source_run["partition_gpu"],
        "account_gpu": args.account_gpu or source_run["account_gpu"],
        "gres_gpu": source_run["gres_gpu"],
        "time_limit": args.time or source_run["time_limit"],
        "sbatch_extra": source_run["sbatch_extra"],
        "filters_json": json.dumps({
            "rerun_of": source_run["run_id"],
            "statuses": statuses,
            "job_ids": [case["job_id"] for case in selected],
        }, sort_keys=True),
        "build_json": json.dumps(build_provenance(), sort_keys=True),
        "scorep_mpp": (args.scorep_mpp or source_run["scorep_mpp"]
                       or os.environ.get("SCOREP_MPP") or "none"),
        "analysis_status": "pending",
        "analysis_started_at": None,
        "analysis_finished_at": None,
    })
    conn.close()

    new_conn, run_dir, new_db = create_run_and_jobs(new_run, selected, runs_dir)
    print(f"Rerun: {new_run_id} (parent {source_run['run_id']}) jobs={len(selected)}")
    args.executor = new_run["executor"]
    args.k = new_run["k"]
    args.cpus_per_job = new_run["cpus_per_job"]
    args.cpus_per_job_cpu = new_run["cpus_per_job_cpu"]
    args.cpus_per_job_gpu = new_run["cpus_per_job_gpu"]
    args.timeout = new_run["timeout_s"]
    args.partition_cpu = new_run["partition_cpu"] or ""
    args.partition_gpu = new_run["partition_gpu"] or ""
    args.scorep_mpp = new_run["scorep_mpp"] or args.scorep_mpp
    # Reuse the source templates (they may have been snapshotted under a
    # different config revision).
    source_templates = runs_dir / source_run["universe_hash"] / source_run["run_id"] / "configs" / "templates"
    target_templates = Path(run_dir) / "configs" / "templates"
    if source_templates.is_dir():
        for path in source_templates.iterdir():
            shutil.copy2(path, target_templates / path.name)
    try:
        preflight(selected, args, run_dir)
    except PreflightError as exc:
        update_run_status(new_conn, new_run_id, "error")
        print(f"Preflight failed: {exc}", file=sys.stderr)
        new_conn.close()
        return 2
    mark_run_started(new_conn, new_run_id)
    if new_run["executor"] == "slurm":
        count = seed_slurm(new_conn, new_run, runs_dir)
        print(f"Seeded {count} SLURM job(s)")
        if args.watch:
            watch_loop(new_conn, new_run, runs_dir, new_db,
                       interval=args.interval, verbose=True)
    else:
        run_local_scheduler(new_conn, new_run, runs_dir, new_db,
                            verbose=args.verbose, quiet=args.quiet)
    new_conn.close()
    return 0


def cmd_cancel(args):
    conn, run, db_file, runs_dir = load_run(args)
    run_id = run["run_id"]
    if run.get("status") in ("complete", "complete_with_failures"):
        print(f"Run {run_id} is already {run.get('status')}; nothing to cancel")
        conn.close()
        return 0
    from matrix_db import cancel_pending
    cancel_pending(conn, run_id)
    active = active_jobs(conn, run_id)
    if run["executor"] == "slurm":
        slurm_ids = [job["slurm_job_id"] for job in active if job.get("slurm_job_id")]
        scancel(slurm_ids)
    else:
        for job in active:
            for key in ("wrapper_pid", "pid"):
                pid = job.get(key)
                if not pid:
                    continue
                try:
                    os.killpg(int(pid), signal.SIGTERM)
                except Exception:
                    pass
    for job in active:
        mark_terminal(conn, run_id, job["job_id"], "cancelled",
                      error_kind="cancelled", error_excerpt="cancelled by user")
    update_run_status(conn, run_id, "cancelled")
    insert_event(conn, run_id, None, "cancelled", {"jobs": len(active)})
    print(f"Cancelled run {run_id} ({len(active)} active job(s))")
    conn.close()
    return 0


def cmd_compact(args):
    conn, run, db_file, runs_dir = load_run(args)
    run_id = run["run_id"]
    if run.get("status") not in ("complete", "complete_with_failures") or not all_terminal(conn, run_id):
        print(f"Cannot compact run {run_id}: run is not complete and terminal", file=sys.stderr)
        conn.close()
        return 2
    run_dir = runs_dir / run["universe_hash"] / run_id
    try:
        files, dirs = compact_run_directory(run_dir, archive_scorep=args.archive_scorep)
    except (FileExistsError, RuntimeError) as exc:
        print(f"[compact] {exc}", file=sys.stderr)
        conn.close()
        return 2
    print(f"Compacted run {run_id}: removed {files} files, {dirs} directories")
    conn.close()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Job-based matrix scheduler with a SQLite ledger.",
        epilog="""Subcommands:
  plan    show the selected cases and their global job IDs (no side effects)
  submit  create a run and schedule it (local process pool or SLURM chain)
  watch   repair/monitor a SLURM run
  resume  rerun pending/active jobs after an interruption
  rerun   rerun failed/timeout jobs in a new run
  cancel  cancel pending/active jobs

Examples:
  python3 matrix_jobs.py plan --providers PHYDLL --devices CPU --all
  python3 matrix_jobs.py submit --job-ids 1729 1735 1741 --executor local -k 2
  python3 matrix_jobs.py submit --providers AIX PHYDLL --devices CPU --models watercnn --api-modes STATIC --api-langs cpp c fortran --workloads 1/1 --batch-sizes 1 --scorep off --executor local -k 3
  python3 matrix_jobs.py submit ... --executor slurm -k 50 --partition-cpu c23ms --account-cpu rwth2150 --partition-gpu c23g --account-gpu rwth0792

Use 'python3 matrix_status.py --list' to list runs and
'python3 matrix_status.py --run-id RUN --watch 5' to monitor one.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser(
        "plan", help="Print the selected universe/IDs",
        epilog="Examples:\n"
               "  matrix_jobs.py plan --providers PHYDLL --devices CPU --api-langs cpp c fortran --all\n"
               "  matrix_jobs.py plan --job-ids 1729 1735 1741 --csv\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_filter_args(plan)
    add_selection_args(plan)
    plan.add_argument("--display-limit", "--limit", dest="display_limit",
                      type=int, default=100,
                      help="Maximum rows to print (default: 100); does not limit the run")
    plan.add_argument("--all", action="store_true",
                      help="Print every selected case")
    plan.add_argument("--csv", action="store_true",
                      help="Print machine-readable CSV")
    plan.set_defaults(func=cmd_plan)

    submit = sub.add_parser(
        "submit", help="Create a run and schedule its jobs",
        epilog="Examples:\n"
               "  matrix_jobs.py submit --providers PHYDLL --devices CPU --models watercnn \\\n"
               "    --api-modes STATIC --api-langs cpp c fortran --workloads 1/1 \\\n"
               "    --batch-sizes 1 --scorep off --executor local -k 2 --skip-build\n"
               "  matrix_jobs.py submit --job-ids 1729 1735 1741 --executor local -k 2\n"
               "  matrix_jobs.py submit ... --executor slurm -k 50 --partition-cpu c23ms \\\n"
               "    --account-cpu rwth2150 --partition-gpu c23g --account-gpu rwth0792\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_filter_args(submit)
    add_selection_args(submit)
    add_resource_args(submit)
    submit.add_argument("--run-id", default="",
                        help="Explicit run ID (default: timestamp + short random suffix)")
    submit.add_argument("--runs-dir", default="runs",
                        help="Directory for ledgers/logs (default: runs)")
    submit.add_argument("--watch", action="store_true",
                        help="Keep watching a SLURM run after seeding")
    submit.add_argument("--interval", type=float, default=60.0,
                        help="Watch poll interval in seconds (default: 60)")
    submit.set_defaults(func=cmd_submit)

    watch = sub.add_parser("watch", help="Repair/monitor a SLURM run")
    watch.add_argument("run_id")
    watch.add_argument("--runs-dir", default="runs")
    watch.add_argument("--universe", default="")
    watch.add_argument("--interval", type=float, default=60.0)
    watch.set_defaults(func=cmd_watch)

    resume = sub.add_parser("resume", help="Repair pending/active jobs")
    resume.add_argument("run_id")
    resume.add_argument("--runs-dir", default="runs")
    resume.add_argument("--universe", default="")
    resume.add_argument("--interval", type=float, default=60.0)
    resume.add_argument("--watch", action="store_true")
    resume.add_argument("--force", action="store_true")
    resume.add_argument("--verbose", action="store_true")
    resume.set_defaults(func=cmd_resume)

    rerun = sub.add_parser("rerun", help="Rerun failures/timeouts in a new run")
    rerun.add_argument("run_id")
    rerun.add_argument("--runs-dir", default="runs")
    rerun.add_argument("--universe", default="")
    rerun.add_argument("--status", nargs="+", default=["failed", "timeout"],
                       choices=list(BAD_STATUSES))
    rerun.add_argument("--new-run-id", default="")
    add_resource_args(rerun, rerun=True)
    rerun.add_argument("--watch", action="store_true")
    rerun.add_argument("--interval", type=float, default=60.0)
    rerun.set_defaults(func=cmd_rerun)

    cancel = sub.add_parser("cancel", help="Cancel pending/active jobs")
    cancel.add_argument("run_id")
    cancel.add_argument("--runs-dir", default="runs")
    cancel.add_argument("--universe", default="")
    cancel.set_defaults(func=cmd_cancel)

    compact = sub.add_parser("compact", help="Reclaim inodes from transient artifacts in a run")
    compact.add_argument("run_id")
    compact.add_argument("--runs-dir", default="runs")
    compact.add_argument("--universe", default="")
    compact.add_argument("--archive-scorep", action="store_true",
                         help="Pack scorep/ directory into scorep.tar")
    compact.set_defaults(func=cmd_compact)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
