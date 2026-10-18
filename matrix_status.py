#!/usr/bin/env python3
"""Progress viewer for the job-based matrix ledger."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from matrix_common import format_duration, median
from matrix_db import (
    BAD_STATUSES,
    TERMINAL_STATUSES,
    connect,
    get_analysis,
    get_counts,
    get_job,
    get_jobs,
    get_run,
    list_universe_dirs,
    list_runs,
)


FAILURE_STATUSES = ("failed", "timeout", "error", "lost")
FEATURE_DIMENSIONS = [
    "provider", "device", "api_lang", "api_mode", "scorep",
    "layout", "dl_mode", "model", "steps", "batch_size",
]


def ts(value):
    if not value:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(value))


def bar(fraction, width=30):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(width * fraction))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def find_latest(runs_dir):
    candidates = []
    for universe_dir in list_universe_dirs(runs_dir):
        db_file = universe_dir / "ledger.sqlite3"
        conn = connect(db_file, readonly=True)
        try:
            for run in list_runs(conn):
                candidates.append((run.get("created_at") or 0, db_file, run["run_id"]))
        except Exception:
            pass
        finally:
            conn.close()
    if not candidates:
        return None, None
    _, db_file, run_id = max(candidates)
    return db_file, run_id


def find_run(runs_dir, run_id):
    for universe_dir in list_universe_dirs(runs_dir):
        db_file = universe_dir / "ledger.sqlite3"
        conn = connect(db_file, readonly=True)
        try:
            if get_run(conn, run_id):
                return db_file
        except Exception:
            pass
        finally:
            conn.close()
    return None


def cmd_list(runs_dir):
    rows = []
    for universe_dir in list_universe_dirs(runs_dir):
        db_file = universe_dir / "ledger.sqlite3"
        conn = connect(db_file, readonly=True)
        try:
            for run in list_runs(conn):
                counts = get_counts(conn, run["run_id"])
                rows.append((run, counts))
        except Exception:
            pass
        finally:
            conn.close()
    rows.sort(key=lambda item: item[0].get("created_at") or 0, reverse=True)
    print(f"{'Run':<28} {'Universe':<12} {'Status':<24} {'Total':>6} "
          f"{'OK':>6} {'Bad':>6} {'Created':<20} Executor")
    for run, counts in rows:
        bad = sum(counts.get(status, 0) for status in BAD_STATUSES)
        print(f"{run['run_id']:<28} {run['universe_hash']:<12} "
              f"{str(run.get('status')):<24} {counts.get('total', 0):>6} "
              f"{counts.get('success', 0):>6} {bad:>6} "
              f"{ts(run.get('created_at')):<20} {run.get('executor')}")
    return 0


def compute_eta(run, jobs):
    total = len(jobs)
    done = sum(1 for job in jobs if job["status"] in TERMINAL_STATUSES)
    pending = [job for job in jobs if job["status"] in ("pending", "scheduling")]
    running = [job for job in jobs if job["status"] == "scheduled" or job["status"] == "running"]
    started = run.get("started_at") or run.get("created_at") or time.time()
    if run.get("status") in ("complete", "complete_with_failures", "cancelled") and run.get("finished_at"):
        elapsed = max(0.0, run["finished_at"] - started)
    else:
        elapsed = max(0.0, time.time() - started)
    k = max(1, int(run.get("k") or 1))

    eta_throughput = None
    if done > 0 and elapsed > 0:
        rate = done / elapsed
        if rate > 0:
            eta_throughput = (total - done) / rate

    by_stratum = defaultdict(list)
    for job in jobs:
        if job["status"] in TERMINAL_STATUSES and job.get("duration_s"):
            key = (job["provider"], job["device"], job["scorep"],
                   job["layout"], job["api_mode"])
            by_stratum[key].append(float(job["duration_s"]))
    all_durations = [d for values in by_stratum.values() for d in values]
    global_median = median(all_durations) if all_durations else 0.0
    expected = 0.0
    for job in pending:
        key = (job["provider"], job["device"], job["scorep"],
               job["layout"], job["api_mode"])
        expected += median(by_stratum[key]) if by_stratum.get(key) else global_median
    for job in running:
        key = (job["provider"], job["device"], job["scorep"],
               job["layout"], job["api_mode"])
        expected += 0.5 * (median(by_stratum[key]) if by_stratum.get(key) else global_median)
    eta_stratified = expected / k if expected else None

    return {
        "total": total,
        "done": done,
        "pending": len(pending),
        "running": len(running),
        "elapsed": elapsed,
        "eta_throughput": eta_throughput,
        "eta_stratified": eta_stratified,
        "median_duration": global_median,
    }


def feature_analysis(jobs):
    bad_jobs = [job for job in jobs if job["status"] in FAILURE_STATUSES]
    total = len(jobs)
    bad_total = len(bad_jobs)
    overall_rate = (bad_total / total) if total else 0.0
    features = {}
    for dimension in FEATURE_DIMENSIONS:
        buckets = defaultdict(lambda: {"total": 0, "bad": 0, "failed": 0,
                                       "timeout": 0, "error": 0, "lost": 0})
        for job in jobs:
            value = job.get(dimension)
            key = str(value)
            buckets[key]["total"] += 1
            if job["status"] in FAILURE_STATUSES:
                buckets[key]["bad"] += 1
                buckets[key][job["status"]] = buckets[key].get(job["status"], 0) + 1
        entries = []
        for value, counts in buckets.items():
            rate = counts["bad"] / counts["total"] if counts["total"] else 0.0
            lift = (rate / overall_rate) if overall_rate > 0 else 0.0
            entries.append({
                "value": value,
                "total": counts["total"],
                "bad": counts["bad"],
                "rate": rate,
                "lift": lift,
            })
        entries.sort(key=lambda item: (-item["bad"], item["value"]))
        features[dimension] = entries
    common = {}
    for dimension in FEATURE_DIMENSIONS:
        values = {str(job.get(dimension)) for job in bad_jobs}
        if len(values) == 1 and bad_total:
            value = next(iter(values))
            if value != "-":
                common[dimension] = value
    return {
        "total": total,
        "bad": bad_total,
        "rate": overall_rate,
        "features": features,
        "common": common,
        "bad_jobs": bad_jobs,
    }


def print_summary(run, counts, jobs, analysis):
    total = counts.get("total", 0)
    done = total - counts.get("pending", 0) - counts.get("scheduled", 0) \
        - counts.get("scheduling", 0) - counts.get("running", 0)
    fraction = (done / total) if total else 0.0
    eta = compute_eta(run, jobs)

    print(f"Run      {run['run_id']}")
    print(f"Universe {run['universe_hash']} (matrix v{run.get('matrix_version')})")
    cpu_cores = run.get("cpus_per_job_cpu")
    if cpu_cores is None:
        cpu_cores = run.get("cpus_per_job")
    gpu_cores = run.get("cpus_per_job_gpu")
    if gpu_cores is None:
        gpu_cores = run.get("cpus_per_job")
    print(f"Status   {run.get('status')}   executor={run.get('executor')} "
          f"k={run.get('k')} cores=cpu:{cpu_cores}/gpu:{gpu_cores} "
          f"(base:{run.get('cpus_per_job')}) timeout={run.get('timeout_s')}s")
    print(f"Created  {ts(run.get('created_at'))}   Started {ts(run.get('started_at'))}   "
          f"Finished {ts(run.get('finished_at'))}")
    print()
    print(f"{bar(fraction)} {done}/{total} ({fraction * 100:.1f}%)  "
          f"elapsed {format_duration(eta['elapsed'])}")
    if eta["eta_throughput"] is not None:
        print(f"ETA      throughput {format_duration(eta['eta_throughput'])}   "
              f"stratified {format_duration(eta['eta_stratified'])}")
    print()
    print("Status counts")
    for status in ("success", "failed", "timeout", "error", "cancelled", "lost",
                   "running", "scheduled", "pending"):
        count = counts.get(status, 0)
        if count:
            print(f"  {status:<12} {count:>6}")
    print()

    features = feature_analysis(jobs)
    if features["bad"]:
        print(f"Failures/timeouts: {features['bad']}/{features['total']} "
              f"({features['rate'] * 100:.1f}%)")
        if features["common"]:
            print("  common to all unsuccessful jobs: " + ", ".join(
                f"{key}={value}" for key, value in features["common"].items()))
        for dimension in FEATURE_DIMENSIONS:
            entries = [entry for entry in features["features"][dimension]
                       if entry["bad"]]
            if not entries or all(entry["value"] == "-" for entry in entries):
                continue
            print(f"  {dimension}:")
            for entry in entries[:5]:
                print(f"    {entry['value']:<18} {entry['bad']:>5}/{entry['total']:<5} "
                      f"rate={entry['rate'] * 100:5.1f}%  lift={entry['lift']:.2f}x")
        print()
        print("  recent failures/timeouts:")
        for job in sorted(features["bad_jobs"], key=lambda item: item["seq"])[:10]:
            print(f"    job {job['job_id']:05d} {job['status']:<8} "
                  f"{job['provider']}/{job['device']}/{job['api_mode']}/{job['api_lang']} "
                  f"{job.get('error_kind') or '-'}  log={job.get('log_path') or '-'}")
        print()

    if analysis:
        summary = analysis.get("summary_json")
        try:
            summary = json.loads(summary) if summary else {}
        except Exception:
            summary = {}
        print(f"Analysis {analysis.get('status')}  groups={analysis.get('groups_total')} "
              f"anomalies={analysis.get('anomaly_count')}  "
              f"report={analysis.get('report_path')}")
        for group in summary.get("groups", []):
            if group.get("anomaly_job_ids"):
                print(f"  {group['model']} {group['steps']}/{group['clients']}/"
                      f"{group['batch_size']}: {len(group['anomaly_job_ids'])} "
                      f"anomalies {group['anomaly_job_ids']}")
        print()


def job_details(job):
    print(f"Job {job['job_id']:05d} seq={job['seq']} status={job['status']}")
    print(f"  {job['provider']}/{job['device']} model={job['model']} "
          f"effective={job['effective_model']}")
    print(f"  api={job['api_mode']}/{job['api_lang']} dl={job['dl_mode']} "
          f"layout={job['layout']} scorep={job['scorep']}")
    print(f"  steps={job['steps']} clients={job['clients']} batch={job['batch_size']}")
    print(f"  scheduled={ts(job.get('scheduled_at'))} started={ts(job.get('started_at'))} "
          f"finished={ts(job.get('finished_at'))} duration={job.get('duration_s')}")
    print(f"  exit={job.get('exit_code')} attempts={job.get('attempts')} "
          f"slurm={job.get('slurm_job_id')} state={job.get('slurm_state')} "
          f"node={job.get('node')}")
    print(f"  log={job.get('log_path')}")
    print(f"  result={job.get('result_json')}")
    print(f"  error={job.get('error_kind')}")
    if job.get("error_excerpt"):
        print("  error excerpt:")
        for line in str(job["error_excerpt"]).splitlines()[-15:]:
            print(f"    {line}")


def load_target(args):
    runs_dir = Path(args.runs_dir).resolve()
    if args.run_id:
        db_file = find_run(runs_dir, args.run_id)
        if not db_file:
            raise SystemExit(f"Unknown run {args.run_id}")
        return db_file, args.run_id
    if args.latest:
        db_file, run_id = find_latest(runs_dir)
        if not db_file:
            raise SystemExit("No runs found")
        return db_file, run_id
    raise SystemExit("Specify a run id, --latest, or --list")


def render(args):
    if args.list:
        return cmd_list(Path(args.runs_dir).resolve())
    db_file, run_id = load_target(args)
    conn = connect(db_file, readonly=True)
    try:
        run = get_run(conn, run_id)
        if not run:
            raise SystemExit(f"Unknown run {run_id}")
        if args.job is not None:
            job = get_job(conn, run_id, args.job)
            if not job:
                raise SystemExit(f"Unknown job {args.job}")
            job_details(job)
            return 0
        jobs = get_jobs(conn, run_id)
        counts = get_counts(conn, run_id)
        analysis = get_analysis(conn, run_id)
        if args.json:
            features = feature_analysis(jobs)
            payload = {
                "run": run,
                "counts": counts,
                "eta": compute_eta(run, jobs),
                "features": features,
                "analysis": analysis,
            }
            print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        else:
            print_summary(run, counts, jobs, analysis)
        return 0
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="View job-based matrix progress.")
    parser.add_argument("run_id", nargs="?", default="")
    parser.add_argument("--run-id", dest="run_id_opt", default="")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--job", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--watch", type=float, default=0.0,
                        help="Refresh interval in seconds")
    args = parser.parse_args(argv)
    args.run_id = args.run_id or args.run_id_opt

    if args.watch > 0:
        try:
            while True:
                print("\033[2J\033[H", end="")
                render(args)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            return 0
    return render(args)


if __name__ == "__main__":
    sys.exit(main())
