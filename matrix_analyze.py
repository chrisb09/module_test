#!/usr/bin/env python3
"""Compare successful results within a run.

The grouping and fuzzy comparison are identical to the sequential matrix:
successful jobs are grouped by ``(base_model, steps, clients, batch_size)``;
within each group the modal result is found with a relative/absolute tolerance,
and any result that differs from the mode is reported as an anomaly.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from matrix_common import base_model, compare_results, format_duration
from matrix_db import (
    connect,
    db_path,
    get_run,
    get_jobs,
    claim_analysis,
    finish_analysis,
    list_universe_dirs,
)


def find_db_for_run(runs_dir, run_id):
    for universe_dir in list_universe_dirs(runs_dir):
        candidate = universe_dir / "ledger.sqlite3"
        conn = connect(candidate, readonly=True)
        try:
            row = conn.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        except Exception:
            row = None
        finally:
            conn.close()
        if row:
            return candidate
    return None


def _group_results(jobs):
    groups = {}
    for job in jobs:
        model = base_model(job["effective_model"] or job["model"])
        key = (model, job["steps"], job["clients"], job["batch_size"])
        groups.setdefault(key, []).append(job)
    return groups


def _modal_result(entries):
    result_groups = []  # [summary, count]
    for entry in entries:
        summary = entry["result_json"]
        found = False
        for index, (base_summary, count) in enumerate(result_groups):
            if compare_results(summary, base_summary):
                result_groups[index] = (base_summary, count + 1)
                found = True
                break
        if not found:
            result_groups.append((summary, 1))
    if not result_groups:
        return None, 0
    mode_result, mode_count = max(result_groups, key=lambda item: item[1])
    return mode_result, mode_count


def _short_result(value, limit=160):
    if value is None:
        return "-"
    text = str(value)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def write_report(run, groups, out_path):
    lines = []
    lines.append("# Matrix consistency analysis")
    lines.append("")
    lines.append(f"- Run: `{run['run_id']}`")
    lines.append(f"- Universe: `{run['universe_hash']}` (matrix v{run.get('matrix_version')})")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Executor: {run.get('executor')}  k={run.get('k')}")
    lines.append("")
    lines.append("## Groups")
    lines.append("")
    lines.append("| Model | Steps | Clients | Batch | Runs | Mode count | Anomalies |")
    lines.append("|-------|-------|---------|-------|------|------------|-----------|")
    for key in sorted(groups, key=lambda item: (item[0], item[1], item[2], item[3])):
        model, steps, clients, batch = key
        group = groups[key]
        lines.append(
            f"| {model} | {steps} | {clients} | {batch} | {len(group['entries'])} | "
            f"{group['mode_count']} | {len(group['anomalies'])} |"
        )
    lines.append("")

    anomaly_total = sum(len(group["anomalies"]) for group in groups.values())
    if anomaly_total:
        lines.append("## Anomalies")
        lines.append("")
        lines.append("| Job | Model | Provider | DL | Layout | ScoreP | API | Lang | Device | Result | Log |")
        lines.append("|-----|-------|----------|----|--------|--------|-----|------|--------|--------|-----|")
        for key in sorted(groups, key=lambda item: (item[0], item[1], item[2], item[3])):
            for job in groups[key]["anomalies"]:
                lines.append(
                    f"| {job['job_id']:05d} | {key[0]} | {job['provider']} | "
                    f"{job['dl_mode']} | {job['layout']} | {job['scorep']} | "
                    f"{job['api_mode']} | {job['api_lang']} | {job['device']} | "
                    f"`{_short_result(job['result_json'])}` | `{job.get('log_path') or '-'}` |"
                )
        lines.append("")
    else:
        lines.append("No anomalies found.")
        lines.append("")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return anomaly_total


def analyze_run(db_file, run_id, claim=True, force=False, out_path=None):
    conn = connect(db_file)
    try:
        run = get_run(conn, run_id)
        if not run:
            raise SystemExit(f"Unknown run: {run_id}")

        run_dir = Path(db_file).parent / run_id
        out_path = Path(out_path) if out_path else run_dir / "analysis.md"

        if force:
            conn.execute(
                "UPDATE runs SET analysis_status='running', analysis_started_at=? WHERE run_id=?",
                (time.time(), run_id),
            )
            conn.commit()
        elif claim:
            if not claim_analysis(conn, run_id):
                return None
        elif run.get("analysis_status") not in (None, "pending"):
            return None

        jobs = [
            job for job in get_jobs(conn, run_id, statuses=("success",))
            if job.get("result_valid")
        ]
        raw_groups = _group_results(jobs)
        groups = {}
        for key, entries in raw_groups.items():
            mode_result, mode_count = _modal_result(entries)
            anomalies = [
                entry for entry in entries
                if not compare_results(entry["result_json"], mode_result)
            ]
            groups[key] = {
                "entries": entries,
                "mode_result": mode_result,
                "mode_count": mode_count,
                "anomalies": anomalies,
            }

        anomaly_total = sum(len(group["anomalies"]) for group in groups.values())
        write_report(run, groups, out_path)

        summary = {
            "run_id": run_id,
            "universe_hash": run["universe_hash"],
            "successful": len(jobs),
            "groups": [
                {
                    "model": key[0],
                    "steps": key[1],
                    "clients": key[2],
                    "batch_size": key[3],
                    "runs": len(group["entries"]),
                    "mode_count": group["mode_count"],
                    "mode_result": group["mode_result"],
                    "anomaly_job_ids": [job["job_id"] for job in group["anomalies"]],
                }
                for key, group in sorted(groups.items())
            ],
            "anomaly_count": anomaly_total,
        }
        status = "success" if jobs else "empty"
        finish_analysis(conn, run_id, status, len(groups), anomaly_total,
                        json.dumps(summary, sort_keys=True), out_path)
        return summary
    except Exception as exc:
        try:
            report = Path(out_path) if out_path else Path(db_file).parent / run_id / "analysis.md"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(
                f"# Matrix consistency analysis failed\n\n`{exc}`\n",
                encoding="utf-8",
            )
            finish_analysis(conn, run_id, "error", 0, 0,
                            json.dumps({"error": str(exc)}), report)
        except Exception:
            pass
        raise
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze consistency of a matrix run.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--universe", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir).resolve()
    if args.universe:
        db_file = db_path(runs_dir, args.universe)
    else:
        db_file = find_db_for_run(runs_dir, args.run_id)
        if not db_file:
            print(f"Could not find a ledger containing run {args.run_id}", file=sys.stderr)
            return 2

    summary = analyze_run(
        db_file, args.run_id,
        claim=not args.force, force=args.force,
        out_path=args.out or None,
    )
    if summary is None:
        print(f"Analysis already claimed/completed for run {args.run_id}")
        return 0
    report = Path(args.out) if args.out else Path(db_file).parent / args.run_id / "analysis.md"
    print(f"Analysis: {len(summary['groups'])} group(s), "
          f"{summary['anomaly_count']} anomaly/anomalies, report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
