#!/usr/bin/env python3
"""Reanalyze historical successful jobs without writing to their ledger."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from matrix_analyze import _group_results, _modal_result, write_report
from matrix_common import compare_results, parse_summary
from matrix_db import connect, get_jobs, get_run


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return digest(path.read_bytes()) if path.exists() else None


def final_attempt(text, run_id, job_id, seq, universe):
    headers = list(re.finditer(r"^# matrix job (\d+) seq=(\d+) run=(\S+) universe=(\S+)\s*$", text, re.MULTILINE))
    if not headers:
        raise ValueError("no matrix job attempt header")
    last = headers[-1]
    if (int(last[1]), int(last[2]), last[3], last[4]) != (job_id, seq, run_id, universe):
        raise ValueError("final attempt header does not match ledger job")
    return text[last.end():], len(headers)


def reanalyze(db_file, run_id, out_dir):
    db_file = Path(db_file).resolve(strict=True)
    out_dir = Path(out_dir).absolute()
    # Exclusive creation ensures no existing report, ledger, or directory is replaced.
    out_dir.mkdir(parents=False, exist_ok=False)
    original_report = db_file.parent / run_id / "analysis.md"
    sources = [db_file, Path(str(db_file) + "-wal"), original_report]
    before = {str(path): file_hash(path) for path in sources}
    conn = connect(db_file, readonly=True)
    try:
        conn.execute("BEGIN")
        run = get_run(conn, run_id)
        if run is None:
            raise ValueError(f"Unknown run: {run_id}")
        jobs = get_jobs(conn, run_id)
    finally:
        conn.close()

    records, valid = [], []
    for job in jobs:
        if job["status"] != "success":
            continue
        record = {"job_id": job["job_id"], "seq": job["seq"],
                  "ledger_attempts": job["attempts"], "log_path": job["log_path"],
                  "log_sha256": None, "error": None}
        try:
            if not job["log_path"]:
                raise ValueError("missing log path")
            path = Path(job["log_path"])
            if not path.is_absolute():
                path = Path(__file__).resolve().parent / path
            data = path.read_bytes()
            record["log_sha256"] = digest(data)
            text = data.decode("utf-8")
            attempt, record["headers_seen"] = final_attempt(
                text, run_id, job["job_id"], job["seq"], run["universe_hash"])
            result = parse_summary(attempt)
            if result == "N/A":
                raise ValueError("updated parser found no valid matching first and sum matrices")
            record["result"] = json.loads(result)
            valid.append(dict(job, result_json=result))
        except (OSError, UnicodeError, ValueError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        records.append(record)

    groups = {}
    for key, entries in _group_results(valid).items():
        mode, count = _modal_result(entries)
        groups[key] = {"entries": entries, "mode_result": mode, "mode_count": count,
                       "anomalies": [entry for entry in entries
                                     if not compare_results(entry["result_json"], mode)]}
    after = {str(path): file_hash(path) for path in sources}
    if before != after:
        raise RuntimeError("source ledger, WAL, or original report changed during reanalysis")
    root = Path(__file__).resolve().parent
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=True).stdout.strip()
    code = [root / name for name in ("matrix_reanalyze.py", "matrix_common.py",
                                      "matrix_analyze.py", "matrix_db.py")]
    errors = [record for record in records if record["error"]]
    summary = {
        "run_id": run_id, "universe_hash": run["universe_hash"],
        "source_hashes_sha256_before_and_after": before,
        "git_revision": revision,
        "code_hashes_sha256": {str(path): file_hash(path) for path in code},
        "counts": {"jobs": len(jobs), "successful_jobs": len(records),
                   "parsed": len(valid), "errors": len(errors), "groups": len(groups),
                   "anomalies": sum(len(group["anomalies"]) for group in groups.values())},
        "jobs": records,
    }
    report = out_dir / "analysis.md"
    write_report(run, groups, report)
    with report.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## Reanalysis coverage\n\nSuccessful jobs: {len(records)}; "
                     f"parsed: {len(valid)}; errors: {len(errors)}. "
                     "Only parsed final attempts are compared. See `provenance.json` for all log hashes and results.\n")
        if errors:
            handle.write("\n| Job | Log | Error |\n|-----|-----|-------|\n")
            for record in errors:
                error = record["error"].replace("|", "\\|").replace("\n", " ")
                log = str(record["log_path"]).replace("|", "\\|").replace("\n", " ")
                handle.write(f"| {record['job_id']:05d} | `{log}` | {error} |\n")
    (out_dir / "provenance.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="Source ledger (opened read-only)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True, help="New directory; parent must exist")
    args = parser.parse_args(argv)
    try:
        summary = reanalyze(args.db, args.run_id, args.out_dir)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Reanalysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
