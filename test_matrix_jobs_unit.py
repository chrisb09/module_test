#!/usr/bin/env python3
"""Unit tests for the job-based matrix tooling (stdlib unittest only)."""

import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from matrix_common import (
    DEFAULT_CPU_CORES,
    DEFAULT_GPU_CORES,
    API_LANGS_ALL,
    MATRIX_VERSION,
    MODULE_TEST_DIR,
    UNIVERSE_HASH,
    config_template_name,
    cores_for_job,
    iter_cases,
    resolve_cores,
    select_cases,
    update_toml,
)
import matrix_db as db
from matrix_analyze import analyze_run
from matrix_slurm import classify_state
from matrix_status import compute_eta


SMALL_FILTERS = dict(
    providers=["PHYDLL"],
    devices=["CPU"],
    models=["watercnn"],
    api_modes=["STATIC"],
    api_langs=["cpp", "c", "fortran"],
    workloads=[(1, 1)],
    batch_sizes=[1],
    scorep=["off"],
    transport_layouts=["packed"],
    dl_modes=["cpp"],
)


def minimal_run(run_id, n_jobs, k=2, executor="local"):
    return {
        "run_id": run_id,
        "universe_hash": UNIVERSE_HASH,
        "matrix_version": MATRIX_VERSION,
        "created_at": time.time(),
        "started_at": time.time(),
        "n_jobs": n_jobs,
        "k": k,
        "timeout_s": 60.0,
        "cpus_per_job": 1,
        "executor": executor,
        "analysis_status": "pending",
    }


class UniverseTests(unittest.TestCase):
    def test_universe_count_and_ranges(self):
        cases = list(iter_cases())
        self.assertEqual(len(cases), 4320)
        for provider, low, high in (
            ("AIX", 1, 720),
            ("PHYDLL", 721, 3600),
            ("SMARTSIM", 3601, 4320),
        ):
            selected = select_cases(providers=[provider])
            ids = [case["job_id"] for case in selected]
            self.assertEqual(min(ids), low)
            self.assertEqual(max(ids), high)

    def test_hashes_are_unique_and_stable(self):
        cases = select_cases()
        hashes = {case["params_hash"] for case in cases}
        self.assertEqual(len(hashes), len(cases))
        again = select_cases(providers=["AIX"])
        expected = {case["job_id"] for case in cases if case["provider"] == "AIX"}
        self.assertEqual({case["job_id"] for case in again}, expected)
        self.assertEqual(UNIVERSE_HASH, "e921ee89cc55")

    def test_language_restriction(self):
        cases = select_cases(api_modes=["ORDERED"])
        self.assertTrue(cases)
        self.assertEqual({case["api_lang"] for case in cases}, {"cpp"})


class CoreTests(unittest.TestCase):
    def test_default_gpu_cores_match_96c4g(self):
        self.assertEqual(DEFAULT_CPU_CORES, 4)
        self.assertEqual(DEFAULT_GPU_CORES, 24)
        self.assertEqual(resolve_cores(), (4, 4, 24))

    def test_explicit_core_resolution(self):
        self.assertEqual(resolve_cores(cpus=8), (8, 8, 8))
        self.assertEqual(resolve_cores(cpu=24), (4, 24, 24))
        self.assertEqual(resolve_cores(cpu=24, gpu=12), (4, 24, 12))
        source = {"cpus_per_job": 4, "cpus_per_job_cpu": 6, "cpus_per_job_gpu": 24}
        self.assertEqual(resolve_cores(source=source), (4, 6, 24))
        self.assertEqual(resolve_cores(cpu=8, source=source), (4, 8, 24))
        self.assertEqual(resolve_cores(cpus=8, source=source), (8, 8, 8))

    def test_per_device_cores(self):
        run = {"cpus_per_job": 4, "cpus_per_job_cpu": 2, "cpus_per_job_gpu": 8}
        self.assertEqual(cores_for_job(run, "CPU"), 2)
        self.assertEqual(cores_for_job(run, "GPU"), 8)
        self.assertEqual(cores_for_job({"cpus_per_job": 6}, "CPU"), 6)
        self.assertEqual(cores_for_job({}, "GPU"), 1)


class ConfigTests(unittest.TestCase):
    def test_update_toml_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = MODULE_TEST_DIR / config_template_name("PHYDLL", "CPU", "STATIC")
            dst = Path(tmp) / "job.toml"
            shutil.copy2(src, dst)
            update_toml(dst, "PHYDLL", "CPU", "watercnn", transport_layout="uniform_chunks")
            text = dst.read_text()
            self.assertIn("watercnn_cpu.pt", text)
            self.assertIn('transport_layout = "uniform_chunks"', text)

    def test_config_template_names(self):
        self.assertEqual(config_template_name("AIX", "GPU", "ORDERED"),
                         "config_aix_gpu_flex.toml")
        self.assertEqual(config_template_name("SMARTSIM", "CPU", "STATIC"),
                         "config_smartsim_cpu.toml")


class EtaTests(unittest.TestCase):
    def test_eta_throughput_and_stratified(self):
        now = time.time()
        run = {"started_at": now - 100, "created_at": now - 100,
               "status": "running", "k": 2, "finished_at": None}
        base = {
            "provider": "PHYDLL", "device": "CPU", "scorep": "off",
            "layout": "packed", "api_mode": "STATIC",
        }
        jobs = []
        for index in range(6):
            job = dict(base)
            job["status"] = "success" if index < 2 else "pending"
            job["duration_s"] = 10.0 if index < 2 else None
            jobs.append(job)
        eta = compute_eta(run, jobs)
        self.assertEqual(eta["done"], 2)
        self.assertAlmostEqual(eta["eta_stratified"], 20.0, places=3)
        self.assertIsNotNone(eta["eta_throughput"])

    def test_slurm_completed_maps_to_success(self):
        self.assertEqual(classify_state("COMPLETED"), ("success", None))
        self.assertEqual(classify_state("TIMEOUT")[0], "timeout")
        self.assertEqual(classify_state("CANCELLED")[0], "cancelled")


class LedgerTests(unittest.TestCase):
    def test_transitions_and_analysis_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = db.db_path(tmp, UNIVERSE_HASH)
            conn = db.init_db(path, UNIVERSE_HASH, MATRIX_VERSION)
            cases = select_cases(**SMALL_FILTERS)[:2]
            run = minimal_run("run-1", len(cases))
            db.create_run(conn, run)
            db.insert_jobs(conn, run["run_id"], cases)
            counts = db.get_counts(conn, run["run_id"])
            self.assertEqual(counts["pending"], 2)
            self.assertFalse(db.all_terminal(conn, run["run_id"]))

            job = db.get_jobs(conn, run["run_id"])[0]
            self.assertTrue(db.mark_running(conn, run["run_id"], job["job_id"]))
            self.assertFalse(db.mark_running(conn, run["run_id"], job["job_id"]))
            self.assertTrue(db.mark_terminal(conn, run["run_id"], job["job_id"],
                                             "success", duration_s=1.5,
                                             exit_code=0, result_json='{"first": [[1]]}',
                                             result_valid=True))
            self.assertFalse(db.all_terminal(conn, run["run_id"]))
            self.assertFalse(db.claim_analysis(conn, run["run_id"]))

            second = db.get_jobs(conn, run["run_id"])[1]
            db.mark_terminal(conn, run["run_id"], second["job_id"], "failed",
                             exit_code=1, error_excerpt="boom")
            self.assertTrue(db.all_terminal(conn, run["run_id"]))
            self.assertTrue(db.claim_analysis(conn, run["run_id"]))
            self.assertFalse(db.claim_analysis(conn, run["run_id"]))
            conn.close()


class AnalysisTests(unittest.TestCase):
    def test_anomaly_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = db.db_path(tmp, UNIVERSE_HASH)
            conn = db.init_db(path, UNIVERSE_HASH, MATRIX_VERSION)
            cases = select_cases(**SMALL_FILTERS)[:3]
            run = minimal_run("run-2", len(cases))
            db.create_run(conn, run)
            db.insert_jobs(conn, run["run_id"], cases)
            results = ['{"first": [[0.035]], "sum": [[0.035]]}',
                       '{"first": [[0.035]], "sum": [[0.035]]}',
                       '{"first": [[9.999]], "sum": [[9.999]]}']
            for case, result in zip(cases, results):
                db.mark_terminal(conn, run["run_id"], case["job_id"], "success",
                                 duration_s=2.0, exit_code=0,
                                 result_json=result, result_valid=True)
            conn.close()

            summary = analyze_run(path, "run-2", claim=True)
            self.assertIsNotNone(summary)
            self.assertEqual(summary["anomaly_count"], 1)
            self.assertEqual(summary["groups"][0]["anomaly_job_ids"],
                             [cases[2]["job_id"]])
            report = Path(tmp) / UNIVERSE_HASH / "run-2" / "analysis.md"
            self.assertTrue(report.is_file())
            self.assertIn("Anomalies", report.read_text())

            conn = db.connect(path)
            stored = db.get_analysis(conn, "run-2")
            self.assertEqual(stored["status"], "success")
            self.assertEqual(stored["anomaly_count"], 1)
            conn.close()


if __name__ == "__main__":
    unittest.main()
