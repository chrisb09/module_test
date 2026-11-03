#!/usr/bin/env python3
"""Unit tests for the job-based matrix tooling (stdlib unittest only)."""

import json
import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

from matrix_common import (
    DEFAULT_CPU_CORES,
    DEFAULT_GPU_CORES,
    API_LANGS_ALL,
    API_MODES,
    MATRIX_VERSION,
    MODULE_TEST_DIR,
    UNIVERSE_HASH,
    config_template_name,
    cores_for_job,
    iter_cases,
    parse_summary,
    prepare_phydll_dl_client,
    resolve_cores,
    select_cases,
    update_toml,
)
import matrix_db as db
import matrix_job
import matrix_jobs
from matrix_analyze import analyze_run
from matrix_jobs import apply_selection, create_run_and_jobs
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


class ClientBuildTests(unittest.TestCase):
    @patch("matrix_common.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_scorep_client_configuration(self, run):
        path = prepare_phydll_dl_client("on", "/tmp/cmi", "/tmp/module_test")
        self.assertEqual(path, Path("/tmp/cmi/dl_clients/build-module-test-scorep-none"))
        configure = run.call_args_list[0].args[0][-1]
        self.assertIn("unset CC CXX CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER", configure)
        self.assertIn("-DWITH_SCOREP=ON", configure)
        self.assertIn("-DCPPML_SCOREP_MPP=none", configure)
        self.assertIn("SCOREP_CONFIG_EXECUTABLE", configure)
        self.assertIn("--target phydll_dl_client", run.call_args_list[1].args[0][-1])

    @patch("matrix_common.subprocess.run", return_value=SimpleNamespace(returncode=0))
    def test_off_client_configuration_unchanged(self, run):
        path = prepare_phydll_dl_client("off", "/tmp/cmi", "/tmp/module_test")
        self.assertEqual(path, Path("/tmp/cmi/dl_clients/build-module-test"))
        self.assertIn("-DWITH_SCOREP=OFF", run.call_args_list[0].args[0][-1])

    def test_job_client_environment(self):
        job = dict(job_id=1, provider="PHYDLL", device="CPU", effective_model="watercnn",
                   api_mode="STATIC", api_lang="cpp", dl_mode="cpp", layout="packed",
                   steps=1, clients=1, batch_size=1)
        run = minimal_run("run", 1)
        with mock.patch.dict(matrix_job.os.environ, {"PHYDLL_DL_CLIENT": "/custom/client"}):
            for mode, directory in (("on", "build-module-test-scorep-none"),
                                    ("off", "build-module-test")):
                env = matrix_job.build_env(run, {**job, "scorep": mode},
                                           Path("config"), Path("state"), Path("logs/job.log"))
                self.assertEqual(env["PHYDLL_DL_BUILD_DIR"],
                                 str(matrix_job.CMI_DIR / "dl_clients" / directory))
                self.assertEqual(env["PHYDLL_DL_CLIENT"], "/custom/client")
                self.assertEqual(env["PHYDLL_REBUILD_DL_CLIENT"], "0")

    @patch("matrix_common.subprocess.run", return_value=SimpleNamespace(returncode=1))
    def test_failed_scorep_configuration_does_not_build(self, run):
        with self.assertRaisesRegex(RuntimeError, "CMake cache uses Score-P wrappers"):
            prepare_phydll_dl_client("on", "/tmp/cmi", "/tmp/module_test")
        self.assertEqual(run.call_count, 1)


class SummaryTests(unittest.TestCase):
    def test_interleaved_rank_output(self):
        output = """Gathered output[0] from all ranks:
###########################################################################
All steps completed. Final outputs of rank 1 (out_size=7168):
  step 0: first=0.123537 sum=724.029358
  step 1: first=0.123768 sum=724.181702
  step 2: first=0.124200 sum=724.459167
  Last step output buffer: [0.1242 0.1195 ... 0.0812 ]
[PHYDLL:DL] client exiting
[[0.0003, 0.1332], [0.1235, 0.1238]]
Gathered output sums from all ranks:
[PHYDLL:DL] other output
[[0.9055, 896.3141], [724.0294, 724.1817]]
"""
        self.assertEqual(json.loads(parse_summary(output)), {
            "first": [[0.0003, 0.1332], [0.1235, 0.1238]],
            "sum": [[0.9055, 896.3141], [724.0294, 724.1817]],
        })

    def test_missing_sections_do_not_borrow_next_matrix(self):
        for output in (
            "",
            "Gathered output[0] from all ranks:\n[[1]]\n",
            "Gathered output sums from all ranks:\n[[2]]\n",
            "Gathered output[0] from all ranks:\nGathered output sums from all ranks:\n[[2]]\n",
            "Gathered output[0] from all ranks:\n[[1]]\nGathered output sums from all ranks:\nGathered outputs from all ranks:\n[[2]]\n",
            "Gathered outputs from all ranks:\n[[1]]\n",
        ):
            with self.subTest(output=output):
                self.assertEqual(parse_summary(output), "N/A")

    def test_malformed_matrices_are_invalid(self):
        for matrix in ('[[bad]]', '[[1],]', '[["text"]]', '[[true]]',
                       '[[NaN]]', '[[]]', '[[1], [2, 3]]', '[[1, 2]]'):
            with self.subTest(matrix=matrix):
                output = ("Gathered output[0] from all ranks:\n" + matrix +
                          "\n[[1]]\nGathered output sums from all ranks:\n[[2]]\n")
                self.assertEqual(parse_summary(output), "N/A")


class UniverseTests(unittest.TestCase):
    def test_universe_count_and_ranges(self):
        cases = list(iter_cases())
        self.assertEqual(len(cases), 9504)
        for provider, low, high in (
            ("AIX", 1, 1584),
            ("PHYDLL", 1585, 7920),
            ("SMARTSIM", 7921, 9504),
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
        self.assertEqual(MATRIX_VERSION, 2)
        self.assertEqual(UNIVERSE_HASH, "7db159fd5844")

    def test_languages_for_every_mode_and_filtered_ids(self):
        cases = list(iter_cases())
        for mode in API_MODES:
            self.assertEqual({case["api_lang"] for case in cases
                              if case["api_mode"] == mode}, set(API_LANGS_ALL))
        filters = {**SMALL_FILTERS, "api_modes": ["ORDERED"],
                   "api_langs": ["c", "fortran"]}
        selected = select_cases(**filters)
        self.assertEqual([(case["job_id"], case["api_lang"]) for case in selected],
                         [(3769, "c"), (3775, "fortran")])
        full_by_id = {case["job_id"]: case for case in select_cases()}
        self.assertEqual(selected, [full_by_id[3769], full_by_id[3775]])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.cases = select_cases(**SMALL_FILTERS)

    def _args(self, **kwargs):
        values = dict(max_cases=None, sample=None, job_ids=None,
                      ids_file="", seed=0)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_max_cases(self):
        selected, meta = apply_selection(self.cases, self._args(max_cases=2))
        self.assertEqual([case["job_id"] for case in selected],
                         [case["job_id"] for case in self.cases[:2]])
        self.assertEqual(meta["mode"], "max_cases")

    def test_sample_is_deterministic(self):
        first, _ = apply_selection(self.cases, self._args(sample=2, seed=7))
        second, _ = apply_selection(self.cases, self._args(sample=2, seed=7))
        self.assertEqual([case["job_id"] for case in first],
                         [case["job_id"] for case in second])
        self.assertEqual(len(first), 2)

    def test_job_ids_and_ids_file(self):
        wanted = [self.cases[2]["job_id"], self.cases[0]["job_id"]]
        selected, meta = apply_selection(self.cases, self._args(job_ids=wanted))
        self.assertEqual([case["job_id"] for case in selected], wanted)
        self.assertEqual(meta["mode"], "job_ids")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(f"{wanted[0]}\n# comment\n{wanted[1]}\n")
            path = handle.name
        try:
            selected, _ = apply_selection(self.cases, self._args(ids_file=path))
            self.assertEqual([case["job_id"] for case in selected], wanted)
        finally:
            Path(path).unlink()

    def test_missing_job_id(self):
        with self.assertRaises(SystemExit):
            apply_selection(self.cases, self._args(job_ids=[999999]))


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


class ScorePTests(unittest.TestCase):
    def test_build_override_rejects_only_mixed_modes(self):
        args = SimpleNamespace(scorep_mpp="none", skip_build=True, executor="local",
                               cpus_per_job_cpu=1, cpus_per_job_gpu=1, k=1)
        case = dict(provider="AIX", device="CPU", api_lang="cpp", scorep="off", dl_mode="cpp")
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(matrix_jobs, "solver_binary", return_value=Path(tmp)), \
             mock.patch.object(matrix_jobs, "scorep_elf", return_value="libscorep_measurement.so"), \
             mock.patch.dict(matrix_jobs.os.environ, {"MODULE_TEST_BUILD_DIR": tmp}):
            with self.assertRaisesRegex(matrix_jobs.PreflightError, "MODULE_TEST_BUILD_DIR.*mixed"):
                matrix_jobs.preflight([case, {**case, "scorep": "on"}], args, tmp)
            # Use a real file for the single-mode binary checks.
            solver = Path(tmp) / "solver"
            solver.touch()
            with mock.patch.object(matrix_jobs, "solver_binary", return_value=solver):
                matrix_jobs.preflight([case], args, tmp)
                matrix_jobs.preflight([{**case, "scorep": "on"}], args, tmp)
            with mock.patch.dict(matrix_jobs.os.environ, {"MODULE_TEST_BUILD_DIR": ""}):
                with mock.patch.object(matrix_jobs, "solver_binary", return_value=solver):
                    matrix_jobs.preflight([case, {**case, "scorep": "on"}], args, tmp)

    def test_client_build_failure_is_preflight_error(self):
        args = SimpleNamespace(scorep_mpp="none", skip_build=False, executor="local",
                               cpus_per_job_cpu=1, cpus_per_job_gpu=1, k=1)
        case = dict(provider="PHYDLL", device="CPU", api_lang="cpp", scorep="off", dl_mode="cpp")
        with tempfile.TemporaryDirectory() as tmp:
            solver = Path(tmp) / "solver"
            solver.touch()
            with mock.patch.object(matrix_jobs, "solver_binary", return_value=solver), \
                 mock.patch.object(matrix_jobs, "prepare_phydll_dl_client",
                                   side_effect=RuntimeError("configure failed")):
                with self.assertRaisesRegex(matrix_jobs.PreflightError, "configure failed"):
                    matrix_jobs.preflight([case], args, tmp)

    def test_submit_client_failure_reports_error_without_traceback(self):
        case = select_cases(**SMALL_FILTERS)[0]
        with tempfile.TemporaryDirectory() as tmp:
            solver = Path(tmp) / "solver"
            solver.touch()
            run = {**minimal_run("run", 1), "cpus_per_job_cpu": 1, "cpus_per_job_gpu": 1}
            with mock.patch.object(matrix_jobs, "select_cases", return_value=[case]), \
                 mock.patch.object(matrix_jobs, "build_run_record", return_value=run), \
                 mock.patch.object(matrix_jobs, "create_run_and_jobs", return_value=(mock.Mock(), tmp, None)), \
                 mock.patch.object(matrix_jobs, "solver_binary", return_value=solver), \
                 mock.patch.object(matrix_jobs, "prepare_phydll_dl_client", side_effect=RuntimeError("configure failed")), \
                 mock.patch.object(matrix_jobs, "update_run_status") as status, \
                 mock.patch.object(matrix_jobs, "run_local_scheduler") as scheduler, \
                 mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                args = matrix_jobs.build_parser().parse_args(["submit", "--run-id", "run"])
                self.assertEqual(matrix_jobs.cmd_submit(args), 2)
                self.assertEqual(status.call_args.args[2], "error")
                self.assertIn("Preflight failed: configure failed", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                scheduler.assert_not_called()

    def test_rerun_scorep_mpp_inheritance_and_preflight_failure(self):
        case = select_cases(**SMALL_FILTERS)[0]
        source = {**minimal_run("source", 1), "scorep_mpp": "mpi",
                  "partition_cpu": "", "partition_gpu": "", "account_cpu": "",
                  "account_gpu": "", "gres_gpu": "", "time_limit": "",
                  "sbatch_extra": ""}
        for option, expected in (([], "mpi"), (["--scorep-mpp", "none"], "none")):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp, \
                 mock.patch.dict(matrix_jobs.os.environ, {"SCOREP_MPP": "other"}), \
                 mock.patch.object(matrix_jobs, "load_run", return_value=(mock.Mock(), source, None, Path(tmp))), \
                 mock.patch.object(matrix_jobs, "get_jobs", return_value=[case]), \
                 mock.patch.object(matrix_jobs, "build_provenance", return_value={}), \
                 mock.patch.object(matrix_jobs, "create_run_and_jobs", return_value=(mock.Mock(), tmp, None)) as create, \
                 mock.patch.object(matrix_jobs, "preflight", side_effect=matrix_jobs.PreflightError("configure failed")) as preflight, \
                 mock.patch.object(matrix_jobs, "update_run_status") as status, \
                 mock.patch.object(matrix_jobs, "run_local_scheduler") as scheduler, \
                 mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
                args = matrix_jobs.build_parser().parse_args(["rerun", "source", *option])
                self.assertEqual(matrix_jobs.cmd_rerun(args), 2)
                self.assertEqual(create.call_args.args[0]["scorep_mpp"], expected)
                self.assertEqual(preflight.call_args.args[1].scorep_mpp, expected)
                self.assertEqual(status.call_args.args[2], "error")
                self.assertIn("Preflight failed: configure failed", stderr.getvalue())
                self.assertNotIn("Traceback", stderr.getvalue())
                scheduler.assert_not_called()

    def test_preflight_elf_checks_and_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            solver = root / "solver"
            solver.touch()
            client = root / "dl_clients" / "build-module-test-scorep-none" / "phydll_dl_client"
            client.parent.mkdir(parents=True)
            client.touch()
            aix = root / "extern" / "AIxeleratorService" / "INSTALL-SCOREP" / "lib" / "libAIxeleratorService.so"
            aix.parent.mkdir(parents=True)
            aix.touch()
            args = SimpleNamespace(scorep_mpp="none", skip_build=True, executor="local",
                                   cpus_per_job_cpu=1, cpus_per_job_gpu=1, k=1)
            case = dict(provider="PHYDLL", device="CPU", api_lang="cpp", scorep="on", dl_mode="cpp")
            good = "Shared library: [libscorep_measurement.so.11]"
            with mock.patch.object(matrix_jobs, "solver_binary", return_value=solver), \
                 mock.patch.object(matrix_jobs, "CMI_DIR", root), \
                 mock.patch.object(matrix_jobs, "scorep_elf") as inspect:
                inspect.side_effect = lambda path: good if path != aix else "[libnvidia-ml.so.1]"
                with self.assertRaisesRegex(matrix_jobs.PreflightError, "libnvidia-ml"):
                    matrix_jobs.preflight([case], args, root)
                inspect.side_effect = lambda path: "" if path == client else good
                with self.assertRaisesRegex(matrix_jobs.PreflightError, "C\\+\\+ client"):
                    matrix_jobs.preflight([case], args, root)
                inspect.side_effect = lambda path: "" if path == solver else good
                with self.assertRaisesRegex(matrix_jobs.PreflightError, "solver"):
                    matrix_jobs.preflight([case], args, root)
                args.scorep_mpp = "mpi"
                with self.assertRaisesRegex(matrix_jobs.PreflightError, "unsupported"):
                    matrix_jobs.preflight([case], args, root)
                args.scorep_mpp = "none"
                inspect.reset_mock(side_effect=True)
                matrix_jobs.preflight([{**case, "scorep": "off", "provider": "AIX"}], args, root)
                inspect.assert_not_called()

    def test_profile_validation_and_unique_attempts(self):
        run = {**minimal_run("run", 1), "scorep_mpp": "none"}
        job = dict(job_id=1, seq=1, provider="PHYDLL", device="CPU", effective_model="watercnn",
                   api_mode="STATIC", api_lang="cpp", dl_mode="cpp", layout="packed",
                   scorep="on", steps=1, clients=2, batch_size=1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prefixes = []
            ranks = [0, 1]
            def launch(*args, **kwargs):
                prefix = kwargs["env"].get("SCOREP_EXPERIMENT_DIRECTORY")
                prefixes.append(prefix)
                if prefix:
                    for rank in ranks:
                        profile = Path(f"{prefix}_rank_{rank}") / "profile.cubex"
                        profile.parent.mkdir(parents=True)
                        profile.write_bytes(b"profile")
                return SimpleNamespace(pid=123, returncode=0,
                                       communicate=lambda **kw: ("ok", None))
            with mock.patch.object(matrix_job.subprocess, "Popen", side_effect=launch), \
                 mock.patch.object(matrix_job.os, "getpgid", return_value=123), \
                 mock.patch.object(matrix_job, "ResourceMonitor") as monitor:
                monitor.return_value.max_total_kb = 0
                monitor.return_value.max_gpu_mb = 0
                def execute():
                    return matrix_job.run_case(run, job, root / "config", root / "state", root / "logs" / "1.log")
                result = execute()
                self.assertIn("rank_2/profile.cubex", result["profile_error"])
                self.assertIn("rank_2/profile.cubex", (root / "logs" / "1.log").read_text())
                ranks.append(2)
                self.assertIsNone(execute()["profile_error"])
                self.assertNotEqual(prefixes[0], prefixes[1])
                job["dl_mode"] = "python"
                ranks[:] = [0, 1]
                self.assertIn("rank_2/profile.cubex", execute()["profile_error"])
                ranks.append(2)
                self.assertIsNone(execute()["profile_error"])
                job["provider"] = "SMARTSIM"
                ranks[:] = [0]
                self.assertIn("rank_1/profile.cubex", execute()["profile_error"])
                ranks[:] = [0, 1]
                self.assertIsNone(execute()["profile_error"])
                job["scorep"] = "off"
                self.assertIsNone(execute()["profile_error"])


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
    def test_old_universe_metadata_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_hash = "e921ee89cc55"
            path = db.db_path(tmp, old_hash)
            conn = db.init_db(path, old_hash, 1)
            conn.close()
            cases = select_cases(**SMALL_FILTERS)[:1]
            original = path.read_bytes()
            conn, _, _ = create_run_and_jobs(minimal_run("new-run", 1), cases, tmp)
            conn.close()
            self.assertEqual(path.read_bytes(), original)
            run = minimal_run("old-rerun", len(cases))
            run.update(universe_hash=old_hash, matrix_version=1)
            conn, _, _ = create_run_and_jobs(run, cases, tmp)
            meta = dict(conn.execute("SELECT key, value FROM schema_meta"))
            self.assertEqual(meta["universe_hash"], old_hash)
            self.assertEqual(meta["matrix_version"], "1")
            self.assertEqual(db.get_run(conn, "old-rerun")["matrix_version"], 1)
            conn.close()

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


class CleanupTests(unittest.TestCase):
    def test_cleanup_successful_job(self):
        import matrix_job
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = root / "state" / "00001"
            state.mkdir(parents=True)
            (state / "temp.txt").write_text("data")
            config = root / "configs" / "00001.toml"
            config.parent.mkdir(parents=True)
            config.write_text("config")
            log = root / "logs" / "00001.log"
            log.parent.mkdir(parents=True)
            log.write_text("log")
            rank_dir = root / "scorep" / "00001_uuid_rank_0"
            rank_dir.mkdir(parents=True)
            (rank_dir / "profile.cubex").write_text("profile")
            (rank_dir / "MANIFEST.md").write_text("manifest")
            (rank_dir / "scorep.cfg").write_text("cfg")

            job = {"job_id": 1, "scorep": "on", "provider": "AIX"}
            run = {"run_id": "r1"}

            matrix_job.cleanup_successful_job(run, job, state, config, log)

            self.assertFalse(state.exists())
            self.assertFalse(config.exists())
            self.assertTrue((rank_dir / "profile.cubex").is_file())
            self.assertFalse((rank_dir / "MANIFEST.md").exists())
            self.assertFalse((rank_dir / "scorep.cfg").exists())

    def test_compact_run_directory(self):
        from matrix_jobs import compact_run_directory
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "state" / "00001").mkdir(parents=True)
            (run_dir / "configs" / "templates").mkdir(parents=True)
            (run_dir / "configs" / "00001.toml").write_text("c1")
            (run_dir / "configs" / "templates" / "t.toml").write_text("t")
            rank_dir = run_dir / "scorep" / "00001_uuid_rank_0"
            rank_dir.mkdir(parents=True)
            (rank_dir / "profile.cubex").write_text("p")
            (rank_dir / "MANIFEST.md").write_text("m")

            files, dirs = compact_run_directory(run_dir)
            self.assertFalse((run_dir / "configs" / "00001.toml").exists())
            self.assertTrue((run_dir / "configs" / "templates" / "t.toml").is_file())
            self.assertFalse((rank_dir / "MANIFEST.md").exists())
            self.assertTrue((rank_dir / "profile.cubex").is_file())
            self.assertFalse((run_dir / "state").exists())

    def test_compact_rejects_active_run(self):
        conn = mock.Mock()
        run = {**minimal_run("run", 1), "status": "running"}
        args = SimpleNamespace(archive_scorep=False)
        with mock.patch.object(matrix_jobs, "load_run", return_value=(conn, run, None, Path("/tmp"))), \
             mock.patch.object(matrix_jobs, "compact_run_directory") as compact, \
             mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(matrix_jobs.cmd_compact(args), 2)
            compact.assert_not_called()
            conn.close.assert_called_once()

    def test_compact_preserves_unrelated_logs_and_existing_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            (run_dir / "state" / "00001").mkdir(parents=True)
            (run_dir / "scorep").mkdir()
            (run_dir / "scorep" / "profile").write_text("new")
            archive = run_dir / "scorep.tar"
            archive.write_text("old")
            log = root / "phydll-other.log"
            log.write_text("other run")
            with mock.patch.object(matrix_jobs, "MODULE_TEST_DIR", root):
                with self.assertRaises(FileExistsError):
                    matrix_jobs.compact_run_directory(run_dir, archive_scorep=True)
                self.assertTrue((run_dir / "state" / "00001").is_dir())
                self.assertEqual(archive.read_text(), "old")
                matrix_jobs.compact_run_directory(run_dir)
                self.assertTrue(log.is_file())


if __name__ == "__main__":
    unittest.main()
