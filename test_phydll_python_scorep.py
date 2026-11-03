#!/usr/bin/env python3
"""Opt-in real MPMD CPU smoke test for the Python Score-P client."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
REGIONS = ("py_recv", "py_inference", "py_input_unpack", "py_h2d",
           "py_torch_forward", "py_d2h", "py_output_reorder", "py_send")


def check_visits(dump, expected):
    for region in REGIONS:
        match = re.search(r"^user:" + region + r"\(id=\d+\)\s+(\d+)\s*$", dump, re.M)
        if not match or int(match.group(1)) != expected:
            raise AssertionError(f"{region}: expected {expected} visits")


class ProfileTests(unittest.TestCase):
    def test_visit_validation(self):
        dump = "\n".join(f"user:{name}(id={i}) 4" for i, name in enumerate(REGIONS))
        check_visits(dump, 4)
        with self.assertRaises(AssertionError):
            check_visits(dump.replace("user:py_send(id=7) 4", "user:py_send(id=7) 0"), 4)

    @unittest.skipUnless(os.environ.get("PHYDLL_SCOREP_SMOKE") == "1", "opt-in real MPI smoke")
    def test_mpmd_cpu_profile(self):
        config = os.environ["PHYDLL_SCOREP_SMOKE_CONFIG"]
        build = os.environ["PHYDLL_SCOREP_SMOKE_BUILD"]
        cube_dump = os.environ["PHYDLL_SCOREP_CUBE_DUMP"]
        with tempfile.TemporaryDirectory(prefix="phydll-scorep-smoke-") as tmp:
            prefix = Path(tmp) / "scorep"
            env = os.environ.copy()
            env.update(PROVIDER="PHYDLL", DEVICE="CPU", API_MODE="KEYED_MULTI",
                       API_LANG="cpp", STEPS="5", CLIENTS="2", MODEL="watercnn",
                       BATCH_SIZE="7", MERGE_STRATEGY="LIST", USE_SCOREP="1",
                       SCOREP_MPP="none", USE_PYTHON_DL_CLIENT="1",
                       PHYDLL_PY_SCOREP_WRAPPER="1", MODULE_TEST_BUILD_DIR=build,
                       CONFIG_FILE=config, MODULE_TEST_STATE_DIR=tmp,
                       SMARTSIM_EXP_DIR=str(Path(tmp) / "smartsim"),
                       SCOREP_EXPERIMENT_DIRECTORY=str(prefix),
                       SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY="false",
                       SKIP_MODEL_CONVERSION="1", OMP_NUM_THREADS="2",
                       MLCOUPLING_INTRA_OP_THREADS="2", MLCOUPLING_INTER_OP_THREADS="1",
                       TMPDIR=tmp)
            subprocess.run(["bash", str(ROOT / "run.sh")], cwd=ROOT, env=env,
                           timeout=180, check=True)
            for rank in range(3):
                profile = Path(f"{prefix}_rank_{rank}") / "profile.cubex"
                self.assertTrue(profile.is_file() and profile.stat().st_size > 0, str(profile))
                dump = subprocess.run([cube_dump, "-m", "visits", "-r", "-t", "aggr", str(profile)],
                                      capture_output=True, text=True, check=True).stdout
                if rank == 2:
                    check_visits(dump, 4)


if __name__ == "__main__":
    unittest.main()
