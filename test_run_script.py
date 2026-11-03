#!/usr/bin/env python3
"""Isolated run.sh regressions using stubbed controller, MPI and CMake."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class RunScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module = self.root / "module_test"
        self.module.mkdir()
        shutil.copy2(Path(__file__).with_name("run.sh"), self.module / "run.sh")
        (self.root / "set_env_claix23_cuda12.4.sh").write_text("")
        self.build = self.root / "build"
        self.build.mkdir()
        self.executable(self.build / "module_test_solver", "#!/bin/sh\nexit 0\n")
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.executable(self.bin / "mpirun", "#!/bin/sh\nsleep 0.3\nexit ${MPI_STATUS:-0}\n")
        self.executable(self.bin / "cmake", '#!/bin/sh\nprintf "%s\\n" "$@" >> "$CMAKE_LOG"\n')
        self.executable(self.bin / "controller", """#!/usr/bin/env python3
import os, pathlib, sys, time
args = sys.argv
endpoint = pathlib.Path(args[args.index('--endpoint-file') + 1])
done = pathlib.Path(args[args.index('--done-file') + 1])
endpoint.write_text('localhost:1234')
if os.environ.get('EARLY_EXIT') != '1':
    while not done.exists():
        time.sleep(0.02)
else:
    time.sleep(0.65)
sys.exit(int(os.environ.get('CONTROLLER_STATUS', '0')))
""")
        self.rai = self.root / "redisai.so"
        self.rai.touch()

    def executable(self, path, content):
        path.write_text(content)
        path.chmod(0o755)

    def run_script(self, **options):
        env = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}",
               "MODULE_TEST_BUILD_DIR": str(self.build), "MODULE_TEST_STATE_DIR": str(self.root / "state"),
               "SMARTSIM_EXP_DIR": str(self.root / "experiment"), "CMI_DIR": str(self.root / "cmi"),
               "SMARTSIM_PYTHON": str(self.bin / "controller"), "RAI_PATH": str(self.rai),
               "PROVIDER": "SMARTSIM", "DEVICE": "GPU", "USE_SCOREP": "0",
               "CMAKE_LOG": str(self.root / "cmake.log"), **options}
        return subprocess.run(["bash", str(self.module / "run.sh")], env=env,
                              text=True, capture_output=True, timeout=10)

    def test_controller_failure_after_endpoint_is_reported(self):
        for early in ("0", "1"):
            with self.subTest(early=early):
                result = self.run_script(CONTROLLER_STATUS="7", EARLY_EXIT=early)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertIn("controller failed during shutdown", result.stderr)
        self.assertEqual(self.run_script().returncode, 0)

    def test_solver_failure_is_not_masked_by_cleanup(self):
        result = self.run_script(MPI_STATUS="9", CONTROLLER_STATUS="7")
        self.assertEqual(result.returncode, 9, result.stderr)

    def test_compile_uses_build_prefix(self):
        result = self.run_script(PROVIDER="AIX", COMPILE="1", USE_SCOREP="1", SCOREP_MPP="none")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"-DAIXELERATOR_PREBUILT_INSTALL_PREFIX={self.build}/aixelerator-install",
                      (self.root / "cmake.log").read_text())
        (self.build / "build_state.env").write_text("AIXELERATOR_PREBUILT_INSTALL_PREFIX=/custom/aix\n")
        result = self.run_script(PROVIDER="AIX", COMPILE="1", USE_SCOREP="1", SCOREP_MPP="none")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-DAIXELERATOR_PREBUILT_INSTALL_PREFIX=/custom/aix",
                      (self.root / "cmake.log").read_text())


if __name__ == "__main__":
    unittest.main()
