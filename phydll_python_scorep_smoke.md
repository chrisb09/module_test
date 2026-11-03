# PhyDLL Python Score-P smoke

The matrix uses independent `SCOREP_MPP=none` measurements, one profile per
world rank. The Python wrapper uses the real C instrumenter with automatic
Python tracing disabled; explicit client regions are enabled, and the first
frame is intentionally excluded. GPU selection and synchronization are unchanged.

Run the opt-in CPU MPMD smoke on a suitable host or existing allocation (it
does not submit a job). The configuration must point to a CPU watercnn model;
the build must contain the Score-P `mpp=none` solver. For this workspace:

```bash
export PHYDLL_SCOREP_SMOKE=1
export PHYDLL_SCOREP_SMOKE_BUILD="$PWD/build-artifact-scorep-none"
export PHYDLL_SCOREP_SMOKE_CONFIG="$PWD/runs/7db159fd5844/phydll-python-layout-retry-v2/configs/04272.toml"
export PHYDLL_SCOREP_CUBE_DUMP="$PWD/../CPP-ML-Interface/tmp/opencode/scorep-8.4-papi72-install/bin/cube_dump"
python -m unittest -v test_phydll_python_scorep
```

The test runs two solver ranks and one Python DL rank for five steps, parses
all three profiles with `cube_dump`, and requires four visits to each Python
region on rank 2. Temporary profiles are removed afterward. To inspect a
normal matrix run, use `cube_sanity <rank_2>/profile.cubex` and
`cube_dump -m visits -r -t aggr <rank_2>/profile.cubex`; ensure at least two
steps and a loaded model. Matrix validation currently checks only solver
profiles for Python DL jobs, so check rank 2 explicitly.

The smoke covers CPU and `mpp=none` only. `mpp=mpi` relies on MPI's collective
Score-P finalization and requires a consistently MPI-instrumented MPMD build;
GPU behavior requires a separate GPU allocation and verification.
