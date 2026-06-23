# Flexible API in `module_test` — Post-Mortem

This document used to be a forward-looking plan. Both phases it described
have been implemented and are exercised by `test_matrix.py`. It is kept as a
short record of what was built and the current shape of the flexible API
test surface.

## What was built (was Phase 1 + Phase 2)

### Phase 1 — single-input plumbing & fallback verification

* `solver.cpp` reads `API_MODE` from the environment and dispatches to one
  of six modes:
  * `STATIC` — `coupling->step()` (or `ml_step()`).
  * `ORDERED` — `coupling->ordered().set(input).inference().get(output)`.
  * `KEYED` — `coupling->keyed().set("input_1", input).inference({"input_1"}, {"output_1"}).get("output_1", output)`.
* `run.sh` forwards `API_MODE` and is wired into the Slurm `mpirun` lines
  for all three providers.
* `test_matrix.py` adds the `API_MODE` dimension to the matrix.

### Phase 2 — multi-input merging verification

* New `ORDERED_MULTI` and `KEYED_MULTI` modes in `solver.cpp` split the 18
  floats into two 9-element tensors (or, for `mmcp_transformer`, the
  5×512 features into five 512-element tensors) and stage them sequentially
  or with different keys.
* `_split_flat` model variants were added so the static SmartSim and AIX
  paths can accept the same multi-input shape that the test code produces.
* `MERGE_STRATEGY` was added to control how the flex fallback aggregates
  inputs into a single `MLCouplingData` collection:
  * `LIST` — keep all tensors as a list (default for single-input).
  * `AUTO` — automatic selection (used for `mmcp_transformer`).
  * `NONE` — no merge (used for SmartSim multi-input, which iterates
    tensors naturally).

## Current run

The matrix (`--batch-sizes 1 7`) sweeps every (provider, dl, api, device,
model, steps, clients, batch_size) combination. Use
`analyze_timings.py` to summarize the per-step CSV timings the solver
emits when `TIMING_LOG` is set.

## Cross-references

* `api_guide.md` in `CPP-ML-Interface/documentation/` for the full
  `.ordered()` / `.keyed()` API examples.
* `~/insights/module_test.md` is the authoritative session log; the
  old `module_test/INSIGHTS.md` mirror has been removed.
* `~/insights/phydll_integration.md` for the PhyDLL-side flex behaviour.
