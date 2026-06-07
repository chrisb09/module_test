# Plan: Testing Flexible API in `module_test`

## Goal
Extend the existing `module_test` framework to validate the newly implemented `ordered()` and `keyed()` proxy views and their fallback mechanisms across all three providers (AIXelerator, PhyDLL, and SmartSim).

## Challenges
The current test models (`perfect`, `transformer`) expect a **single input tensor** of shape `[1, 18]`.
* If we test the flexible API by staging the data as a single chunk (e.g., `.set(data_1x18)`), it will work perfectly and validate the proxy views and API plumbing.
* If we want to test true **multi-input merging** (e.g., `.set(data_1x9_part1).set(data_1x9_part2)`), the fallback will merge this into an `MLCouplingData` object containing **two** tensors. While PhyDLL handles this seamlessly by flattening everything, SmartSim and AIXelerator will pass two distinct tensors to the Torch model, which will throw an error since the model only expects one.

## Proposed Strategy

We will implement the tests in two phases to handle this gracefully.

### Phase 1: API Plumbing & Fallback Verification (Single-Input)
Validate that the new proxy views work seamlessly with the existing static providers using the current models.

1. **Modify `solver.cpp`:**
   Introduce an `API_MODE` environment variable.
   * `API_MODE=STATIC`: Uses `coupling->step();` (Current behavior)
   * `API_MODE=ORDERED`: Uses `coupling->ordered().set(input_data).inference().get(output_data);`
   * `API_MODE=KEYED`: Uses `coupling->keyed().set("input_1", input_data).inference({"input_1"}, {"output_1"}).get("output_1", output_data);`
2. **Update `run.sh` and `test_matrix.py`:**
   Add the `API_MODE` dimension to the test matrix to ensure all providers pass with the new API structures.

### Phase 2: Multi-Input Merging Verification
Validate the `merge_data` fallback logic using multiple staged inputs.

1. **Generate Multi-Input Model:**
   Create a short Python script in `module_test/` (e.g., `generate_multi_model.py`) that exports a simple TorchScript model. This model will explicitly accept two inputs of shape `[1, 9]`, concatenate them, and perform a basic operation to return a `[1, 1]` output.
2. **Expand `solver.cpp`:**
   Add `API_MODE=ORDERED_MULTI` and `KEYED_MULTI`. In these modes, the solver splits the 18 floats into two `MLCouplingData` objects of size 9 and stages them sequentially or with different keys.
3. **Expand Test Matrix:**
   Run a subset of the test matrix using the new multi-input model and the `_MULTI` API modes to confirm the fallback correctly merges the tensors and the providers successfully execute the inference.

## Implementation Steps for Next Session
1. Update `solver.cpp` to include the `API_MODE` switch and the flexible coupling calls.
2. Update `run.sh` to pass `API_MODE`.
3. Test Phase 1 (Single-Input flexible calls) across all providers.
4. Write `generate_multi_model.py`.
5. Implement Phase 2 (Multi-Input flexible calls) and verify the fallback merging works correctly with SmartSim and AIXelerator.
