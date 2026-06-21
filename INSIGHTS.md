# Session Insights: Module Test & Flexible API Verification

## 1. Architectural Findings: Multi-Input Staging
The handling of multiple staged inputs (`ORDERED_MULTI`, `KEYED_MULTI`) relies on a two-step process involving both the interface and the provider:

*   **Interface-Level Fallback:** Currently, none of the providers (SmartSim, AIX, PhyDLL) explicitly implement the optimized `flex_ordered_*` or `flex_keyed_*` methods. Instead, they rely on the **base class fallback logic**. This logic aggregates multiple `set()` calls into a single `MLCouplingData` object containing multiple tensors, which is then passed to the provider's standard `static_inference()` method.
*   **Provider-Specific "Static" Multi-Input Support:**
    *   **SmartSim:** Its `static_inference` is inherently designed to iterate over all tensors in a collection and stage them as distinct keys (e.g., `input_0_0`, `input_0_1`). This allows it to support the aggregation from the flex-fallback "natively."
    *   **PhyDLL:** Its `static_inference` flattens all tensors in the collection into a single stream. Like SmartSim, its static path is broad enough to handle the results of the flex-fallback.
    *   **AIX:** Its `static_inference` is currently restricted to a single flat memory region (processing only `tensors[0]`). It only worked in multi-staging tests because the tensors happened to be contiguous.

## 2. The Point of the Flexible API
The primary purpose of the `ordered()` and `keyed()` views is to allow providers to implement **optimized paths** (e.g., streaming data to the GPU as it's staged, or performing partial updates).
*   **Current State:** We have successfully verified that the **aggregation logic** and the **hand-off to static paths** are correct.
*   **Future Path:** Providers can now be extended to override the `flex_` methods for performance, without breaking the existing verified "fallback-to-static" flow.

## 3. SmartSim & GPU Indexing Logic
We discovered a critical detail in how SmartRedis/SmartSim handles GPU selection:

*   **Upper-Bound Logic:** The `num_gpus` parameter is **not a count**, but the **upper bound** of a zero-based index interval: `[first_gpu, num_gpus)`.
*   **Absolute Mapping:** If using absolute IDs on a 4-GPU node to target GPU 3, you must set:
    *   `first_gpu = 3`
    *   `num_gpus = 4`
*   **Relative Mapping (Preferred):** To avoid index headaches, use `CUDA_VISIBLE_DEVICES`. If `CUDA_VISIBLE_DEVICES=3` is set, the hardware appears as index 0 to the process. In this case, set:
    *   `first_gpu = 0`
    *   `num_gpus = 1`

## 4. SmartSim Model Caching on GPU
SmartSim on GPU may **cache models** in memory based on the internal `model_name` string. If you change the `model_path` in the TOML but keep the same `model_name`, it may continue to execute the previous model. 
*   **Insight:** Always assign a unique `model_name` in the TOML for each distinct architecture/variant (e.g., `perfect` vs `perfect_split_flat`) to force a reload.

## 5. Tooling & Automation
*   **Robust TOML Updating:** Use regex without line-start anchors (`^`) to update TOML files, as keys are often indented under section headers like `[provider]`.
*   **Model Selection:** `test_matrix.py` was updated to dynamically append `_split_flat` for SmartSim multi-input runs, ensuring the correct model variant is used automatically.
*   **Generalized Controller (smartsim_controller.py):**
    *   **Launcher Constraints:** Discovered that `db.set_run_arg("export", "ALL")` is strictly a Slurm-side configuration. Calling it with the `local` launcher results in an `AttributeError` within SmartSim's Orchestrator logic.
    *   **Fix:** The controller was patched to make the `export ALL` call conditional on the `slurm` launcher, ensuring stability for both local development and cluster execution.
