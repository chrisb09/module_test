# Session Insights: Module Test & Flexible API Verification

## 1. Architectural Findings: Multi-Input Staging
There are two distinct ways multi-input staging (`ORDERED_MULTI`, `KEYED_MULTI`) is handled depending on the provider:

*   **Merging Fallback (AIX, PhyDLL):** These providers do not natively support multiple staged inputs. Instead, the interface transparently **merges** all staged tensors into a single contiguous tensor before passing it to the provider. This allows these providers to work with standard single-input models (e.g., `perfect.pt`) even when the user stages data in chunks.
*   **Native Multi-Input (SmartSim):** SmartSim passes each staged tensor as a **distinct input** to the backend. This requires models specifically designed for multiple inputs (e.g., `perfect_split_flat.pt`). If a single-input model is used with multi-staging in SmartSim, it will fail with: *"Number of keys given as INPUTS here does not match model definition"*.

## 2. SmartSim & GPU Indexing Logic
We discovered a critical detail in how SmartRedis/SmartSim handles GPU selection:

*   **Upper-Bound Logic:** The `num_gpus` parameter is **not a count**, but the **upper bound** of a zero-based index interval: `[first_gpu, num_gpus)`.
*   **Absolute Mapping:** If using absolute IDs on a 4-GPU node to target GPU 3, you must set:
    *   `first_gpu = 3`
    *   `num_gpus = 4`
*   **Relative Mapping (Preferred):** To avoid index headaches, use `CUDA_VISIBLE_DEVICES`. If `CUDA_VISIBLE_DEVICES=3` is set, the hardware appears as index 0 to the process. In this case, set:
    *   `first_gpu = 0`
    *   `num_gpus = 1`

## 3. SmartSim Model Caching on GPU
SmartSim on GPU may **cache models** in memory based on the internal `model_name` string. If you change the `model_path` in the TOML but keep the same `model_name`, it may continue to execute the previous model. 
*   **Insight:** Always assign a unique `model_name` in the TOML for each distinct architecture/variant (e.g., `perfect` vs `perfect_split_flat`) to force a reload.

## 4. Verification Results
*   **Bit-Perfection:** We confirmed that `ORDERED_MULTI` staging with the `_split_flat` model variants (which concatenate inputs internally) produces the **exact same numerical output** as `STATIC` mode with single-input models. This verifies that the splitting/merging logic in the interface is bit-perfect.
*   **Fuzzy Comparison:** For consistency checks in automated scripts, a tolerance of `1e-3` (rel/abs) is sufficient to handle minor floating-point fluctuations in complex models (like `transformer` or `giant`) without triggering false anomalies.

## 5. Tooling & Automation
*   **Robust TOML Updating:** Use regex without line-start anchors (`^`) to update TOML files, as keys are often indented under section headers like `[provider]`.
*   **Model Selection:** `test_matrix.py` was updated to dynamically append `_split_flat` for SmartSim multi-input runs, ensuring the correct model variant is used automatically.
