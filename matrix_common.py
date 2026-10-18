#!/usr/bin/env python3
"""Shared definitions for the sequential matrix and the job-based scheduler.

The axis definition below is the single source of truth for the test universe.
Changing an axis or an applicability rule changes ``universe_hash`` and therefore
creates a new run folder; bump ``MATRIX_VERSION`` when the *meaning* of the
matrix changes without the spec changing.
"""

import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_TEST_DIR = Path(__file__).resolve().parent
BASE_DIR = MODULE_TEST_DIR.parent
TRAIN_MODELS_DIR = BASE_DIR / "mini_app" / "train_models" / "model_a"
CMI_DIR = Path(os.environ.get("CMI_DIR", str(BASE_DIR / "CPP-ML-Interface")))

# ---------------------------------------------------------------------------
# Axes (order defines the global job ordinal)
# ---------------------------------------------------------------------------
MATRIX_VERSION = 1

# Cluster defaults: GPU nodes are 96c/4g, so one GPU is billed as 24 cores.
DEFAULT_CPU_CORES = 4
DEFAULT_GPU_CORES = 24

PROVIDERS = ["AIX", "PHYDLL", "SMARTSIM"]
DEVICES = ["CPU", "GPU"]
MODELS = ["perfect", "transformer", "giant", "watercnn", "mmcp_transformer"]
API_MODES = ["STATIC", "ORDERED", "KEYED", "ORDERED_MULTI", "KEYED_MULTI"]
WORKLOADS = [(1, 1), (5, 1), (5, 2)]
BATCH_SIZES = [1, 7]

PHYDLL_DL_MODES = ["cpp", "python"]
PHYDLL_TRANSPORT_LAYOUTS = ["packed", "uniform_chunks"]
API_LANGS_ALL = ["cpp", "c", "fortran"]

# Fields stored per case.  ``dl_mode``/``layout`` use "-" for non-PhyDLL jobs.
CASE_FIELDS = [
    "provider",
    "device",
    "model",
    "dl_mode",
    "layout",
    "scorep",
    "api_mode",
    "api_lang",
    "steps",
    "clients",
    "batch_size",
]

# Canonical axis order used for the global ordinal.  This matches the row order
# of the original sequential loop in test_matrix.py.
AXIS_ORDER = [
    "provider",
    "device",
    "model",
    "dl_mode",
    "layout",
    "scorep",
    "api_mode",
    "api_lang",
    "workload",
    "batch_size",
]

UNIVERSE_SPEC = {
    "version": MATRIX_VERSION,
    "axis_order": AXIS_ORDER,
    "providers": PROVIDERS,
    "devices": DEVICES,
    "models": MODELS,
    "api_modes": API_MODES,
    "workloads": [list(w) for w in WORKLOADS],
    "batch_sizes": BATCH_SIZES,
    "phydll_dl_modes": PHYDLL_DL_MODES,
    "phydll_transport_layouts": PHYDLL_TRANSPORT_LAYOUTS,
    "api_langs": API_LANGS_ALL,
    "rules": {
        "mmcp_requires_multi": True,
        "non_static_cpp_only": True,
        "phydll_owns_dl_mode_layout": True,
        "smartsim_multi_split_flat": True,
    },
}

_SPEC_CANONICAL = json.dumps(UNIVERSE_SPEC, sort_keys=True, separators=(",", ":"))
UNIVERSE_HASH = hashlib.sha256(_SPEC_CANONICAL.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Case enumeration
# ---------------------------------------------------------------------------
def _pick(value, allowed):
    if value is None:
        return list(allowed)
    wanted = set(value)
    return [item for item in allowed if item in wanted]


def langs_for_mode(api_mode, api_langs=None):
    langs = list(api_langs) if api_langs is not None else list(API_LANGS_ALL)
    if api_mode == "STATIC":
        return langs
    return [lang for lang in langs if lang == "cpp"]


def effective_model(provider, model, api_mode):
    if provider == "SMARTSIM" and "MULTI" in api_mode and "mmcp" not in model:
        return f"{model}_split_flat"
    return model


def config_template_name(provider, device, api_mode):
    provider_l = provider.lower()
    device_l = device.lower()
    if provider == "AIX" and api_mode != "STATIC":
        return f"config_aix_{device_l}_flex.toml"
    return f"config_{provider_l}_{device_l}.toml"


def case_from_params(provider, device, model, dl_mode, layout, scorep,
                     api_mode, api_lang, steps, clients, batch_size):
    return {
        "provider": provider,
        "device": device,
        "model": model,
        "dl_mode": dl_mode,
        "layout": layout,
        "scorep": scorep,
        "api_mode": api_mode,
        "api_lang": api_lang,
        "steps": int(steps),
        "clients": int(clients),
        "batch_size": int(batch_size),
    }


def iter_cases(providers=None, devices=None, models=None, api_modes=None,
               api_langs=None, workloads=None, batch_sizes=None, scorep=None,
               transport_layouts=None, dl_modes=None):
    """Yield every valid case matching the filters, in canonical order.

    ``None`` means "all defaults".  This is the same set that the sequential
    matrix runs and the same set the global job ordinals are computed over.
    """
    providers = _pick(providers, PROVIDERS)
    devices = _pick(devices, DEVICES)
    models = _pick(models, MODELS)
    api_modes = _pick(api_modes, API_MODES)
    api_langs = _pick(api_langs, API_LANGS_ALL)
    workloads = list(workloads) if workloads is not None else list(WORKLOADS)
    batch_sizes = list(batch_sizes) if batch_sizes is not None else list(BATCH_SIZES)
    scorep = _pick(scorep, ["off", "on"])
    dl_modes = _pick(dl_modes, PHYDLL_DL_MODES)
    transport_layouts = _pick(transport_layouts, PHYDLL_TRANSPORT_LAYOUTS)

    for provider in providers:
        if provider == "PHYDLL":
            dl_variants = [
                (dl_mode, layout)
                for dl_mode in dl_modes
                for layout in transport_layouts
            ]
        else:
            dl_variants = [("-", "-")]

        for device in devices:
            for model in models:
                for dl_mode, layout in dl_variants:
                    for scorep_mode in scorep:
                        for api_mode in api_modes:
                            if model == "mmcp_transformer" and "MULTI" not in api_mode:
                                continue
                            for api_lang in langs_for_mode(api_mode, api_langs):
                                for steps, clients in workloads:
                                    for batch_size in batch_sizes:
                                        yield case_from_params(
                                            provider, device, model,
                                            dl_mode, layout, scorep_mode,
                                            api_mode, api_lang,
                                            steps, clients, batch_size,
                                        )


def case_index():
    """Return {job_id: case} and {params_hash: job_id} for the full universe."""
    by_id = {}
    by_hash = {}
    for ordinal, case in enumerate(iter_cases(), start=1):
        case = dict(case)
        case["job_id"] = ordinal
        case["params_hash"] = params_hash(case)
        by_id[ordinal] = case
        by_hash[case["params_hash"]] = ordinal
    return by_id, by_hash


def select_cases(**filters):
    """Return the selected cases with their *global* job_id and params_hash."""
    _, by_hash = case_index()
    selected = []
    for case in iter_cases(**filters):
        digest = params_hash(case)
        job_id = by_hash.get(digest)
        if job_id is None:
            raise KeyError(f"case is not part of the matrix universe: {case}")
        enriched = dict(case)
        enriched["job_id"] = job_id
        enriched["params_hash"] = digest
        selected.append(enriched)
    return selected


def _lookup_id(by_id, digest, case):
    for job_id, candidate in by_id.items():
        if candidate["params_hash"] == digest:
            return job_id
    # Should never happen: fall back to an exact field comparison.
    for job_id, candidate in by_id.items():
        if all(candidate.get(field) == case.get(field) for field in CASE_FIELDS):
            return job_id
    raise KeyError(f"case is not part of the matrix universe: {case}")


def params_hash(case):
    payload = {field: case[field] for field in CASE_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def base_model(model):
    return model.replace("_split_flat", "")


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------
def update_toml(toml_path, provider, device, model_name, transport_layout="packed"):
    suffix = "cuda" if device == "GPU" else "cpu"
    if model_name == "multi_input":
        model_file = MODULE_TEST_DIR / "multi_input_model.pt"
    elif model_name == "mmcp_transformer":
        model_file = Path(
            "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/"
            "input/transformer_inference_scripted_fw2.pt"
        )
    else:
        model_file = TRAIN_MODELS_DIR / f"{model_name}_{suffix}.pt"

    with open(toml_path, "r", encoding="utf-8") as handle:
        content = handle.read()

    content = re.sub(r'model_file\s*=.*', f'model_file = "{model_file}"', content)
    content = re.sub(r'model_path\s*=.*', f'model_path = "{model_file}"', content)
    content = re.sub(r'model_name\s*=.*', f'model_name = "{model_name}"', content)
    content = re.sub(r'device\s*=.*', f'device = "{device}"', content)

    if provider == "SMARTSIM":
        num_gpus_val = "1" if device == "GPU" else "0"
        for key, val in [("num_gpus", num_gpus_val), ("first_gpu", "0")]:
            pattern = rf'{key}\s*=.*'
            if re.search(pattern, content):
                content = re.sub(pattern, f'{key} = {val}', content)
            else:
                content = content.replace("[library]", f"[library]\n{key} = {val}")
    elif provider == "PHYDLL":
        pattern = r'transport_layout\s*=.*'
        replacement = f'transport_layout = "{transport_layout}"'
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
        else:
            content = content.replace("[library]", f"[library]\n{replacement}")

    with open(toml_path, "w", encoding="utf-8") as handle:
        handle.write(content)


def make_job_config(template_path, out_path, case):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(template_path, "r", encoding="utf-8") as src:
        content = src.read()
    with open(out_path, "w", encoding="utf-8") as dst:
        dst.write(content)
    update_toml(
        out_path,
        case["provider"],
        case["device"],
        effective_model(case["provider"], case["model"], case["api_mode"]),
        transport_layout=case.get("layout") if case.get("layout") != "-" else "packed",
    )
    return out_path


# ---------------------------------------------------------------------------
# Output parsing / comparison
# ---------------------------------------------------------------------------
def parse_summary(output):
    outputs_first_vals = None
    outputs_sum_vals = None
    legacy_vals = None

    lines = output.splitlines()
    for index, line in enumerate(lines):
        line_str = line.strip()
        if "Gathered output[0] from all ranks:" in line_str:
            for candidate_line in lines[index + 1:index + 5]:
                candidate = candidate_line.strip()
                if candidate.startswith("[[") and candidate.endswith("]]"):
                    try:
                        outputs_first_vals = json.loads(candidate)
                        break
                    except Exception:
                        pass
        elif "Gathered output sums from all ranks:" in line_str:
            for candidate_line in lines[index + 1:index + 5]:
                candidate = candidate_line.strip()
                if candidate.startswith("[[") and candidate.endswith("]]"):
                    try:
                        outputs_sum_vals = json.loads(candidate)
                        break
                    except Exception:
                        pass
        elif "Gathered outputs from all ranks:" in line_str:
            for candidate_line in lines[index + 1:index + 5]:
                candidate = candidate_line.strip()
                if candidate.startswith("[[") and candidate.endswith("]]"):
                    try:
                        legacy_vals = json.loads(candidate)
                        break
                    except Exception:
                        pass

    if outputs_first_vals is not None or outputs_sum_vals is not None:
        result = {}
        if outputs_first_vals is not None:
            result["first"] = outputs_first_vals
        if outputs_sum_vals is not None:
            result["sum"] = outputs_sum_vals
        return json.dumps(result)
    if legacy_vals is not None:
        return json.dumps({"first": legacy_vals})
    return "N/A"


def compare_results(a_str, b_str, rel_tol=1e-3, abs_tol=1e-3):
    if a_str == b_str:
        return True
    try:
        a = json.loads(a_str)
        b = json.loads(b_str)

        def _deep_close(x, y):
            if type(x) != type(y):
                return False
            if isinstance(x, (int, float)):
                return math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)
            if isinstance(x, list):
                if len(x) != len(y):
                    return False
                return all(_deep_close(i, j) for i, j in zip(x, y))
            if isinstance(x, dict):
                if x.keys() != y.keys():
                    return False
                return all(_deep_close(x[k], y[k]) for k in x)
            return x == y

        return _deep_close(a, b)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def safe_token(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def parse_workload(text):
    steps, clients = text.split("/", 1)
    return int(steps), int(clients)


def median(values):
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return 0.5 * (values[mid - 1] + values[mid])


def format_duration(seconds):
    if seconds is None:
        return "-"
    seconds = int(round(float(seconds)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d{hours:02d}h{minutes:02d}m"
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def git_revision(path):
    try:
        rev = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
        return {"revision": rev, "dirty": dirty}
    except Exception:
        return {"revision": None, "dirty": None}


def prepare_phydll_dl_client(scorep_mode, cmi_dir=None, module_test_dir=None):
    cmi_dir = Path(cmi_dir or CMI_DIR)
    module_test_dir = Path(module_test_dir or MODULE_TEST_DIR)
    build_dir = cmi_dir / "dl_clients" / (
        "build-scorep-none" if scorep_mode == "on" else "build-module-test"
    )
    cmake_args = ["cmake", "-S", str(cmi_dir / "dl_clients"), "-B", str(build_dir)]
    if scorep_mode == "on":
        cmake_args += ["-DWITH_SCOREP=ON", "-DCPPML_SCOREP_MPP=none"]
    else:
        cmake_args += ["-DWITH_SCOREP=OFF"]

    print(f"[Preparing] PhyDLL C++ client ({scorep_mode}): {build_dir}", flush=True)
    setup = (
        f"export USE_SCOREP={'1' if scorep_mode == 'on' else '0'} SCOREP_MPP=none; "
        f"source {shlex.quote(str(module_test_dir.parent / 'set_env_claix23_cuda12.4.sh'))}; "
    )
    if subprocess.run(["bash", "-lc", setup + shlex.join(cmake_args)],
                      cwd=module_test_dir).returncode != 0:
        raise RuntimeError(f"Failed to configure PhyDLL C++ client ({scorep_mode})")
    build_args = ["cmake", "--build", str(build_dir), "--target", "phydll_dl_client", "-j"]
    if subprocess.run(["bash", "-lc", setup + shlex.join(build_args)],
                      cwd=module_test_dir).returncode != 0:
        raise RuntimeError(f"Failed to build PhyDLL C++ client ({scorep_mode})")
    return build_dir


def build_provenance(cmi_dir=None):
    cmi_dir = Path(cmi_dir or CMI_DIR)
    return {
        "matrix_version": MATRIX_VERSION,
        "universe_hash": UNIVERSE_HASH,
        "cmi": git_revision(cmi_dir),
        "module_test": git_revision(MODULE_TEST_DIR),
        "python": sys.version.split()[0],
        "scorep_mpp": os.environ.get("SCOREP_MPP", "none"),
    }


def resolve_cores(cpus=None, cpu=None, gpu=None, source=None):
    """Resolve (base, cpu, gpu) core counts.

    * explicit per-device values win;
    * ``--cpus-per-job`` sets both;
    * otherwise fresh runs use 4 CPU / 24 GPU cores;
    * reruns inherit the source run's values.
    """
    if cpus is not None:
        base = int(cpus)
        cpu_value = int(cpu) if cpu is not None else base
        gpu_value = int(gpu) if gpu is not None else base
        return base, cpu_value, gpu_value

    if source is not None:
        base = int(source.get("cpus_per_job") or DEFAULT_CPU_CORES)
        source_cpu = source.get("cpus_per_job_cpu")
        source_gpu = source.get("cpus_per_job_gpu")
    else:
        base = DEFAULT_CPU_CORES
        source_cpu = None
        source_gpu = None

    cpu_value = int(cpu) if cpu is not None else (
        int(source_cpu) if source_cpu is not None else DEFAULT_CPU_CORES
    )
    gpu_value = int(gpu) if gpu is not None else (
        int(source_gpu) if source_gpu is not None else DEFAULT_GPU_CORES
    )
    return base, cpu_value, gpu_value


def cores_for_job(run, device):
    """Per-device core/thread cap with a fallback to the global setting."""
    key = "cpus_per_job_cpu" if device == "CPU" else "cpus_per_job_gpu"
    value = run.get(key)
    if value is None:
        value = run.get("cpus_per_job")
    return max(1, int(value or 1))


def run_build_dir(scorep_mode, scorep_mpp="none", module_test_dir=None):
    root = Path(module_test_dir or MODULE_TEST_DIR)
    if scorep_mode == "on":
        if scorep_mpp and scorep_mpp != "mpi":
            return root / f"build-artifact-scorep-{scorep_mpp}"
        return root / "build-artifact-scorep"
    return root / "build-artifact"


def solver_binary(api_lang, scorep_mode, scorep_mpp="none", module_test_dir=None):
    build_dir = run_build_dir(scorep_mode, scorep_mpp, module_test_dir)
    names = {
        "cpp": "module_test_solver",
        "c": "module_test_solver_c",
        "fortran": "module_test_solver_fortran",
    }
    return build_dir / names[api_lang]
