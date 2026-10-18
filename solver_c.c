/*
 * C API solver driver for the module_test matrix.
 *
 * Mirrors solver.cpp for API_MODE=STATIC using the C binding:
 *   config file -> create coupling with live buffers + model_file override
 *   -> step -> MPI gather -> print the machine-readable output blocks.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "c_api.h"

static const char *env_str(const char *name, const char *fallback)
{
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? value : fallback;
}

static int env_int(const char *name, int fallback)
{
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? atoi(value) : fallback;
}

static void fail(const char *message)
{
    fprintf(stderr, "[module_test_c] %s\n", message);
    fflush(stderr);
}

static int check_status(int status, const char *what)
{
    if (status != CMI_SUCCESS) {
        fprintf(stderr, "[module_test_c] %s failed: %s\n", what, cmi_get_last_error());
        fflush(stderr);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    const char *config_path = (argc > 1) ? argv[1] : "config.toml";
    const char *provider = getenv("PROVIDER");
    if (provider == NULL || provider[0] == '\0') {
        fail("PROVIDER is not set. Aborting.");
        return 1;
    }

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fail("MPI_Init failed.");
        return 1;
    }

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    /* Match the C++ solver's hardcoded solver color. */
    const int color = 0;
    MPI_Comm local_comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &local_comm);
    if (local_comm == MPI_COMM_NULL) {
        if (world_rank == 0) {
            fail("Solver communicator is null; check MPMD split logic.");
        }
        MPI_Finalize();
        return 1;
    }

    printf("Hello from rank %d!\n", world_rank);
    if (world_rank == 0) {
        printf("Running with %s provider\n", provider);
        printf("Loading config from %s\n", config_path);
    }
    fflush(stdout);

    const char *api_mode = env_str("API_MODE", "STATIC");
    if (strcmp(api_mode, "STATIC") != 0) {
        fail("C solver currently supports only API_MODE=STATIC.");
        MPI_Finalize();
        return 1;
    }

    const char *model_name = env_str("MODEL", "");
    const int is_mmcp = strstr(model_name, "mmcp") != NULL;
    const int batch_size = env_int("BATCH_SIZE", 1);
    const int in_size = (is_mmcp ? (5 * 512) : 18) * batch_size;
    const int out_size = (is_mmcp ? (2 * 512) : 1) * batch_size;

    float *flat_data = (float *)malloc(sizeof(float) * (size_t)in_size);
    float *output_buffer = (float *)malloc(sizeof(float) * (size_t)out_size);
    if (flat_data == NULL || output_buffer == NULL) {
        fail("Out of memory allocating solver buffers.");
        MPI_Finalize();
        return 1;
    }

    for (int b = 0; b < batch_size; ++b) {
        if (!is_mmcp) {
            const int offset = b * 18;
            for (int i = 0; i < 9; ++i) {
                flat_data[offset + i] = (float)((4 + i * 17) % 100);
            }
            for (int i = 0; i < 9; ++i) {
                flat_data[offset + 9 + i] = (float)((7 + i * 24) % 200);
            }
        }
    }
    if (is_mmcp) {
        for (int i = 0; i < in_size; ++i) {
            flat_data[i] = (float)((4 + i * 17) % 100);
        }
    }
    for (int i = 0; i < in_size; ++i) {
        flat_data[i] *= (float)world_rank;
    }
    for (int i = 0; i < out_size; ++i) {
        output_buffer[i] = -1.0f;
    }

    int in_dims[3] = {batch_size, 18, 0};
    int in_ndims = 2;
    int out_dims[3] = {batch_size, 0, 0};
    int out_ndims = 1;
    if (is_mmcp) {
        in_dims[1] = 5;
        in_dims[2] = 512;
        in_ndims = 3;
        out_dims[1] = 2;
        out_dims[2] = 512;
        out_ndims = 3;
    }

    cmi_tensor_t input_tensor = NULL;
    cmi_tensor_t output_tensor = NULL;
    cmi_data_t input_data = NULL;
    cmi_data_t output_data = NULL;

    if (!check_status(cmi_tensor_create_flat(&input_tensor, flat_data, in_dims, in_ndims,
                                             CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS,
                                             CMI_OWNERSHIP_EXTERNAL), "create input tensor") ||
        !check_status(cmi_data_create(&input_data, CMI_DTYPE_FLOAT), "create input data") ||
        !check_status(cmi_data_add_tensor(input_data, input_tensor), "add input tensor") ||
        !check_status(cmi_tensor_create_flat(&output_tensor, output_buffer, out_dims, out_ndims,
                                             CMI_DTYPE_FLOAT, CMI_LAYOUT_CONTIGUOUS,
                                             CMI_OWNERSHIP_EXTERNAL), "create output tensor") ||
        !check_status(cmi_data_create(&output_data, CMI_DTYPE_FLOAT), "create output data") ||
        !check_status(cmi_data_add_tensor(output_data, output_tensor), "add output tensor")) {
        MPI_Finalize();
        return 1;
    }

    char actual_model_path[4096];
    if (is_mmcp) {
        snprintf(actual_model_path, sizeof(actual_model_path), "%s",
                 "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt");
    } else {
        const char *base_dir = env_str("BASE_DIR", "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim");
        snprintf(actual_model_path, sizeof(actual_model_path),
                 "%s/mini_app/train_models/model_a/%s_cpu.pt", base_dir, model_name);
    }

    cmi_config_overrides_t overrides = NULL;
    if (!check_status(cmi_config_overrides_create(&overrides), "create config overrides")) {
        MPI_Finalize();
        return 1;
    }
    /*
     * SmartSim uses model_path rather than model_file; setting an unknown key
     * would be rejected by the strict factory matching, so only override for
     * providers whose config exposes model_file.
     */
    if (strcmp(provider, "SMARTSIM") != 0) {
        if (!check_status(cmi_config_overrides_set_dotted(overrides, "library.model_file",
                                                          CMI_PARAM_STRING, actual_model_path),
                          "set model_file override")) {
            MPI_Finalize();
            return 1;
        }
    }

    const char *merge_strategy = getenv("MERGE_STRATEGY");
    int merge_strategy_value = -1;
    if (merge_strategy != NULL) {
        if (strcmp(merge_strategy, "STACK") == 0) merge_strategy_value = CMI_MERGE_STACK;
        else if (strcmp(merge_strategy, "LIST") == 0) merge_strategy_value = CMI_MERGE_LIST;
        else if (strcmp(merge_strategy, "AUTO") == 0) merge_strategy_value = CMI_MERGE_AUTO;
        else if (strcmp(merge_strategy, "NONE") == 0) merge_strategy_value = CMI_MERGE_NONE;
    }

    cmi_coupling_t coupling = NULL;
    int status = cmi_coupling_create_from_config_with_data_and_overrides(
        &coupling, config_path,
        CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT, CMI_DTYPE_FLOAT,
        input_data, output_data, overrides);
    if (!check_status(status, "create coupling from config") || coupling == NULL) {
        MPI_Finalize();
        return 2;
    }

    if (merge_strategy_value >= 0) {
        if (!check_status(cmi_coupling_set_merge_strategy(coupling, merge_strategy_value),
                          "set merge strategy")) {
            MPI_Finalize();
            return 2;
        }
    }

    const int num_steps = env_int("STEPS", 1);
    float *outputs_first = (float *)malloc(sizeof(float) * (size_t)num_steps);
    float *outputs_sum = (float *)malloc(sizeof(float) * (size_t)num_steps);
    if (outputs_first == NULL || outputs_sum == NULL) {
        fail("Out of memory allocating step output buffers.");
        MPI_Finalize();
        return 1;
    }

    for (int step = 0; step < (num_steps > 0 ? num_steps : 1); ++step) {
        for (int i = 0; i < in_size; ++i) {
            flat_data[i] += (float)step;
        }

        int delta = 0;
        status = cmi_coupling_step(coupling, &delta);
        if (!check_status(status, "coupling step")) {
            if (world_rank == 0) {
                fprintf(stderr, "[module_test_c] Inference failed at step %d\n", step);
            }
            MPI_Finalize();
            return 3;
        }

        if (world_rank == 0) {
            printf("Inference output: [%f]\n", output_buffer[0]);
        }
        outputs_first[step] = output_buffer[0];

        float sum = 0.0f;
        for (int i = 0; i < out_size; ++i) {
            sum += output_buffer[i];
        }
        outputs_sum[step] = sum;
    }

    const int steps = (num_steps > 0) ? num_steps : 1;

    int local_size = 0;
    MPI_Comm_size(local_comm, &local_size);

    float *all_first = (float *)malloc(sizeof(float) * (size_t)(local_size * steps));
    float *all_sums = (float *)malloc(sizeof(float) * (size_t)(local_size * steps));
    if (all_first == NULL || all_sums == NULL) {
        fail("Out of memory allocating gather buffers.");
        MPI_Finalize();
        return 1;
    }

    MPI_Gather(outputs_first, steps, MPI_FLOAT, all_first, steps, MPI_FLOAT, 0, local_comm);
    MPI_Gather(outputs_sum, steps, MPI_FLOAT, all_sums, steps, MPI_FLOAT, 0, local_comm);

    if (world_rank == 0) {
        printf("\n###########################################################################\n");
        printf("Gathered output[0] from all ranks:\n");
        printf("[");
        for (int rank = 0; rank < local_size; ++rank) {
            if (rank > 0) printf(", ");
            printf("[");
            for (int step = 0; step < steps; ++step) {
                if (step > 0) printf(", ");
                printf("%.4f", all_first[rank * steps + step]);
            }
            printf("]");
        }
        printf("]\n");

        printf("Gathered output sums from all ranks:\n");
        printf("[");
        for (int rank = 0; rank < local_size; ++rank) {
            if (rank > 0) printf(", ");
            printf("[");
            for (int step = 0; step < steps; ++step) {
                if (step > 0) printf(", ");
                printf("%.4f", all_sums[rank * steps + step]);
            }
            printf("]");
        }
        printf("]\n");
        fflush(stdout);
    }

    free(all_first);
    free(all_sums);
    free(outputs_first);
    free(outputs_sum);

    cmi_coupling_destroy(coupling);
    cmi_config_overrides_destroy(overrides);
    cmi_data_destroy(input_data);
    cmi_data_destroy(output_data);
    cmi_tensor_destroy(input_tensor);
    cmi_tensor_destroy(output_tensor);

    free(flat_data);
    free(output_buffer);

    MPI_Finalize();
    /* Match the C++ solver: skip static teardown, which can hang for
     * AIxeleratorService workers after MPI_Finalize. */
    _Exit(0);
}
