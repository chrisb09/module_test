#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <fstream>

#include <iomanip> // formatting stuff

#include "../CPP-ML-Interface/include/ml_coupling.hpp"

int main(int argc, char** argv)
{
	const std::string config_path = (argc > 1) ? argv[1] : "config.toml";

	const char* provider = std::getenv("PROVIDER");
	
	if (provider == nullptr || std::string(provider).empty()) {
		std::cerr << "PROVIDER is not set. Aborting.\n";
		return 1;
	}

	const std::string provider_name(provider);
	MPI_Init(&argc, &argv);

	int world_rank = 0;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


	// Hardcode solver color to avoid OpenMPI 5 + Slurm PMIx MPI_APPNUM issues
	const int color = 0;
	MPI_Comm local_comm = MPI_COMM_NULL;
	MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &local_comm);
	if (local_comm == MPI_COMM_NULL) {
		if (world_rank == 0) {
			std::cerr << "Solver communicator is null; check MPMD split logic. Aborting.\n";
		}
		MPI_Finalize();
		return 1;
	}

	std::cout << "Hello from rank " << world_rank << "!\n";

	if (provider_name == "SMARTSIM") {
		if (world_rank == 0) std::cout << "Running with SmartSim provider\n";
		const char* ssdb = std::getenv("SSDB");
		if (ssdb == nullptr || std::string(ssdb).empty()) {
			std::cerr << "SSDB is not set. Aborting.\n";
			MPI_Finalize();
			return 1;
		}

		if (world_rank == 0) {
			std::cout << "Using SSDB=" << ssdb << "\n";
			std::cout << "Loading config from " << config_path << "\n";
		}
	} else if (provider_name == "AIX") {
		if (world_rank == 0) {
			std::cout << "Running with AIX provider\n";
			std::cout << "Loading config from " << config_path << "\n";
		}
	} else if (provider_name == "PHYDLL") {
		if (world_rank == 0) {
			std::cout << "Running with PhyDLL provider\n";
			std::cout << "Loading config from " << config_path << "\n";
		}
	} else {
		std::cerr << "Unsupported provider: " << provider << "\n";
		MPI_Finalize();
		return 1;
	}


	// ******************************
	// Create dummy data
	// ******************************

	const char* model_env = std::getenv("MODEL");
	std::string model_name = (model_env != nullptr) ? model_env : "";
	bool is_mmcp = (model_name.find("mmcp") != std::string::npos);

	const char* batch_size_env = std::getenv("BATCH_SIZE");
	int batch_size = (batch_size_env != nullptr && std::strlen(batch_size_env) > 0) ? std::stoi(batch_size_env) : 1;

	int in_size = (is_mmcp ? (5 * 512) : 18) * batch_size;
	int out_size = (is_mmcp ? (2 * 512) : 1) * batch_size;

	float* flat_data = new float[in_size];
	if (!is_mmcp) {
		for (int b = 0; b < batch_size; ++b) {
			int offset = b * 18;
			for (int i = 0; i < 9; ++i) {
				flat_data[offset + i] = (4 + i * 17) % 100; // First 9 values (water)
			}
			for (int i = 0; i < 9; ++i) {
				flat_data[offset + 9 + i] = (7 + i * 24) % 200; // Next 9 values (terrain)
			}
		}
	} else {
		for (int i = 0; i < in_size; ++i) {
			flat_data[i] = (4 + i * 17) % 100;
		}
	}

	for (int i = 0; i < in_size; ++i) {
		flat_data[i] *= world_rank;
	}

	std::vector<int> single_shape = is_mmcp ? std::vector<int>{batch_size, 5, 512} : std::vector<int>{batch_size, 18};
	MLCouplingTensor<float> input_tensor = MLCouplingTensor<float>::wrap_flat(
		flat_data,
		single_shape,
		MLCouplingMemLayoutContiguous,
		MLCouplingOwnershipExternal);

	MLCouplingData<float> input_data{std::vector<MLCouplingTensor<float>>{
		input_tensor
	}};

	std::cout << "Input data:\n";
	std::cout << input_data.to_string() << "\n";

	MLCouplingData<float> output_data;

    float* output_buffer = new float[out_size];

	// Just to ensure the buffer is changed, we set it to -1 initially
	for (int i = 0; i < out_size; ++i) {
		output_buffer[i] = -1.0f;
	}

	std::vector<int> out_shape = is_mmcp ? std::vector<int>{batch_size, 2, 512} : std::vector<int>{batch_size};
    output_data.add_tensor(MLCouplingTensor<float>::wrap_flat(
		output_buffer,
		out_shape,
		MLCouplingMemLayoutContiguous,
		MLCouplingOwnershipExternal
	));

	std::cout << "Output data before inference:\n";
	std::cout << output_data.to_string() << "\n";

	// *************************************
	// Create coupling object
	// *************************************

	std::string actual_model_path = "";
	if (is_mmcp) {
		actual_model_path = "/rwthfs/rz/cluster/hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/input/transformer_inference_scripted_fw2.pt";
	} else {
		const char* base_dir = std::getenv("BASE_DIR");
		std::string base_dir_str = base_dir ? std::string(base_dir) : "/rwthfs/rz/cluster/hpcwork/ro092286/smartsim";
		actual_model_path = base_dir_str + "/mini_app/train_models/model_a/" + model_name + "_cpu.pt";
	}

	MLCoupling<float, float>* coupling = MLCoupling<float, float>::create_from_config(config_path, std::move(input_data), output_data,
		ConfigCastMode::Strict,
		ConfigParameterMatchMode::Lenient,
		ConfigOverrides{
			{"provider.model_file", actual_model_path}
		});

	if (coupling == nullptr) {
		std::cerr << "Failed to create MLCoupling from config.\n";
		MPI_Finalize();
		return 2;
	}

	const char* merge_strategy_env = std::getenv("MERGE_STRATEGY");
	if (merge_strategy_env != nullptr) {
		std::string strategy(merge_strategy_env);
		if (strategy == "STACK") {
			if (world_rank == 0) std::cout << "Setting merge strategy to STACK\n";
			coupling->set_merge_strategy(MLCouplingMergeStrategy::Stack);
		} else if (strategy == "LIST") {
			if (world_rank == 0) std::cout << "Setting merge strategy to LIST\n";
			coupling->set_merge_strategy(MLCouplingMergeStrategy::List);
		} else if (strategy == "AUTO") {
			if (world_rank == 0) std::cout << "Setting merge strategy to AUTO\n";
			coupling->set_merge_strategy(MLCouplingMergeStrategy::Auto);
		} else if (strategy == "NONE") {
			if (world_rank == 0) std::cout << "Setting merge strategy to NONE\n";
			coupling->set_merge_strategy(MLCouplingMergeStrategy::None);
		}
	}



	// *************************************
	// Perform model calls
	// *************************************

	const char* steps_env = std::getenv("STEPS");
	int num_steps = (steps_env != nullptr) ? std::atoi(steps_env) : 1;
	if (num_steps < 1) num_steps = 1;

	const char* api_mode_env = std::getenv("API_MODE");
	std::string api_mode = (api_mode_env != nullptr) ? api_mode_env : "STATIC";

	if (world_rank == 0) {
		std::cout << "Using API_MODE=" << api_mode << "\n";
	}

	const char* timing_log_env = std::getenv("TIMING_LOG");
	bool enable_timing = (timing_log_env != nullptr && std::string(timing_log_env) != "0");
	std::string timing_file = enable_timing ? std::string(timing_log_env) + "_rank_" + std::to_string(world_rank) + ".csv" : "";
	
	if (enable_timing) {
		std::ofstream ofs(timing_file, std::ios_base::trunc);
		if (ofs.is_open()) {
			ofs << "Rank,Step,StartTime_ns,EndTime_ns,Duration_us\n";
		}
	}

	float* outputs = new float[num_steps];

	for (int step = 0; step < num_steps; ++step) {
		if (num_steps > 1) {
			std::cout << "--- Coupling Step " << step << " ---\n";
		}
		// Increase the input data's values by step number to simulate changing input across steps
		for (size_t i = 0; i < in_size; ++i) {
			flat_data[i] += step;
		}

		auto t_start = std::chrono::high_resolution_clock::now();

		try {
			if (api_mode == "STATIC") {
				coupling->ml_step();
			} else if (api_mode == "ORDERED") {
				MLCouplingData<float> current_input{std::vector<MLCouplingTensor<float>>{
					MLCouplingTensor<float>::wrap_flat(flat_data, single_shape, MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal)
				}};
				coupling->ordered()
					.set(current_input)
					.inference()
					.get(output_data);
			} else if (api_mode == "KEYED") {
				MLCouplingData<float> current_input{std::vector<MLCouplingTensor<float>>{
					MLCouplingTensor<float>::wrap_flat(flat_data, single_shape, MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal)
				}};
				coupling->keyed()
					.set("input_0", current_input)
					.inference({"input_0"}, {"output_0"})
					.get("output_0", output_data);
			} else if (api_mode == "ORDERED_MULTI") {
				int num_inputs = is_mmcp ? 5 : 2;
				std::vector<int> shape = is_mmcp ? std::vector<int>{batch_size, 512} : std::vector<int>{batch_size, 9};
				int offset_step = (is_mmcp ? 512 : 9) * batch_size;
				
				auto proxy = coupling->ordered();
				for (int i = 0; i < num_inputs; ++i) {
					MLCouplingData<float> input_part{std::vector<MLCouplingTensor<float>>{
						MLCouplingTensor<float>::wrap_flat(flat_data + i * offset_step, shape, MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal)
					}};
					proxy.set(input_part);
				}
				proxy.inference().get(output_data);
			} else if (api_mode == "KEYED_MULTI") {
				int num_inputs = is_mmcp ? 5 : 2;
				std::vector<int> shape = is_mmcp ? std::vector<int>{batch_size, 512} : std::vector<int>{batch_size, 9};
				int offset_step = (is_mmcp ? 512 : 9) * batch_size;
				
				auto proxy = coupling->keyed();
				std::vector<std::string> keys;
				for (int i = 0; i < num_inputs; ++i) {
					MLCouplingData<float> input_part{std::vector<MLCouplingTensor<float>>{
						MLCouplingTensor<float>::wrap_flat(flat_data + i * offset_step, shape, MLCouplingMemLayoutContiguous, MLCouplingOwnershipExternal)
					}};
					std::string key = is_mmcp ? "t" + std::to_string(i) : (i == 0 ? "x_water" : "x_terrain");
					proxy.set(key, input_part);
					keys.push_back(key);
				}
				proxy.inference(keys, {"output_0"}).get("output_0", output_data);
			} else {
				throw std::runtime_error("Unknown API_MODE: " + api_mode);
			}

			if (enable_timing) {
				auto t_end = std::chrono::high_resolution_clock::now();
				auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
				std::ofstream ofs(timing_file, std::ios_base::app);
				if (ofs.is_open()) {
					ofs << world_rank << "," << step << ","
					    << std::chrono::duration_cast<std::chrono::nanoseconds>(t_start.time_since_epoch()).count() << ","
					    << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end.time_since_epoch()).count() << ","
					    << duration_us << "\n";
				}
			}

		} catch (const std::exception& e) {
			if (world_rank == 0) std::cerr << "Inference failed at step " << step << " (API_MODE=" << api_mode << "): " << e.what() << "\n";
			delete coupling;
			MPI_Finalize();
			return 3;
		}

		if (world_rank == 0) {
			std::cout << "Inference output: [";
				std::cout << output_buffer[0];
			std::cout << "]\n";
		}
		outputs[step] = output_buffer[0];
	}

	std::cout << "###########################################################################\n";

	std::cout << "All steps completed. Final outputs of rank " << world_rank << ":\n  ";
	for (int step = 0; step < num_steps; ++step) {
		std::cout << outputs[step] << "  ";
	}
	std::cout << "\n";

	// Let's gather the outputs in rank 0 in a 2D array of shape (num_ranks, num_steps) to see the full picture
	if (local_comm != MPI_COMM_NULL) {
		int world_size = 0;
		MPI_Comm_size(local_comm, &world_size);
		std::vector<float> all_outputs(world_size * num_steps);
		MPI_Gather(outputs, num_steps, MPI_FLOAT, all_outputs.data(), num_steps, MPI_FLOAT, 0, local_comm);
		
		if (world_rank == 0) {
			std::cout << "\n###########################################################################\n";
			std::cout << "Gathered outputs from all ranks:\n";
			
			std::cout << "[";
			for (int rank = 0; rank < world_size; ++rank) {
				if (rank > 0) std::cout << ", ";
				std::cout << "[";
				for (int step = 0; step < num_steps; ++step) {
					if (step > 0) std::cout << ", ";
					std::cout << std::fixed << std::setprecision(4) << all_outputs[rank * num_steps + step];
				}
				std::cout << "]";
			}
			std::cout << "]\n";
		}
	}

	delete[] outputs;

	delete coupling;

	delete[] flat_data;
	delete[] output_buffer;
	
	MPI_Finalize();
	return 0;
}
