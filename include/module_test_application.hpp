#pragma once

#include "application/ml_coupling_application.hpp"

// @registry_name: ModuleTestApplication
// @registry_aliases: module-test-app, module_test_app
template <typename CouplingInput,
          typename CouplingOutput,
          typename LibraryInput = CouplingInput,
          typename LibraryOutput = CouplingOutput>
class ModuleTestApplication : public MLCouplingApplication<CouplingInput,
                                                            CouplingOutput,
                                                            LibraryInput,
                                                            LibraryOutput>
{
public:
    ModuleTestApplication(MLCouplingData<CouplingInput> input_data,
                          MLCouplingData<CouplingOutput> output_data,
                          MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization)
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(
              std::move(input_data), std::move(output_data), normalization) {}

    ModuleTestApplication(MLCouplingData<CouplingInput> coupling_input,
                          MLCouplingData<LibraryInput> library_input,
                          MLCouplingData<LibraryOutput> library_output,
                          MLCouplingData<CouplingOutput> coupling_output,
                          MLCouplingNormalization<LibraryInput, CouplingOutput>* normalization)
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(
              std::move(coupling_input),
              std::move(library_input),
              std::move(library_output),
              std::move(coupling_output),
              normalization) {}

protected:
    MLCouplingData<LibraryInput>
    preprocess_coupling_input(MLCouplingData<CouplingInput> input_data) override
    {
        return input_data;
    }

    MLCouplingData<CouplingOutput>
    postprocess_library_output(MLCouplingData<LibraryOutput> output_data_before_postprocessing) override
    {
        return output_data_before_postprocessing;
    }
};
