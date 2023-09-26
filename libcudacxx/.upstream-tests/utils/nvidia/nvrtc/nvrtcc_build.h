//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "nvrtcc_common.h"

#include <nvrtc.h>
#include <cuda.h>

#include <stdio.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using ArchConfig = std::tuple<std::string, bool>;
constexpr auto archString = [](const ArchConfig& a) {return std::get<0>(a);};
constexpr auto isArchReal = [](const ArchConfig& a) {return std::get<1>(a);};

using ArgList = std::vector<std::string>;

const char * program = R"program(
__host__ __device__ int fake_main(int argc, char ** argv);
#define main fake_main

// extern "C" to stop the name from being mangled
extern "C" __global__ void main_kernel() {
    fake_main(0, NULL);
}
)program";

// Takes arguments for building a file and returns the path to the output file
std::string nvrtc_build_prog(const std::string& input_file, const std::string& output_template, const ArchConfig& config, const ArgList& argList) {
    std::ifstream istr(input_file);
    std::string test_cu(
        std::istreambuf_iterator<char>{istr},
        std::istreambuf_iterator<char>{} );

    // Prepend fakemain
    test_cu = program + test_cu;

    // Assemble arguments
    std::vector<const char*> optList;

    // Be careful with lifetimes here
    std::for_each(argList.begin(), argList.end(), [&](const auto& it){
        optList.emplace_back(it.c_str());
    });

    // Use the translated architecture
    std::string gpu_arch("--gpu-architecture=" + archString(config));
    optList.emplace_back(gpu_arch.c_str());

    printf("NVRTC opt list:\r\n");
    for (const auto& it: optList) {
        printf("  %s\r\n", it);
    }

    printf ("Compiling program...\r\n");
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(
        &prog,
        test_cu.c_str(),
        "test.cu",
        0, NULL, NULL));

    nvrtcResult compile_result = nvrtcCompileProgram(
        prog,
        optList.size(),
        optList.data());

    printf ("Collecting logs...\r\n");
    size_t log_size;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));

    {
        std::unique_ptr<char[]> log{ new char[log_size] };
        NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
        printf("%s\r\n", log.get());
    }

    if (compile_result != NVRTC_SUCCESS) {
        exit(1);
    }

    size_t codeSize;

    std::unique_ptr<char[]> code{nullptr};
    if (isArchReal(config)) {
        NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));
        code = std::unique_ptr<char[]>{new char[codeSize]};
        NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code.get()));
    }
    else {
        NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &codeSize));
        code = std::unique_ptr<char[]>{new char[codeSize]};
        NVRTC_SAFE_CALL(nvrtcGetPTX(prog, code.get()));
    }
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    std::string output_file = output_template + "." + archString(config) + ".gpu";
    printf("Writing output to: %s\r\n", output_file.c_str());

    std::ofstream ostr(output_file, std::ios::binary);
    ostr.write(code.get(), codeSize);
    ostr.close();

    return output_file;
}
