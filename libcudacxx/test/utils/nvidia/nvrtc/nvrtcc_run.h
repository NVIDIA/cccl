//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda.h>
#include <cuda_runtime.h>

#include "nvrtcc_common.h"
#include <nvrtc.h>

struct ExecutionConfig
{
  RunConfig rc;
  std::vector<std::string> builds;
};

static ExecutionConfig load_execution_config_from_file(const std::string& file)
{
  std::vector<std::string> builds;
  auto config = load_input_file(file);
  std::regex config_regex("^ *- *'(.*gpu)'$");

  fprintf(stderr, "Builds found: \r\n");

  size_t line_begin = 0;
  size_t line_end   = config.find('\n');
  while (line_end != std::string::npos)
  {
    // Match any line with a .gpu file
    // std::regex cannot handle multiline, so we need to make sure that's not included
    std::string line(config.begin() + line_begin, config.begin() + line_end);
    std::smatch match;
    std::regex_match(line, match, config_regex);

    if (match.size())
    {
      builds.emplace_back(match[1].str());
    }

    line_begin = line_end + 1;
    line_end   = config.find('\n', line_begin);
  }

  return {parse_run_config(config), builds};
}

static void load_and_run_gpu_code(const std::string inputFile, const RunConfig& rc)
{
  std::ifstream istr(inputFile, std::ios::binary);
  assert(!!istr);

  std::vector<char> code(std::istreambuf_iterator<char>{istr}, std::istreambuf_iterator<char>{});
  istr.close();

  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;

  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, code.data(), 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "main_kernel"));
  CUDA_SAFE_CALL(cuLaunchKernel(kernel, 1, 1, 1, rc.threadCount, 1, 1, rc.shmemSize, nullptr, nullptr, 0));

  CUDA_API_CALL(cudaGetLastError());
  CUDA_API_CALL(cudaDeviceSynchronize());

  return;
}
