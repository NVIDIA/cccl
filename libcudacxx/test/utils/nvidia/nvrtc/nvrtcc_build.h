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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "nvrtcc_common.h"
#include <nvrtc.h>
#include <stdio.h>

// Arch configs are strings and bools determining architecture and ptx/sass compilation
using ArchConfig          = std::tuple<std::string, bool>;
constexpr auto archString = [](const ArchConfig& a) -> const auto& {
  return std::get<0>(a);
};
constexpr auto isArchReal = [](const ArchConfig& a) -> const auto& {
  return std::get<1>(a);
};

using ArgList = std::vector<std::string>;

using GpuProg = std::vector<char>;

// Takes arguments for building a file and returns the path to the output file
GpuProg nvrtc_build_prog(const std::string& testCu, const ArchConfig& config, const ArgList& argList)
{
  // Assemble arguments
  std::vector<const char*> optList;

  // Be careful with lifetimes here
  std::for_each(argList.begin(), argList.end(), [&](const auto& it) {
    optList.emplace_back(it.c_str());
  });

  // Use the translated architecture
  std::string gpu_arch("--gpu-architecture=" + archString(config));
  optList.emplace_back(gpu_arch.c_str());

  fprintf(stderr, "NVRTC opt list:\r\n");
  for (const auto& it : optList)
  {
    fprintf(stderr, "  %s\r\n", it);
  }

  fprintf(stderr, "Compiling program...\r\n");
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, testCu.c_str(), "test.cu", 0, NULL, NULL));

  nvrtcResult compile_result = nvrtcCompileProgram(prog, optList.size(), optList.data());

  fprintf(stderr, "Collecting logs...\r\n");
  size_t log_size;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &log_size));

  {
    std::unique_ptr<char[]> log{new char[log_size]};
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
    printf("%s\r\n", log.get());
  }

  if (compile_result != NVRTC_SUCCESS)
  {
    exit(1);
  }

  size_t codeSize;
  GpuProg code;

  if (isArchReal(config))
  {
    NVRTC_SAFE_CALL(nvrtcGetCUBINSize(prog, &codeSize));
    code.resize(codeSize);
    NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code.data()));
  }
  else
  {
    NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &codeSize));
    code.resize(codeSize);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, code.data()));
  }
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  return code;
}
