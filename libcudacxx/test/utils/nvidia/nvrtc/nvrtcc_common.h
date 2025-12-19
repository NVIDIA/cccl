//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

#define NVRTC_SAFE_CALL(x)                                                                 \
  do                                                                                       \
  {                                                                                        \
    nvrtcResult result = x;                                                                \
    if (result != NVRTC_SUCCESS)                                                           \
    {                                                                                      \
      printf("\nNVRTC ERROR: %s failed with error %s\n", #x, nvrtcGetErrorString(result)); \
      exit(1);                                                                             \
    }                                                                                      \
  } while (0)

#define CUDA_SAFE_CALL(x)                                         \
  do                                                              \
  {                                                               \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS)                                   \
    {                                                             \
      const char* msg;                                            \
      cuGetErrorName(result, &msg);                               \
      printf("\nCUDA ERROR: %s failed with error %s\n", #x, msg); \
      exit(1);                                                    \
    }                                                             \
  } while (0)

#define CUDA_API_CALL(x)                                                                \
  do                                                                                    \
  {                                                                                     \
    cudaError_t err = x;                                                                \
    if (err != cudaSuccess)                                                             \
    {                                                                                   \
      printf("\nCUDA ERROR: %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

static void write_output_file(const char* data, size_t datasz, const std::string& file)
{
  std::ofstream ostr(file, std::ios::binary);
  assert(!!ostr);

  ostr.write(data, datasz);
  ostr.close();
}

static std::string load_input_file(const std::string& file)
{
  if (file == "-")
  {
    return std::string(std::istream_iterator<char>{std::cin}, std::istream_iterator<char>{});
  }
  else
  {
    std::ifstream istr(file);
    assert(!!istr);
    return std::string(std::istreambuf_iterator<char>{istr}, std::istreambuf_iterator<char>{});
  }
}

static int parse_int_assignment(const std::string& input, std::string var, int def)
{
  auto lineBegin = input.find(var);
  auto lineEnd   = input.find('\n', lineBegin);

  if (lineBegin == std::string::npos || lineEnd == std::string::npos)
  {
    return def;
  }

  std::string line(input.begin() + lineBegin, input.begin() + lineEnd);
  std::regex varRegex("^" + var + ".*?([0-9]+).*?$");
  std::smatch match;
  std::regex_match(line, match, varRegex);

  if (match.size())
  {
    return std::stoi(match[1].str(), nullptr);
  }

  fprintf(stderr, "ERROR: Could not find an integer literal for '%s' on line '%s':\r\n", var.c_str(), line.c_str());
  exit(1);

  return def;
}

struct RunConfig
{
  int threadCount = 1;
  int shmemSize   = 0;
};

static RunConfig parse_run_config(const std::string& input)
{
  return RunConfig{
    parse_int_assignment(input, "cuda_thread_count", 1),
    parse_int_assignment(input, "cuda_block_shmem_size", 0),
  };
}

// Fake main for adapting kernels
static const char* program = R"program(
__host__ __device__ int fake_main(int argc, char ** argv);
#define main fake_main

// extern "C" to stop the name from being mangled
extern "C" __global__ void main_kernel() {
    fake_main(0, nullptr);
}
)program";
