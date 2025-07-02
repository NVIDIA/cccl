/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
This is a simple example demonstrating how to use CCCL with NVRTC.
*/

#if !defined(EXTERNAL_NVRTC_ARGS) || !defined(EXTERNAL_NVRTC_ARCH)
#  error \
    "EXTERNAL_NVRTC_ARGS or EXTERNAL_NVRTC_ARCH is not defined. Please define it to add externally defined NVRTC arguments."
#endif

#define STR2(x)    #x
#define STR(x)     STR2(x)
#define NVRTC_ARCH "--gpu-architecture=" STR(EXTERNAL_NVRTC_ARCH)

#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>

#include <cuda.h>
#include <nvrtc.h>

#include "helpers.h"

using gpu_code_ptr = std::unique_ptr<char[]>;

void load_and_execute_gpu_code(const void* imageData);
gpu_code_ptr compile_gpu_code(std::string_view kernel, std::span<const char*> optList);

#define BLOCK_SIZE 256

const char* kernel = "#define BLOCK_SIZE " STR(BLOCK_SIZE) R"X(

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>

extern "C" __global__ void sumKernel(int const* data, int* result, size_t N)
{
  using BlockReduce = cub::BlockReduce<int, BLOCK_SIZE>;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  int index = threadIdx.x + blockIdx.x * blockDim.x;

  int sum = 0;
  if (index < N)
  {
    sum += data[index];
  }

  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0)
  {
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(*result);
    atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
  }
}
)X";

int main()
{
  const char* nvrtcOptions[] = {"--std=c++20", NVRTC_ARCH, EXTERNAL_NVRTC_ARGS};

  auto gpuCode = compile_gpu_code(kernel, nvrtcOptions);
  load_and_execute_gpu_code(gpuCode.get());
  return 0;
}

gpu_code_ptr compile_gpu_code(std::string_view kernel, std::span<const char*> optList)
{
  nvrtcProgram prog;
  // nvrtcCreateProgram allows headers to be additionally added, skip this and use `-I` instead later.
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel.data(), "example.cu", 0, nullptr, nullptr));

  printf("Compiling with options: \n");
  for (auto opt : optList)
  {
    printf(" %s\n", opt);
  }

  size_t logSize;
  nvrtcResult compile_result = nvrtcCompileProgram(prog, optList.size(), optList.data());
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));

  // ALWAYS GATHER LOGS, there could be warnings or non-fatal issues that should be reported.
  {
    auto log = std::make_unique<char[]>(logSize};
    NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.get()));
    printf("%s\r\n", log.get());
  }

  if (compile_result != NVRTC_SUCCESS)
  {
    // We've gathered logs now exit
    exit(1);
  }

  gpu_code_ptr code{};

  // Prioritize SASS, then PTX. If neither is available, fail
  size_t cubinSize = 0;
  size_t ptxSize   = 0;
  bool sass        = nvrtcGetCUBINSize(prog, &cubinSize) == NVRTC_SUCCESS;
  bool ptx         = nvrtcGetPTXSize(prog, &ptxSize) == NVRTC_SUCCESS;

  if (sass && cubinSize > 0)
  {
    printf("Found SASS (code size %zu bytes).\n", cubinSize);
    code.reset(new char[cubinSize]);
    NVRTC_SAFE_CALL(nvrtcGetCUBIN(prog, code.get()));
  }
  else if (ptx && ptxSize > 0)
  {
    printf("Found PTX (code size %zu bytes).\n", ptxSize);
    code.reset(new char[ptxSize]);
    NVRTC_SAFE_CALL(nvrtcGetPTX(prog, code.get()));
  }
  else
  {
    printf("Error: No CUBIN or PTX code available for the program.\n");
    exit(1);
  }
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));
  return code;
}

void load_and_execute_gpu_code(const void* imageData)
{
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;

  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuDevicePrimaryCtxRetain(&context, cuDevice));
  CUDA_SAFE_CALL(cuCtxSetCurrent(context));

  size_t N       = 1000;
  int* data      = nullptr;
  int* result    = nullptr;
  int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

  CUDA_SAFE_CALL(cuMemHostAlloc((void**) &data, N * sizeof(int), 0));
  CUDA_SAFE_CALL(cuMemHostAlloc((void**) &result, sizeof(int), 0));

  std::fill(data, data + N, 1);

  void* kernelParams[] = {(void*) &data, (void*) &result, (void*) &N};

  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, imageData, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "sumKernel"));
  CUDA_SAFE_CALL(cuLaunchKernel(kernel, num_blocks, 1, 1, BLOCK_SIZE, 1, 1, 0, nullptr, kernelParams, nullptr));
  CUDA_SAFE_CALL(cuCtxSynchronize());

  printf("Sum is %i\n", result[0]);
  assert(result[0] == N);

  return;
}
