//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>

using namespace cuda::experimental::stf;

__global__ void dummy() {}

int main(int argc, char** argv)
{
  const int dev_id = 0;
  cuda_safe_call(cudaSetDevice(dev_id));

  cudaDeviceProp prop;
  cuda_safe_call(cudaGetDeviceProperties(&prop, dev_id));

  const size_t total_mem_ref = prop.totalGlobalMem;

  size_t free_mem, total_mem;
  cuda_safe_call(cudaMemGetInfo(&free_mem, &total_mem));

  std::cout << "Device memory: " << total_mem_ref / (1024 * 1024.) << " MB"
            << " FREE/TOTAL=" << free_mem / (1024 * 1024.) << "/" << total_mem / (1024 * 1024.) << std::endl;

  // Warning: this should represent 6% of device's available memory
  size_t block_size = free_mem / 32;
  int nblocks       = 2;

  // We preallocate most available device memory to avoid being limited by host memory.
  void* wasted_mem;
  size_t wasted_size = block_size * 29;
  cuda_safe_call(cudaMalloc(&wasted_mem, wasted_size));

  std::cout << "Wasted: " << wasted_size / (1024 * 1024.) << " MB" << std::endl;

  graph_ctx ctx;

  std::vector<logical_data<slice<char>>> handles(nblocks);

  char* h_buffer = new char[nblocks * block_size];

  for (int i = 0; i < 2; i++)
  {
    handles[i] = ctx.logical_data(make_slice(&h_buffer[i * block_size], block_size));
    handles[i].set_symbol("D_" + std::to_string(i));
  }

  // There can only be a single handle on the device at the same time
  for (int i = 0; i < nblocks; i++)
  {
    ctx.task(handles[i % nblocks].rw())->*[](cudaStream_t stream, auto /*unused*/) {
      dummy<<<1, 1, 0, stream>>>();
    };
  }

  // Checking the amount of memory actually available now
  cuda_safe_call(cudaMemGetInfo(&free_mem, &total_mem));
  std::cout
    << "Device memory: FREE/TOTAL=" << free_mem / (1024 * 1024.) << "/" << total_mem / (1024 * 1024.) << std::endl;

  ctx.submit();

  if (argc > 1)
  {
    std::cout << "Generating DOT output in " << argv[1] << std::endl;
    ctx.print_to_dot(argv[1]);
  }

  ctx.finalize();
}
