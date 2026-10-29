// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Single-call CUB warp reduction used by check_warp_reduce.py to compare
// cuda.coop results with CUB on identical inputs.
//
// Usage: check_warp_reduce <T> <sum|reduce_plus> <input.bin> <output.bin> <num_items>
// Each warp of 256-thread blocks reduces 32 consecutive inputs; lane 0 writes
// the warp's result to output[warp].

#include <cuda_fp16.h>

#include <cub/warp/warp_reduce.cuh>

#include <cuda/std/functional>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

constexpr int block_size = 256;

template <typename T, bool UseSum>
__global__ void __launch_bounds__(block_size) warp_reduce_kernel(const T* input, T* output)
{
  using WarpReduce = cub::WarpReduce<T>;
  __shared__ typename WarpReduce::TempStorage temp_storage[block_size / 32];
  const int tid     = blockIdx.x * block_size + threadIdx.x;
  const int warp_id = threadIdx.x / 32;
  T result;
  if constexpr (UseSum)
  {
    result = WarpReduce{temp_storage[warp_id]}.Sum(input[tid]);
  }
  else
  {
    result = WarpReduce{temp_storage[warp_id]}.Reduce(input[tid], cuda::std::plus<>{});
  }
  if (threadIdx.x % 32 == 0)
  {
    output[tid / 32] = result;
  }
}

static void check(cudaError_t err)
{
  if (err != cudaSuccess)
  {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(2);
  }
}

template <typename T>
static int run(const std::string& op, const char* input_path, const char* output_path, int num_items)
{
  if (num_items % block_size != 0)
  {
    std::fprintf(stderr, "num_items must be a multiple of %d\n", block_size);
    return 1;
  }
  std::vector<T> input(num_items);
  std::ifstream(input_path, std::ios::binary).read(reinterpret_cast<char*>(input.data()), num_items * sizeof(T));
  const int num_warps = num_items / 32;
  std::vector<T> output(num_warps);

  T* d_input  = nullptr;
  T* d_output = nullptr;
  check(cudaMalloc(&d_input, num_items * sizeof(T)));
  check(cudaMalloc(&d_output, num_warps * sizeof(T)));
  check(cudaMemcpy(d_input, input.data(), num_items * sizeof(T), cudaMemcpyHostToDevice));
  if (op == "sum")
  {
    warp_reduce_kernel<T, true><<<num_items / block_size, block_size>>>(d_input, d_output);
  }
  else if (op == "reduce_plus")
  {
    warp_reduce_kernel<T, false><<<num_items / block_size, block_size>>>(d_input, d_output);
  }
  else
  {
    std::fprintf(stderr, "unknown op %s\n", op.c_str());
    return 1;
  }
  check(cudaGetLastError());
  check(cudaMemcpy(output.data(), d_output, num_warps * sizeof(T), cudaMemcpyDeviceToHost));
  check(cudaFree(d_input));
  check(cudaFree(d_output));
  std::ofstream(output_path, std::ios::binary)
    .write(reinterpret_cast<const char*>(output.data()), num_warps * sizeof(T));
  return 0;
}

int main(int argc, char** argv)
{
  if (argc != 6)
  {
    std::fprintf(stderr, "usage: %s <T> <sum|reduce_plus> <input.bin> <output.bin> <num_items>\n", argv[0]);
    return 1;
  }
  const std::string type = argv[1];
  const std::string op   = argv[2];
  const int num_items    = std::atoi(argv[5]);
  if (type == "I8")
  {
    return run<int8_t>(op, argv[3], argv[4], num_items);
  }
  if (type == "I16")
  {
    return run<int16_t>(op, argv[3], argv[4], num_items);
  }
  if (type == "I32")
  {
    return run<int32_t>(op, argv[3], argv[4], num_items);
  }
  if (type == "I64")
  {
    return run<int64_t>(op, argv[3], argv[4], num_items);
  }
  if (type == "F16")
  {
    return run<__half>(op, argv[3], argv[4], num_items);
  }
  if (type == "F32")
  {
    return run<float>(op, argv[3], argv[4], num_items);
  }
  if (type == "F64")
  {
    return run<double>(op, argv[3], argv[4], num_items);
  }
  std::fprintf(stderr, "unknown type %s\n", type.c_str());
  return 1;
}
