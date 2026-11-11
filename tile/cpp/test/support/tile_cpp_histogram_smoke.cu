//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/coop/tile.cuh>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct                  = cuda::tiles;
namespace coop                = cuda::coop::tile;
using input_type              = unsigned char;
using output_type             = int;
constexpr size_t input_count  = 256;
constexpr size_t output_count = 128;

__tile_global__ void family_kernel(const input_type* input, output_type* output, input_type* preserved)
{
  auto values = ct::partition_view{ct::tensor_span{input, ct::shape<input_count>{}}, ct::shape<input_count>{}}.load(0);
  auto result = coop::histogram<128>(values);
  ct::partition_view{ct::tensor_span{output, ct::shape<output_count>{}}, ct::shape<output_count>{}}.store(result, 0);
  ct::partition_view{ct::tensor_span{preserved, ct::shape<input_count>{}}, ct::shape<input_count>{}}.store(values, 0);
}

bool check(cudaError_t status)
{
  if (status == cudaSuccess)
  {
    return true;
  }
  std::cerr << cudaGetErrorString(status) << '\n';
  return false;
}

int main()
{
  std::vector<input_type> input(input_count), preserved(input_count);
  std::vector<output_type> output(output_count), expected(output_count);
  for (size_t index = 0; index < input_count; ++index)
  {
    input[index] = static_cast<unsigned char>(index % 128);
  }
  input_type* device_input      = nullptr;
  input_type* device_preserved  = nullptr;
  output_type* device_output    = nullptr;
  constexpr size_t input_bytes  = input_count * sizeof(input_type);
  constexpr size_t output_bytes = output_count * sizeof(output_type);
  if (!check(cudaMalloc(&device_input, input_bytes)) || !check(cudaMalloc(&device_output, output_bytes))
      || !check(cudaMalloc(&device_preserved, input_bytes))
      || !check(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice)))
  {
    return EXIT_FAILURE;
  }
  family_kernel<<<1, 1>>>(device_input, device_output, device_preserved);
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost))
      || !check(cudaMemcpy(preserved.data(), device_preserved, input_bytes, cudaMemcpyDeviceToHost)))
  {
    return EXIT_FAILURE;
  }
  for (auto value : input)
  {
    ++expected[value];
  }
  if (input != preserved || output != expected)
  {
    std::cerr << "histogram result or input preservation mismatch\n";
    return EXIT_FAILURE;
  }
  if (!check(cudaFree(device_input)) || !check(cudaFree(device_output)) || !check(cudaFree(device_preserved)))
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: histogram\n";
}
