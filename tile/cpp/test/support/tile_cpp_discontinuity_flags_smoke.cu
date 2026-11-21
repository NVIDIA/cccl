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

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct                  = cuda::tiles;
namespace coop                = cuda::coop::tile;
constexpr size_t input_count  = 256;
constexpr size_t output_count = 5 * input_count;

__tile__ void store_values(ct::tile<int, ct::shape<input_count>> values, int* output)
{
  ct::partition_view{ct::tensor_span{output, ct::shape<input_count>{}}, ct::shape<input_count>{}}.store(values, 0);
}

template <size_t Threads>
[[cutile::hint(0, num_worker_warps_per_cta = Threads / 32)]]
__tile_global__ void family_kernel(const int* input, int* output)
{
  const auto values =
    ct::partition_view{ct::tensor_span{input, ct::shape<input_count>{}}, ct::shape<input_count>{}}.load(0);
  const auto heads                        = coop::discontinuity_flag_heads(values);
  const auto tails                        = coop::discontinuity_flag_tails(values);
  const auto [paired_heads, paired_tails] = coop::discontinuity_flag_heads_and_tails(values);
  store_values(heads, output);
  store_values(tails, output + input_count);
  store_values(paired_heads, output + 2 * input_count);
  store_values(paired_tails, output + 3 * input_count);
  store_values(values, output + 4 * input_count);
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

template <size_t Threads>
bool run_case()
{
  std::vector<int> input(input_count), output(output_count), expected(output_count);
  for (size_t index = 0; index < input_count; ++index)
  {
    input[index] = static_cast<int>(index / 5);
  }
  for (size_t index = 0; index < input_count; ++index)
  {
    const int head  = index == 0 || input[index] != input[index - 1];
    const int tail  = index + 1 == input_count || input[index] != input[index + 1];
    expected[index] = expected[2 * input_count + index] = head;
    expected[input_count + index] = expected[3 * input_count + index] = tail;
    expected[4 * input_count + index]                                 = input[index];
  }
  int* device_input             = nullptr;
  int* device_output            = nullptr;
  constexpr size_t input_bytes  = input_count * sizeof(int);
  constexpr size_t output_bytes = output_count * sizeof(int);
  if (!check(cudaMalloc(&device_input, input_bytes)) || !check(cudaMalloc(&device_output, output_bytes))
      || !check(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice)))
  {
    return false;
  }
  family_kernel<Threads><<<1, 1>>>(device_input, device_output);
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost)))
  {
    return false;
  }
  if (output != expected)
  {
    std::cerr << "discontinuity heads/tails/pair or logical input preservation mismatch, threads=" << Threads << '\n';
    return false;
  }
  return check(cudaFree(device_input)) && check(cudaFree(device_output));
}

int main()
{
  if (!run_case<128>() || !run_case<256>())
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: discontinuity heads/tails/pair and logical input preservation at one/two items per worker\n";
}
