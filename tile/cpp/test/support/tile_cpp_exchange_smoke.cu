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
using value_type              = unsigned;
constexpr size_t input_count  = 512;
constexpr size_t output_count = input_count * 4;

__tile__ void store_values(ct::tile<value_type, ct::shape<input_count>> values, value_type* output)
{
  ct::partition_view{ct::tensor_span{output, ct::shape<input_count>{}}, ct::shape<input_count>{}}.store(values, 0);
}

template <size_t BlockThreads>
[[cutile::hint(0, num_worker_warps_per_cta = BlockThreads / 32)]]
__tile_global__ void family_kernel(const value_type* input, value_type* output)
{
  const auto values =
    ct::partition_view{ct::tensor_span{input, ct::shape<input_count>{}}, ct::shape<input_count>{}}.load(0);
  const auto blocked   = coop::exchange_striped_to_blocked(values, coop::exchange_policy<BlockThreads>{});
  const auto striped   = coop::exchange_blocked_to_striped(values, coop::exchange_policy<BlockThreads>{});
  const auto roundtrip = coop::exchange_blocked_to_striped(blocked, coop::exchange_policy<BlockThreads>{});
  store_values(blocked, output);
  store_values(striped, output + input_count);
  store_values(roundtrip, output + input_count * 2);
  store_values(values, output + input_count * 3);
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

template <size_t BlockThreads>
bool run_case()
{
  std::vector<value_type> input(input_count), output(output_count), expected(output_count);
  for (size_t index = 0; index < input_count; ++index)
  {
    input[index] = static_cast<value_type>(index * 7 + 3);
  }
  constexpr size_t items_per_thread = input_count / BlockThreads;
  for (size_t thread = 0; thread < BlockThreads; ++thread)
  {
    for (size_t item = 0; item < items_per_thread; ++item)
    {
      const size_t blocked            = thread * items_per_thread + item;
      const size_t striped            = item * BlockThreads + thread;
      expected[blocked]               = input[striped];
      expected[input_count + striped] = input[blocked];
    }
  }
  for (size_t index = 0; index < input_count; ++index)
  {
    expected[input_count * 2 + index] = input[index];
    expected[input_count * 3 + index] = input[index];
  }

  value_type* device_input      = nullptr;
  value_type* device_output     = nullptr;
  constexpr size_t input_bytes  = input_count * sizeof(value_type);
  constexpr size_t output_bytes = output_count * sizeof(value_type);
  if (!check(cudaMalloc(&device_input, input_bytes)) || !check(cudaMalloc(&device_output, output_bytes))
      || !check(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice)))
  {
    return false;
  }
  family_kernel<BlockThreads><<<1, 1>>>(device_input, device_output);
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost)))
  {
    return false;
  }
  if (output != expected)
  {
    std::cerr
      << "exchange direction, roundtrip, or logical input preservation mismatch at width " << BlockThreads << '\n';
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
  std::cout << "PASS: exchange\n";
}
