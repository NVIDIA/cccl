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

namespace ct   = cuda::tiles;
namespace coop = cuda::coop::tile;

template <size_t N>
__tile__ void store_values(ct::tile<int, ct::shape<N>> values, int* output)
{
  ct::partition_view{ct::tensor_span{output, ct::shape<N>{}}, ct::shape<N>{}}.store(values, 0);
}

template <size_t N, size_t Threads>
[[cutile::hint(0, num_worker_warps_per_cta = Threads / 32)]]
__tile_global__ void adjacent_shuffle_kernel(const int* input, int* output, int offset_distance, int rotate_distance)
{
  const auto values = ct::partition_view{ct::tensor_span{input, ct::shape<N>{}}, ct::shape<N>{}}.load(0);
  const auto left   = coop::adjacent_difference_subtract_left(values);
  const auto right  = coop::adjacent_difference_subtract_right(values, coop::adjacent_difference_policy<>{});
  const auto up     = coop::shuffle_up(values);
  const auto down   = coop::shuffle_down(values);
  const auto sum    = coop::sum(values);
  store_values(left, output);
  store_values(right, output + N);
  store_values(up, output + 2 * N);
  store_values(down, output + 3 * N);
  store_values(sum, output + 4 * N);
  if constexpr (N == Threads)
  {
    const auto rotate_tile = ct::reshape(ct::tile{rotate_distance}, ct::shape<1>{});
    const auto offset      = coop::shuffle_offset(values, offset_distance);
    const auto rotate      = coop::shuffle_rotate(values, rotate_tile);
    store_values(offset, output + 6 * N);
    store_values(rotate, output + 7 * N);
  }
  store_values(values, output + 5 * N);
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

template <size_t N, size_t Threads>
bool run_case(int offset_distance)
{
  constexpr int rotate_distance = 3;
  constexpr size_t output_count = N * (N == Threads ? 8 : 6);
  std::vector<int> input(N), output(output_count), expected(output_count);
  int total = 0;
  for (size_t i = 0; i < N; ++i)
  {
    input[i] = static_cast<int>(3 * i + i % 7);
    total += input[i];
  }
  for (size_t i = 0; i < N; ++i)
  {
    expected[i]         = i == 0 ? input[i] : input[i] - input[i - 1];
    expected[N + i]     = i + 1 == N ? input[i] : input[i] - input[i + 1];
    expected[2 * N + i] = i == 0 ? input[i] : input[i - 1];
    expected[3 * N + i] = i + 1 == N ? input[i] : input[i + 1];
    expected[4 * N + i] = total;
    expected[5 * N + i] = input[i];
    if constexpr (N == Threads)
    {
      const auto source   = static_cast<int>(i) + offset_distance;
      expected[6 * N + i] = source >= 0 && source < static_cast<int>(N) ? input[source] : input[i];
      expected[7 * N + i] = input[(i + rotate_distance) % N];
    }
  }
  int* device_input  = nullptr;
  int* device_output = nullptr;
  if (!check(cudaMalloc(&device_input, N * sizeof(int)))
      || !check(cudaMalloc(&device_output, output_count * sizeof(int)))
      || !check(cudaMemcpy(device_input, input.data(), N * sizeof(int), cudaMemcpyHostToDevice)))
  {
    return false;
  }
  adjacent_shuffle_kernel<N, Threads><<<1, 1>>>(device_input, device_output, offset_distance, rotate_distance);
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, output_count * sizeof(int), cudaMemcpyDeviceToHost)))
  {
    return false;
  }
  if (output != expected)
  {
    std::cerr
      << "Adjacent/Shuffle result or logical input preservation mismatch N=" << N << " threads=" << Threads << '\n';
    return false;
  }
  return check(cudaFree(device_input)) && check(cudaFree(device_output));
}

int main()
{
  if (!run_case<128, 128>(2) || !run_case<128, 128>(-2) || !run_case<256, 128>(0))
  {
    return EXIT_FAILURE;
  }
  std::cout
    << "PASS: C++ Tile API AdjacentDifference/Shuffle/Reduce and logical preservation at one/two items per worker\n";
}
