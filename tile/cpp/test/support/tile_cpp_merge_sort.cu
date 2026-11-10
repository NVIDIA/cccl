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

namespace ct   = cuda::tiles;
namespace coop = cuda::coop::tile;

template <size_t N, size_t Threads>
[[cutile::hint(0, num_worker_warps_per_cta = Threads / 32)]]
__tile_global__ void merge_sort_kernel(
  unsigned* input_keys,
  unsigned long long* input_values,
  unsigned* full_keys,
  unsigned* pair_keys,
  unsigned long long* pair_values,
  unsigned* partial_keys,
  unsigned* partial_pair_keys,
  unsigned long long* partial_pair_values,
  int valid_items)
{
  auto keys                 = ct::partition_view{ct::tensor_span{input_keys, ct::shape<N>{}}, ct::shape<N>{}}.load(0);
  auto values               = ct::partition_view{ct::tensor_span{input_values, ct::shape<N>{}}, ct::shape<N>{}}.load(0);
  auto valid_tile           = ct::reshape(ct::tile{valid_items}, ct::shape<1>{});
  auto sorted               = coop::merge_sort_keys(keys, coop::merge_sort_policy<>{});
  auto [sorted_pair_keys,
        sorted_pair_values] = coop::merge_sort_pairs(keys, values, cuda::std::greater<>{}, coop::merge_sort_policy<>{});
  auto partial              = coop::merge_sort_keys(keys, valid_tile, ~0u, coop::merge_sort_policy<>{});
  auto [partial_sorted_keys, partial_sorted_values] =
    coop::merge_sort_pairs(keys, values, cuda::std::greater<>{}, valid_items, 0u, coop::merge_sort_policy<>{});
  ct::partition_view{ct::tensor_span{full_keys, ct::shape<N>{}}, ct::shape<N>{}}.store(sorted, 0);
  ct::partition_view{ct::tensor_span{pair_keys, ct::shape<N>{}}, ct::shape<N>{}}.store(sorted_pair_keys, 0);
  ct::partition_view{ct::tensor_span{pair_values, ct::shape<N>{}}, ct::shape<N>{}}.store(sorted_pair_values, 0);
  ct::partition_view{ct::tensor_span{partial_keys, ct::shape<N>{}}, ct::shape<N>{}}.store(partial, 0);
  ct::partition_view{ct::tensor_span{partial_pair_keys, ct::shape<N>{}}, ct::shape<N>{}}.store(partial_sorted_keys, 0);
  ct::partition_view{ct::tensor_span{partial_pair_values, ct::shape<N>{}}, ct::shape<N>{}}.store(
    partial_sorted_values, 0);
  ct::partition_view{ct::tensor_span{input_keys, ct::shape<N>{}}, ct::shape<N>{}}.store(keys, 0);
  ct::partition_view{ct::tensor_span{input_values, ct::shape<N>{}}, ct::shape<N>{}}.store(values, 0);
}

bool check_cuda(cudaError_t status)
{
  if (status == cudaSuccess)
  {
    return true;
  }
  std::cerr << cudaGetErrorString(status) << '\n';
  return false;
}

template <size_t N, size_t Threads>
bool run_case()
{
  constexpr int valid_items    = static_cast<int>(3 * N / 4 + 1);
  constexpr size_t key_bytes   = N * sizeof(unsigned);
  constexpr size_t value_bytes = N * sizeof(unsigned long long);
  std::vector<unsigned> keys(N), full_keys(N), pair_keys(N), partial_keys(N), partial_pair_keys(N), preserved_keys(N);
  std::vector<unsigned long long> values(N), pair_values(N), partial_pair_values(N), preserved_values(N);
  for (size_t i = 0; i < N; ++i)
  {
    keys[i]   = i % 2 == 0 ? 0x80000000u + static_cast<unsigned>(i) : static_cast<unsigned>(i);
    values[i] = static_cast<unsigned long long>(i);
  }
  auto expected = keys;
  std::sort(expected.begin(), expected.end());
  std::vector<unsigned> expected_partial(N, ~0u);
  std::copy(keys.begin(), keys.begin() + valid_items, expected_partial.begin());
  std::sort(expected_partial.begin(), expected_partial.begin() + valid_items);
  std::vector<unsigned> expected_partial_pairs(N, 0u);
  std::copy(keys.begin(), keys.begin() + valid_items, expected_partial_pairs.begin());
  std::sort(expected_partial_pairs.begin(), expected_partial_pairs.begin() + valid_items, std::greater<unsigned>{});

  unsigned *device_keys = nullptr, *device_full = nullptr, *device_pair_keys = nullptr;
  unsigned *device_partial = nullptr, *device_partial_pair_keys = nullptr;
  unsigned long long *device_values = nullptr, *device_pair_values = nullptr, *device_partial_pair_values = nullptr;
  if (!check_cuda(cudaMalloc(&device_keys, key_bytes)) || !check_cuda(cudaMalloc(&device_full, key_bytes))
      || !check_cuda(cudaMalloc(&device_pair_keys, key_bytes)) || !check_cuda(cudaMalloc(&device_partial, key_bytes))
      || !check_cuda(cudaMalloc(&device_partial_pair_keys, key_bytes))
      || !check_cuda(cudaMalloc(&device_values, value_bytes))
      || !check_cuda(cudaMalloc(&device_pair_values, value_bytes))
      || !check_cuda(cudaMalloc(&device_partial_pair_values, value_bytes))
      || !check_cuda(cudaMemcpy(device_keys, keys.data(), key_bytes, cudaMemcpyHostToDevice))
      || !check_cuda(cudaMemcpy(device_values, values.data(), value_bytes, cudaMemcpyHostToDevice)))
  {
    return false;
  }
  merge_sort_kernel<N, Threads><<<1, 1>>>(
    device_keys,
    device_values,
    device_full,
    device_pair_keys,
    device_pair_values,
    device_partial,
    device_partial_pair_keys,
    device_partial_pair_values,
    valid_items);
  if (!check_cuda(cudaGetLastError()) || !check_cuda(cudaDeviceSynchronize())
      || !check_cuda(cudaMemcpy(full_keys.data(), device_full, key_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(pair_keys.data(), device_pair_keys, key_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(pair_values.data(), device_pair_values, value_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(partial_keys.data(), device_partial, key_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(partial_pair_keys.data(), device_partial_pair_keys, key_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(
        cudaMemcpy(partial_pair_values.data(), device_partial_pair_values, value_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(preserved_keys.data(), device_keys, key_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(preserved_values.data(), device_values, value_bytes, cudaMemcpyDeviceToHost)))
  {
    return false;
  }
  if (keys != preserved_keys || values != preserved_values)
  {
    std::cerr << "Merge Sort changed its input\n";
    return false;
  }
  for (size_t i = 0; i < N; ++i)
  {
    const auto source = pair_values[i];
    if (full_keys[i] != expected[i] || pair_keys[i] != expected[N - 1 - i] || source >= N
        || pair_keys[i] != keys[static_cast<size_t>(source)] || partial_keys[i] != expected_partial[i]
        || partial_pair_keys[i] != expected_partial_pairs[i])
    {
      std::cerr << "Merge Sort mismatch N=" << N << " threads=" << Threads << " item=" << i << '\n';
      return false;
    }
    if (i < static_cast<size_t>(valid_items))
    {
      const auto partial_source = partial_pair_values[i];
      if (partial_source >= static_cast<unsigned long long>(valid_items)
          || partial_pair_keys[i] != keys[static_cast<size_t>(partial_source)])
      {
        std::cerr << "partial Merge Sort payload mismatch at " << i << '\n';
        return false;
      }
    }
    else if (partial_pair_values[i] != 0)
    {
      std::cerr << "partial Merge Sort padding payload mismatch at " << i << '\n';
      return false;
    }
  }
  if (!check_cuda(cudaFree(device_keys)) || !check_cuda(cudaFree(device_values)) || !check_cuda(cudaFree(device_full))
      || !check_cuda(cudaFree(device_pair_keys)) || !check_cuda(cudaFree(device_pair_values))
      || !check_cuda(cudaFree(device_partial)) || !check_cuda(cudaFree(device_partial_pair_keys))
      || !check_cuda(cudaFree(device_partial_pair_values)))
  {
    return false;
  }
  return true;
}

int main()
{
  if (!run_case<256, 128>() || !run_case<32, 32>())
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: C++ Merge Sort keys/pairs, partial tiles, and warp forms\n";
}
