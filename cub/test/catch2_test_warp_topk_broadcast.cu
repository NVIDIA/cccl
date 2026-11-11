// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/warp/warp_topk_broadcast.cuh>

#include <thrust/memory.h>

#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <functional>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/vector.h>

// Independent logical groups exercise input preservation and all-lane result ownership.
template <typename KeyT, int ItemsPerThread, int K, int LogicalWarpThreads>
__global__ void broadcast_topk_kernel(const KeyT* input, KeyT* output, KeyT sentinel)
{
  const int thread = static_cast<int>(threadIdx.x);
  KeyT keys[ItemsPerThread];
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    keys[i] = input[thread * ItemsPerThread + i];
  }
  KeyT selected[K];
  cub::detail::WarpTopKBroadcast<KeyT, K, LogicalWarpThreads>::TopK(keys, selected, sentinel);
  for (int i = 0; i < K; ++i)
  {
    output[thread * (K + ItemsPerThread) + i] = selected[i];
  }
  for (int i = 0; i < ItemsPerThread; ++i)
  {
    output[thread * (K + ItemsPerThread) + K + i] = keys[i];
  }
}

template <typename KeyT, int ItemsPerThread, int K, int LogicalWarpThreads>
void check_broadcast_topk(int distribution)
{
  using key_type                     = KeyT;
  constexpr int items                = ItemsPerThread;
  constexpr int k                    = K;
  constexpr int logical_warp_threads = LogicalWarpThreads;
  constexpr int threads              = 128;
  constexpr int row_size             = logical_warp_threads * items;
  c2h::host_vector<key_type> input(threads * items);
  for (int i = 0; i < threads * items; ++i)
  {
    // Ties and independent per-warp ranges, plus both monotone orders.
    const int value = distribution == 0 ? (i * 23 % 17 - 8) : (distribution == 1 ? i : -i);
    input[i]        = static_cast<key_type>(value);
    if (distribution == 3)
    {
      if constexpr (cuda::std::is_same_v<key_type, float>)
      {
        const float special[] = {
          -cuda::std::numeric_limits<float>::infinity(), cuda::std::numeric_limits<float>::infinity(), -0.0f, 0.0f};
        input[i] = special[i % 4];
      }
      else
      {
        input[i] =
          i % 2 == 0 ? cuda::std::numeric_limits<key_type>::lowest() : cuda::std::numeric_limits<key_type>::max();
      }
    }
  }
  c2h::device_vector<key_type> device_input = input;
  c2h::device_vector<key_type> device_output(threads * (items + k));
  const key_type sentinel =
    cuda::std::numeric_limits<key_type>::has_infinity
      ? -cuda::std::numeric_limits<key_type>::infinity()
      : cuda::std::numeric_limits<key_type>::lowest();
  broadcast_topk_kernel<key_type, items, k, logical_warp_threads><<<1, threads>>>(
    thrust::raw_pointer_cast(device_input.data()), thrust::raw_pointer_cast(device_output.data()), sentinel);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  const c2h::host_vector<key_type> actual = device_output;
  c2h::host_vector<key_type> sorted       = input;
  for (int warp = 0; warp < threads / logical_warp_threads; ++warp)
  {
    std::sort(sorted.begin() + warp * row_size, sorted.begin() + (warp + 1) * row_size, std::greater<key_type>{});
    for (int lane = 0; lane < logical_warp_threads; ++lane)
    {
      const int thread = warp * logical_warp_threads + lane;
      for (int i = 0; i < k; ++i)
      {
        REQUIRE(actual[thread * (items + k) + i] == sorted[warp * row_size + i]);
      }
      for (int i = 0; i < items; ++i)
      {
        REQUIRE(actual[thread * (items + k) + k + i] == input[thread * items + i]);
      }
    }
  }
}

using key_types          = c2h::type_list<float, std::int64_t>;
using item_counts        = c2h::enum_type_list<int, 1, 4, 16, 64>;
using k_values           = c2h::enum_type_list<int, 1, 4, 32>;
using logical_warp_sizes = c2h::enum_type_list<int, 1, 2, 4, 8, 16>;

CUB_TEST(
  "Warp top-k broadcasts sorted keys and preserves input", "[topk][warp]", CUB_SMALL, key_types, item_counts, k_values)
{
  using key_type      = c2h::get<0, TestType>;
  constexpr int items = c2h::get<1, TestType>::value;
  constexpr int k     = c2h::get<2, TestType>::value;
  check_broadcast_topk<key_type, items, k, 32>(GENERATE(0, 1, 2, 3));
}

CUB_TEST("Warp top-k keeps independent row groups isolated", "[topk][warp]", CUB_SMALL, key_types, logical_warp_sizes)
{
  using key_type      = c2h::get<0, TestType>;
  constexpr int width = c2h::get<1, TestType>::value;
  check_broadcast_topk<key_type, 64, width * 2, width>(GENERATE(0, 1, 2, 3));
}
