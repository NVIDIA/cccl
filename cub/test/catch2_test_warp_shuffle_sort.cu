// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// %PARAM% WARP_SHUFFLE_ITEMS ipt 1:2:4:8:16:32

#include <cub/warp/warp_shuffle_sort.cuh>

#include <cuda/std/type_traits>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "cub_test_macros.h"

inline constexpr int warp_threads    = 32;
inline constexpr int num_blocks      = 3;
inline constexpr int warps_per_block = 4;

enum class input_pattern
{
  random,
  ordered,
  reversed,
  duplicate_heavy,
  alternating_zeros
};

struct record_key
{
  int order;
  unsigned data;
};

template <typename T>
__host__ __device__ T compare_field(T key)
{
  return key;
}

__host__ __device__ inline int compare_field(record_key key)
{
  return key.order;
}

struct compare_keys
{
  bool descending;

  template <typename Key>
  __host__ __device__ bool operator()(Key lhs, Key rhs) const
  {
    return descending ? compare_field(rhs) < compare_field(lhs) : compare_field(lhs) < compare_field(rhs);
  }
};

template <typename Key, int Items, int LogicalWarpThreads, bool AllLanesActive>
__global__ void shuffle_sort_kernel(const Key* input, Key* output, bool descending, bool skip_odd_warps)
{
  using warp_shuffle_sort = cub::detail::WarpShuffleSort<Key, Items, LogicalWarpThreads, AllLanesActive>;
  constexpr int tile_size = LogicalWarpThreads * Items;
  const int tid           = static_cast<int>(threadIdx.x);
  const int warp          = tid / LogicalWarpThreads;
  const int lane          = tid % LogicalWarpThreads;
  const int tile          = static_cast<int>(blockIdx.x * blockDim.x) / LogicalWarpThreads + warp;

  const warp_shuffle_sort warp_sort{};

  Key keys[Items];
  const int thread_offset = tile * tile_size + lane * Items;
  for (int slot = 0; slot < Items; ++slot)
  {
    keys[slot] = input[thread_offset + slot];
  }
  if (AllLanesActive || !skip_odd_warps || tile % 2 == 0)
  {
    warp_sort.Sort(keys, compare_keys{descending});
  }
  for (int slot = 0; slot < Items; ++slot)
  {
    output[thread_offset + slot] = keys[slot];
  }
}

template <typename Key>
Key make_key(int rank, int size, input_pattern pattern)
{
  if constexpr (cuda::std::is_same_v<Key, record_key>)
  {
    return {make_key<int>(rank, size, pattern), static_cast<unsigned>(rank)};
  }
  else
  {
    static std::mt19937 generator;
    switch (pattern)
    {
      case input_pattern::random:
        if constexpr (cuda::std::is_floating_point_v<Key>)
        {
          std::uniform_real_distribution<Key> distribution(-256, 256);
          return distribution(generator);
        }
        else
        {
          return static_cast<Key>(generator());
        }
      case input_pattern::ordered:
        return static_cast<Key>(rank);
      case input_pattern::reversed:
        return static_cast<Key>(size - rank);
      case input_pattern::duplicate_heavy:
        return static_cast<Key>(generator() % 5);
      case input_pattern::alternating_zeros:
        if constexpr (cuda::std::is_floating_point_v<Key>)
        {
          return rank % 2 ? Key{-0.0} : Key{0.0};
        }
        else
        {
          return static_cast<Key>(rank % 2);
        }
    }
    return Key{};
  }
}

// byte-level comparison for +-0 and structures
template <typename Key>
auto sorted_representations(const Key* keys, int size)
{
  using representation = std::array<unsigned char, sizeof(Key)>;
  std::vector<representation> result(size);
  for (int rank = 0; rank < size; ++rank)
  {
    std::memcpy(result[rank].data(), keys + rank, sizeof(Key));
  }
  std::sort(result.begin(), result.end());
  return result;
}

using key_types          = c2h::type_list<uint8_t, int, unsigned, int64_t, uint64_t, float, double, record_key>;
using logical_warp_sizes = c2h::enum_type_list<int, 1, 2, 8, 32>;
using active_lane_modes  = c2h::enum_type_list<bool, true, false>;

CUB_TEST("Warp shuffle warp_sort preserves blocked keys and permutations",
         "[warp_sort][warp][shuffle]",
         CUB_SMALL,
         key_types,
         logical_warp_sizes,
         active_lane_modes)
{
  using key_t                        = c2h::get<0, TestType>;
  constexpr int items                = WARP_SHUFFLE_ITEMS;
  constexpr int logical_warp_threads = c2h::get<1, TestType>::value;
  constexpr bool all_lanes_active    = c2h::get<2, TestType>::value;
  constexpr int tile_size            = logical_warp_threads * items;
  constexpr int block_size           = warp_threads * warps_per_block;
  constexpr int shuffle_sort_tiles   = num_blocks * block_size / logical_warp_threads;
  const auto pattern                 = GENERATE(
    input_pattern::random,
    input_pattern::ordered,
    input_pattern::reversed,
    input_pattern::duplicate_heavy,
    input_pattern::alternating_zeros);
  const bool descending     = GENERATE(false, true);
  const bool skip_odd_warps = !all_lanes_active && GENERATE(false, true);
  CAPTURE(items, logical_warp_threads, all_lanes_active, pattern, descending, skip_odd_warps);

  c2h::host_vector<key_t> input(shuffle_sort_tiles * tile_size);
  for (int tile = 0; tile < shuffle_sort_tiles; ++tile)
  {
    for (int rank = 0; rank < tile_size; ++rank)
    {
      input[tile * tile_size + rank] = make_key<key_t>(rank, tile_size, pattern);
    }
  }

  auto reference = input;
  for (int tile = 0; tile < shuffle_sort_tiles; ++tile)
  {
    if (!skip_odd_warps || tile % 2 == 0)
    {
      auto tile_start = reference.begin() + tile * tile_size;
      auto tile_end   = reference.begin() + (tile + 1) * tile_size;
      std::sort(tile_start, tile_end, compare_keys{descending});
    }
  }

  const c2h::device_vector<key_t> device_input(input);
  c2h::device_vector<key_t> device_output(input.size());

  shuffle_sort_kernel<key_t, items, logical_warp_threads, all_lanes_active><<<num_blocks, block_size>>>(
    thrust::raw_pointer_cast(device_input.data()),
    thrust::raw_pointer_cast(device_output.data()),
    descending,
    skip_odd_warps);
  REQUIRE(cudaSuccess == cudaGetLastError());
  const c2h::host_vector<key_t> output(device_output);

  for (int tile = 0; tile < shuffle_sort_tiles; ++tile)
  {
    const int tile_offset = tile * tile_size;
    const auto* actual    = thrust::raw_pointer_cast(output.data()) + tile_offset;
    const auto* expected  = thrust::raw_pointer_cast(reference.data()) + tile_offset;

    // Preserve +-0 and extra fields of structures
    if constexpr (cuda::std::is_floating_point_v<key_t> || cuda::std::is_same_v<key_t, record_key>)
    {
      if (!skip_odd_warps || tile % 2 == 0)
      {
        REQUIRE(std::is_sorted(actual, actual + tile_size, compare_keys{descending}));
      }
      REQUIRE(sorted_representations(actual, tile_size) == sorted_representations(expected, tile_size));
    }
    else
    {
      REQUIRE(std::equal(actual, actual + tile_size, expected));
    }
  }
}
