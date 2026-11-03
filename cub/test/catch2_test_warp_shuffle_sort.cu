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

inline constexpr int warp_threads       = 32;
inline constexpr int num_blocks         = 3;
inline constexpr int warps_per_block    = 4;
inline constexpr int shuffle_sort_tiles = num_blocks * warps_per_block;

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

template <typename Key, int Items>
__global__ void shuffle_sort_kernel(const Key* input, Key* output, bool descending, bool with_storage)
{
  using warp_shuffle_sort = cub::detail::WarpShuffleSort<Key, Items>;
  constexpr int tile_size = warp_threads * Items;
  const int tid           = static_cast<int>(threadIdx.x);
  const int warp          = tid / warp_threads;
  const int lane          = tid % warp_threads;
  const int tile          = static_cast<int>(blockIdx.x) * warps_per_block + warp;

  typename warp_shuffle_sort::TempStorage unused;
  const warp_shuffle_sort warp_sort = with_storage ? warp_shuffle_sort{unused} : warp_shuffle_sort{};

  Key keys[Items];
  const int thread_offset = tile * tile_size + lane * Items;
  for (int slot = 0; slot < Items; ++slot)
  {
    keys[slot] = input[thread_offset + slot];
  }
  warp_sort.Sort(keys, compare_keys{descending});
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

using key_types = c2h::type_list<uint8_t, int, unsigned, int64_t, uint64_t, float, double, record_key>;

CUB_TEST(
  "Warp shuffle warp_sort preserves blocked keys and permutations", "[warp_sort][warp][shuffle]", CUB_SMALL, key_types)
{
  using key_t             = c2h::get<0, TestType>;
  constexpr int items     = WARP_SHUFFLE_ITEMS;
  constexpr int tile_size = warp_threads * items;
  const auto pattern      = GENERATE(
    input_pattern::random,
    input_pattern::ordered,
    input_pattern::reversed,
    input_pattern::duplicate_heavy,
    input_pattern::alternating_zeros);
  const bool descending    = GENERATE(false, true);
  constexpr int block_size = warp_threads * warps_per_block;
  const bool with_storage  = GENERATE(false, true);
  CAPTURE(items, pattern, descending, block_size, with_storage);

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
    auto tile_start = reference.begin() + tile * tile_size;
    auto tile_end   = reference.begin() + (tile + 1) * tile_size;
    std::sort(tile_start, tile_end, compare_keys{descending});
  }

  const c2h::device_vector<key_t> device_input(input);
  c2h::device_vector<key_t> device_output(input.size());

  shuffle_sort_kernel<key_t, items><<<num_blocks, block_size>>>(
    thrust::raw_pointer_cast(device_input.data()),
    thrust::raw_pointer_cast(device_output.data()),
    descending,
    with_storage);
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
      REQUIRE(std::is_sorted(actual, actual + tile_size, compare_keys{descending}));
      REQUIRE(sorted_representations(actual, tile_size) == sorted_representations(expected, tile_size));
    }
    else
    {
      REQUIRE(std::equal(actual, actual + tile_size, expected));
    }
  }
}
