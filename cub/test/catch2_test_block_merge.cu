// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_merge.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

#include "cub_test_macros.h"
#include <c2h/custom_type.h>
#include <c2h/generators.h>
#include <c2h/vector.h>

// %PARAM% GEOMETRY geometry 0:1:2:3:4
#ifndef GEOMETRY
#  define GEOMETRY 0
#endif

template <int X, int Y, int Z, int Items>
struct merge_geometry
{
  static constexpr int x     = X;
  static constexpr int y     = Y;
  static constexpr int z     = Z;
  static constexpr int items = Items;
};

using geometries =
  c2h::type_list<merge_geometry<128, 1, 1, 3>,
                 merge_geometry<256, 1, 1, 7>,
                 merge_geometry<8, 4, 2, 2>,
                 merge_geometry<7, 3, 2, 3>,
                 merge_geometry<512, 1, 1, 1>>;
using geometry   = c2h::get<GEOMETRY, geometries>;
using algorithms = c2h::enum_type_list<cub::BlockMergeAlgorithm,
                                       cub::BlockMergeAlgorithm::PING_PONG,
                                       cub::BlockMergeAlgorithm::REGISTER_STAGED>;
using stable_key = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;
using key_types  = c2h::type_list<std::int8_t, std::uint32_t, std::int64_t, float, double, stable_key>;

inline constexpr int max_runs      = 32;
inline constexpr int offset_stride = max_runs + 1;
inline constexpr int reuse_count   = 2;

// Keep the larger custom payload on the existing stability-test geometry so
// both shared-memory buffers fit on supported devices.
template <typename T>
using test_geometry =
  cuda::std::conditional_t<cuda::std::is_same_v<T, stable_key>, merge_geometry<64, 1, 1, 3>, geometry>;

struct merge_order
{
  bool descending;
  template <typename T>
  __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
  {
    return descending ? rhs < lhs : lhs < rhs;
  }
};

template <typename T, typename Geometry, cub::BlockMergeAlgorithm Algorithm>
_CCCL_KERNEL_ATTRIBUTES void merge_runs(
  const T* input, const int* boundaries, T* output, merge_order compare, bool shared_output, bool private_storage)
{
  constexpr int threads = Geometry::x * Geometry::y * Geometry::z;
  constexpr int count   = threads * Geometry::items;
  using merge_t = cub::BlockMerge<T, Geometry::x, Geometry::items, max_runs, Algorithm, Geometry::y, Geometry::z>;
  struct input_storage
  {
    T keys[count + 1];
    int offsets[offset_stride];
  };
  __shared__ cub::Uninitialized<input_storage> shared;
  __shared__ typename merge_t::TempStorage temporary;
  auto& data      = shared.Alias();
  const int tid   = cub::RowMajorTid(Geometry::x, Geometry::y, Geometry::z);
  const int block = static_cast<int>(blockIdx.x);
  const int runs  = block + 1;

  for (int i = tid; i <= runs; i += threads)
  {
    data.offsets[i] = boundaries[block * offset_stride + i];
  }

  // Repeat using the same storage to cover synchronization before shared-memory reuse.
  for (int repeat = 0; repeat < reuse_count; ++repeat)
  {
    for (int i = tid; i <= count; i += threads)
    {
      data.keys[i] = i < count ? input[block * count + i] : T{};
    }
    __syncthreads();
    T items[Geometry::items];
    const T* merged = nullptr;
    auto merge      = private_storage ? merge_t{} : merge_t{temporary};

    if (shared_output)
    {
      merged = merge.MergeToShared(data.keys, data.offsets, runs, compare);
    }
    else
    {
      merge.Merge(data.keys, data.offsets, runs, items, compare);
    }

    for (int i = 0; i < Geometry::items; ++i)
    {
      const int rank = tid * Geometry::items + i;

      output[(block * reuse_count + repeat) * count + rank] =
        shared_output ? (rank < data.offsets[runs] ? merged[rank] : T{}) : items[i];
    }
    __syncthreads();
  }
}

// Generate values once per scenario; stable payloads identify their input order.
template <typename T>
[[nodiscard]] c2h::host_vector<T> make_keys(c2h::seed_t seed, int count)
{
  c2h::device_vector<T> generated(count, thrust::default_init);

  if constexpr (cuda::std::is_arithmetic_v<T>)
  {
    c2h::gen(seed, generated, T{0}, T{31});
  }
  else
  {
    c2h::gen(seed, generated);
  }

  c2h::host_vector<T> keys = generated;

  if constexpr (cuda::std::is_same_v<T, stable_key>)
  {
    constexpr int distinct_keys = 7;

    for (std::size_t i = 0; i < keys.size(); ++i)
    {
      keys[i].key %= distinct_keys;
      keys[i].val = i;
    }
  }

  return keys;
}

[[nodiscard]] inline c2h::host_vector<int> even_boundaries(int total)
{
  c2h::host_vector<int> boundaries(max_runs * offset_stride, 0);

  for (int runs = 1; runs <= max_runs; ++runs)
  {
    const auto cuts = boundaries.begin() + (runs - 1) * offset_stride;

    for (int run = 0; run <= runs; ++run)
    {
      cuts[run] = total * run / runs;
    }
  }

  return boundaries;
}

[[nodiscard]] inline c2h::host_vector<int> random_boundaries(c2h::seed_t seed, int total)
{
  c2h::host_vector<int> boundaries(max_runs * offset_stride, 0);
  std::mt19937 random(static_cast<std::mt19937::result_type>(seed.get()));
  std::uniform_int_distribution<int> offset(0, total);

  for (int runs = 1; runs <= max_runs; ++runs)
  {
    const auto cuts = boundaries.begin() + (runs - 1) * offset_stride;

    cuts[runs] = total;

    for (int run = 1; run < runs; ++run)
    {
      cuts[run] = offset(random);
    }

    std::sort(cuts, cuts + runs + 1);
  }

  return boundaries;
}

[[nodiscard]] inline c2h::host_vector<int> single_run_boundaries(int total)
{
  c2h::host_vector<int> boundaries(max_runs * offset_stride, 0);

  for (int runs = 1; runs <= max_runs; ++runs)
  {
    const int populated_run = runs / 2;
    const auto cuts         = boundaries.begin() + (runs - 1) * offset_stride;

    for (int run = populated_run + 1; run <= runs; ++run)
    {
      cuts[run] = total;
    }
  }

  return boundaries;
}

template <typename T, typename Geometry, cub::BlockMergeAlgorithm Algorithm>
void check_runs(c2h::host_vector<T> input,
                const c2h::host_vector<int>& boundaries,
                bool descending,
                bool shared_output,
                bool private_storage)
{
  constexpr int capacity = Geometry::x * Geometry::y * Geometry::z * Geometry::items;
  c2h::host_vector<T> expected(max_runs * capacity * reuse_count, T{});
  const merge_order compare{descending};

  for (int runs = 1; runs <= max_runs; ++runs)
  {
    const int tile  = runs - 1;
    const auto keys = input.begin() + tile * capacity;
    const auto cuts = boundaries.begin() + tile * offset_stride;
    const int total = cuts[runs];

    for (int run = 0; run < runs; ++run)
    {
      std::stable_sort(keys + cuts[run], keys + cuts[run + 1], compare);
    }

    const auto merged = expected.begin() + tile * reuse_count * capacity;

    std::copy_n(keys, total, merged);
    std::stable_sort(merged, merged + total, compare);

    for (int repeat = 1; repeat < reuse_count; ++repeat)
    {
      std::copy_n(merged, capacity, merged + repeat * capacity);
    }
  }

  c2h::device_vector<T> d_input        = input;
  c2h::device_vector<int> d_boundaries = boundaries;
  c2h::device_vector<T> d_output(expected.size(), thrust::default_init);
  const dim3 block(Geometry::x, Geometry::y, Geometry::z);

  merge_runs<T, Geometry, Algorithm><<<max_runs, block>>>(
    thrust::raw_pointer_cast(d_input.data()),
    thrust::raw_pointer_cast(d_boundaries.data()),
    thrust::raw_pointer_cast(d_output.data()),
    compare,
    shared_output,
    private_storage);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(expected == d_output);
}

CUB_TEST("BlockMerge merges evenly sized runs", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  const auto input           = make_keys<type>(seed, max_runs * capacity);
  const auto boundaries      = even_boundaries(capacity);

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge merges uneven runs", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  const auto input           = make_keys<type>(seed, max_runs * capacity);
  const auto boundaries      = random_boundaries(seed, capacity);

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge merges a single populated run", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  const auto input           = make_keys<type>(seed, max_runs * capacity);
  const auto boundaries      = single_run_boundaries(capacity);

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge merges a partial tile", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  const auto input           = make_keys<type>(seed, max_runs * capacity);
  const int live_keys        = capacity / 2 + 1;
  const auto boundaries      = random_boundaries(seed, live_keys);

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge merges a single live key", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  const auto input           = make_keys<type>(seed, max_runs * capacity);
  const auto boundaries      = single_run_boundaries(1);

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge merges empty runs", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  const auto input           = make_keys<type>(seed, max_runs * capacity);
  const c2h::host_vector<int> boundaries(max_runs * offset_stride, 0);

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge merges equivalent keys", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  auto input                 = make_keys<type>(seed, max_runs * capacity);
  const auto boundaries      = random_boundaries(seed, capacity);

  // Keep payload positions distinct while making every comparison key equal.
  for (auto& key : input)
  {
    if constexpr (cuda::std::is_same_v<type, stable_key>)
    {
      key.key = 1;
    }
    else
    {
      key = type{1};
    }
  }

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge merges already ordered runs", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  using test_geometry_t      = test_geometry<type>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  constexpr int capacity     = test_geometry_t::x * test_geometry_t::y * test_geometry_t::z * test_geometry_t::items;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  auto input                 = make_keys<type>(seed, max_runs * capacity);
  const auto boundaries      = random_boundaries(seed, capacity);

  const merge_order compare{descending};

  for (int tile = 0; tile < max_runs; ++tile)
  {
    const auto first = input.begin() + tile * capacity;

    std::stable_sort(first, first + capacity, compare);
  }

  check_runs<type, test_geometry_t, algorithm>(input, boundaries, descending, shared_output, private_storage);
}
