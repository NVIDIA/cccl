// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_merge.cuh>
#include <cub/detail/uninitialized.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/type_traits>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>

#include "cub_test_macros.h"
#include <c2h/custom_type.h>
#include <c2h/generators.h>
#include <c2h/vector.h>

// %PARAM% GEOMETRY geometry 0:1:2:3
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
                 merge_geometry<7, 3, 2, 3>>;
using geometry = c2h::get<GEOMETRY, geometries>;
using algorithms =
  c2h::enum_type_list<cub::BlockMergeAlgorithm, cub::BLOCK_MERGE_PING_PONG, cub::BLOCK_MERGE_REGISTER_STAGED>;
using key_types  = c2h::type_list<std::int8_t, std::uint32_t, std::int64_t, float, double>;
using stable_key = c2h::custom_type_t<c2h::less_comparable_t, c2h::equal_comparable_t>;

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
  using merge_t         = cub::BlockMerge<T, Geometry::x, Geometry::items, 32, Algorithm, Geometry::y, Geometry::z>;
  struct input_storage
  {
    T keys[count + 1];
    int offsets[33];
  };
  __shared__ cub::Uninitialized<input_storage> shared;
  __shared__ typename merge_t::TempStorage temporary;
  auto& data      = shared.Alias();
  const int tid   = cub::RowMajorTid(Geometry::x, Geometry::y, Geometry::z);
  const int block = static_cast<int>(blockIdx.x);
  const int runs  = block / 8 + 1;
  for (int i = tid; i <= runs; i += threads)
  {
    data.offsets[i] = boundaries[block * 33 + i];
  }

  // Repeat using the same storage to cover synchronization before shared-memory reuse.
  for (int repeat = 0; repeat < 2; ++repeat)
  {
    for (int i = tid; i <= count; i += threads)
    {
      data.keys[i] = i < count ? input[block * count + i] : T{};
    }
    __syncthreads();
    T items[Geometry::items];
    const T* merged = nullptr;
    if (private_storage)
    {
      if (shared_output)
      {
        merged = merge_t{}.MergeToShared(data.keys, data.offsets, runs, compare);
      }
      else
      {
        merge_t{}.Merge(data.keys, data.offsets, runs, items, compare);
      }
    }
    else
    {
      if (shared_output)
      {
        merged = merge_t{temporary}.MergeToShared(data.keys, data.offsets, runs, compare);
      }
      else
      {
        merge_t{temporary}.Merge(data.keys, data.offsets, runs, items, compare);
      }
    }
    for (int i = 0; i < Geometry::items; ++i)
    {
      const int rank = tid * Geometry::items + i;
      output[(block * 2 + repeat) * count + rank] =
        shared_output ? (rank < data.offsets[runs] ? merged[rank] : T{}) : items[i];
    }
    __syncthreads();
  }
}

template <typename T, typename Geometry, cub::BlockMergeAlgorithm Algorithm>
void check_runs(c2h::seed_t seed, bool descending, bool shared_output, bool private_storage)
{
  constexpr int count  = Geometry::x * Geometry::y * Geometry::z * Geometry::items;
  constexpr int blocks = 32 * 8;
  c2h::device_vector<T> d_input(blocks * count);
  if constexpr (cuda::std::is_arithmetic_v<T>)
  {
    c2h::gen(seed, d_input, T{0}, T{31});
  }
  else
  {
    c2h::gen(seed, d_input);
  }
  c2h::host_vector<T> input = d_input;
  c2h::host_vector<int> boundaries(blocks * 33, 0);
  c2h::host_vector<T> expected(blocks * count * 2, T{});
  std::mt19937 random(42);
  const merge_order compare{descending};
  for (int runs = 1; runs <= 32; ++runs)
  {
    for (int shape = 0; shape < 8; ++shape)
    {
      const int block = (runs - 1) * 8 + shape;
      const int total = shape == 5 ? 0 : shape == 4 ? 1 : shape == 3 ? count / 2 + 1 : count;
      auto keys       = input.begin() + block * count;
      auto cuts       = boundaries.begin() + block * 33;
      cuts[0]         = 0;
      cuts[runs]      = total;
      for (int r = 1; r < runs; ++r)
      {
        cuts[r] = shape == 0 ? total * r / runs
                : shape == 2 || shape == 4
                  ? (r <= runs / 2 ? 0 : total)
                  : static_cast<int>(random() % (total + 1));
      }
      std::sort(cuts, cuts + runs + 1);
      if constexpr (cuda::std::is_same_v<T, stable_key>)
      {
        for (int i = 0; i < count; ++i)
        {
          keys[i].key %= 7;
          keys[i].val = static_cast<std::size_t>(i);
        }
      }
      if (shape == 6)
      {
        for (int i = 0; i < count; ++i)
        {
          if constexpr (cuda::std::is_same_v<T, stable_key>)
          {
            keys[i].key = 1;
          }
          else
          {
            keys[i] = T{1};
          }
        }
      }
      if (shape == 7)
      {
        std::stable_sort(keys, keys + total, compare);
      }
      for (int r = 0; r < runs; ++r)
      {
        std::stable_sort(keys + cuts[r], keys + cuts[r + 1], compare);
      }
      std::copy_n(keys, total, expected.begin() + block * 2 * count);
      std::stable_sort(expected.begin() + block * 2 * count, expected.begin() + block * 2 * count + total, compare);
      std::copy_n(expected.begin() + block * 2 * count, count, expected.begin() + (block * 2 + 1) * count);
    }
  }
  d_input                              = input;
  c2h::device_vector<int> d_boundaries = boundaries;
  c2h::device_vector<T> d_output(expected.size());
  const dim3 block(Geometry::x, Geometry::y, Geometry::z);
  merge_runs<T, Geometry, Algorithm><<<blocks, block>>>(
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

CUB_TEST("BlockMerge merges empty, uneven and partial runs", "[block][merge]", CUB_SMALL, key_types, algorithms)
{
  using type                 = c2h::get<0, TestType>;
  constexpr auto algorithm   = c2h::get<1, TestType>::value;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  check_runs<type, geometry, algorithm>(seed, descending, shared_output, private_storage);
}

CUB_TEST("BlockMerge preserves the order of equivalent keys", "[block][merge][stable]", CUB_SMALL, algorithms)
{
  constexpr auto algorithm   = c2h::get<0, TestType>::value;
  const auto seed            = C2H_SEED(2);
  const bool descending      = GENERATE(false, true);
  const bool shared_output   = GENERATE(false, true);
  const bool private_storage = GENERATE(false, true);
  check_runs<stable_key, merge_geometry<64, 1, 1, 3>, algorithm>(seed, descending, shared_output, private_storage);
}
