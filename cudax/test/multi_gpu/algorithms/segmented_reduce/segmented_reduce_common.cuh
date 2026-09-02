//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDAX_TEST_MULTI_GPU_ALGORITHMS_SEGMENTED_REDUCE_SEGMENTED_REDUCE_COMMON_CUH
#define CUDAX_TEST_MULTI_GPU_ALGORITHMS_SEGMENTED_REDUCE_SEGMENTED_REDUCE_COMMON_CUH

#include <cuda/buffer>
#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/random>
#include <cuda/std/span>
#include <cuda/std/type_traits>

#include <algorithm>
#include <numeric>
#include <vector>

#include <algorithm_common.h>
#include <nccl_test_common.h>

#include <c2h/catch2_test_helper.h>

using offset_type = cuda::std::int32_t;

// `custom_value` only orders on `key`, so two elements that share a key but differ in `val` are
// an unbroken tie: `cuda::maximum<>` may return either one and the result then depends on the
// fold order, which the host reference and the device do not share. Deriving `val` from `key`
// makes tied elements fully equal and the reduction order-independent.
template <class T, class RNG>
void fill_random(std::vector<T>& local, cuda::std::size_t count, RNG& rng)
{
  constexpr cuda::std::int64_t lo = cuda::std::is_same_v<T, custom_value> ? 0 : -1024;
  cuda::std::uniform_int_distribution<cuda::std::int64_t> dist{lo, 1024};

  local.resize(count);
  std::generate(local.begin(), local.end(), [&] {
    const auto key = dist(rng);

    return make_value<T>(key, key);
  });
}

// Segment sizes to the prefix-sum offsets `segmented_reduce` consumes. The result holds one more
// entry than there are segments, since the last offset closes the final segment.
[[nodiscard]] inline std::vector<offset_type> make_offsets(cuda::std::span<const offset_type> segment_sizes)
{
  std::vector<offset_type> offsets{0};

  offsets.reserve(segment_sizes.size() + 1);
  for (const auto size : segment_sizes)
  {
    offsets.push_back(offsets.back() + size);
  }
  return offsets;
}

// Every rank gets the same segment layout, but its own random values.
template <class T, cuda::std::size_t N, class RNG>
[[nodiscard]] std::vector<std::vector<T>>
uniform_values(int num_ranks, const cuda::std::array<offset_type, N>& segment_sizes, RNG& rng)
{
  const auto count =
    static_cast<cuda::std::size_t>(std::accumulate(segment_sizes.begin(), segment_sizes.end(), offset_type{0}));
  std::vector<std::vector<T>> ret(static_cast<cuda::std::size_t>(num_ranks));

  for (auto& values : ret)
  {
    fill_random(values, count, rng);
  }
  return ret;
}

[[nodiscard]] inline std::vector<std::vector<offset_type>>
uniform_offsets(int num_ranks, cuda::std::span<const offset_type> segment_sizes)
{
  return std::vector<std::vector<offset_type>>(static_cast<cuda::std::size_t>(num_ranks), make_offsets(segment_sizes));
}

// Segment `s` of the result folds segment `s` of every rank, so the host reference concatenates
// the per-rank slices of that segment and accumulates them seeded by `init`. `segmented_reduce`
// broadcasts the result, so every local output must hold the same values.
template <class T, class Op>
void check_outputs(
  const std::vector<cuda::device_buffer<T>>& out,
  const std::vector<std::vector<T>>& values_by_rank,
  const std::vector<std::vector<offset_type>>& offsets_by_rank,
  cuda::std::size_t num_segments,
  const T& init,
  Op op)
{
  std::vector<T> expected;

  expected.reserve(num_segments);
  for (cuda::std::size_t s = 0; s < num_segments; ++s)
  {
    std::vector<T> reference;

    for (cuda::std::size_t r = 0; r < values_by_rank.size(); ++r)
    {
      const auto& values  = values_by_rank[r];
      const auto& offsets = offsets_by_rank[r];

      reference.insert(reference.end(), values.begin() + offsets[s], values.begin() + offsets[s + 1]);
    }
    expected.push_back(std::accumulate(reference.begin(), reference.end(), init, op));
  }

  for (cuda::std::size_t i = 0; i < out.size(); ++i)
  {
    INFO("device = " << i);

    const auto exp = cuda::make_buffer<T>(out[i].stream(), cuda::mr::legacy_pinned_memory_resource{}, expected);

    REQUIRE_THAT(out[i], Equals(exp));
  }
}

#endif // CUDAX_TEST_MULTI_GPU_ALGORITHMS_SEGMENTED_REDUCE_SEGMENTED_REDUCE_COMMON_CUH
