//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/numeric>
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

namespace
{
// Use the same checks for exhaustive and non-exhaustive mapping.
template <bool Exhaustive, class Mapping, class ParentGroup, class PrevMappingResult>
__device__ auto map_group_as(const Mapping& mapping, const ParentGroup& parent, const PrevMappingResult& prev)
{
  if constexpr (Exhaustive)
  {
    return mapping.map(cuda::gpu_thread, parent, prev);
  }
  else
  {
    return mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);
  }
}

// Compute expected ranks and lane masks independently of the mapping implementation.
template <class Range, class Result>
__device__ void check_mapping_result(const Range& counts, const Result& result, unsigned rank_in_warp)
{
  const auto total_units = cuda::std::accumulate(cuda::std::ranges::begin(counts), cuda::std::ranges::end(counts), 0u);
  const bool is_valid    = rank_in_warp < total_units;

  CHECK(result.is_valid() == is_valid);

  // Excluded threads cannot query group ranks, sizes, or lane masks.
  if (!is_valid)
  {
    return;
  }

  // Find the subgroup containing this thread by walking the requested counts.
  unsigned group_rank  = 0;
  unsigned group_start = 0;
  for (const auto count : counts)
  {
    const auto unit_count = static_cast<unsigned>(count);

    if (rank_in_warp < group_start + unit_count)
    {
      const auto lane_mask = unit_count < 32 ? ((1u << unit_count) - 1) << group_start : ~0u;

      CHECK(result.group_count() == cuda::std::ranges::size(counts));
      CHECK(result.group_rank() == group_rank);

      CHECK(result.unit_count() == unit_count);
      CHECK(result.unit_rank() == rank_in_warp - group_start);

      CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask});

      return;
    }

    group_start += unit_count;
    ++group_rank;
  }
}

// Runtime ranges must preserve their view type and expose dynamic result counts.
template <bool Exhaustive, class Config, class Range>
__device__ void test_dynamic_group_as(Config config, Range&& counts)
{
  using Mapping = cudax::coop::group_as<cuda::std::views::all_t<Range>>;

  static_assert(!cuda::std::is_default_constructible_v<Mapping>);
  static_assert(cuda::std::is_constructible_v<Mapping, Range>);

  const cudax::coop::group_as mapping{counts};
  static_assert(cuda::std::is_same_v<cuda::std::remove_const_t<decltype(mapping)>, Mapping>);

  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;

  const auto result = map_group_as<Exhaustive>(mapping, parent, prev);
  using Result      = decltype(result);

  static_assert(cudax::coop::__group_mapping_result<Result>);
  static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
  static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);

  static_assert(Result::is_always_exhaustive() == Exhaustive);
  static_assert(Result::is_always_contiguous());

  check_mapping_result(counts, result, cuda::gpu_thread.rank_as<unsigned>(parent));
}

template <bool Exhaustive, cuda::std::uint32_t... Counts, class Config>
__device__ void test_dynamic_group_as(Config config)
{
  const cuda::std::uint32_t c_array[]{Counts...};
  const cuda::std::array array{Counts...};

  // Exercise owning containers and a non-owning view with identical counts.
  test_dynamic_group_as<Exhaustive>(config, c_array);
  test_dynamic_group_as<Exhaustive>(config, array);
  test_dynamic_group_as<Exhaustive>(config, cuda::std::span{c_array, sizeof...(Counts)});
}

#if _CCCL_STD_VER >= 2020
// Class-valued template arguments allow array partitions to be passed through cw.
inline constexpr cuda::std::array static_counts{4, 12, 16};
inline constexpr cuda::std::array partial_static_counts{4u, 6u, 8u};
#endif // _CCCL_STD_VER >= 2020

// Exercise a compile-time range whose iterators only model forward_iterator.
struct forward_counts
{
  struct iterator
  {
    using value_type        = int;
    using difference_type   = cuda::std::ptrdiff_t;
    using iterator_category = cuda::std::forward_iterator_tag;
    using pointer           = const int*;
    using reference         = const int&;

    const int* ptr{};

    __host__ __device__ constexpr const int& operator*() const
    {
      return *ptr;
    }

    __host__ __device__ constexpr iterator& operator++()
    {
      ++ptr;
      return *this;
    }

    __host__ __device__ constexpr iterator operator++(int)
    {
      const auto old = *this;
      ++*this;
      return old;
    }

    __host__ __device__ friend constexpr bool operator==(iterator a, iterator b)
    {
      return a.ptr == b.ptr;
    }

    __host__ __device__ friend constexpr bool operator!=(iterator a, iterator b)
    {
      return !(a == b);
    }
  };

  int counts[3]{4, 12, 16};

  __host__ __device__ constexpr iterator begin() const
  {
    return {counts};
  }

  __host__ __device__ constexpr iterator end() const
  {
    return {counts + 3};
  }

  __host__ __device__ constexpr cuda::std::size_t size() const
  {
    return 3;
  }
};

// Validate the fixture itself, including postfix increment in constant evaluation.
static_assert([] {
  constexpr forward_counts counts;
  auto it        = counts.begin();
  const auto old = it++;

  return counts.size() == 3 && old != it && *old == 4 && *it == 12 && it != counts.end();
}());

static_assert(cuda::std::ranges::forward_range<const forward_counts>);
static_assert(!cuda::std::ranges::random_access_range<const forward_counts>);

// Sequence and cw inputs must normalize to the same type and retain every static count.
template <bool Exhaustive, class Config, class Input, cuda::std::size_t... Counts>
__device__ void test_static_group_as(Config config, Input input, cuda::std::index_sequence<Counts...>)
{
  using ExpectedMapping = cudax::coop::group_as<cuda::std::index_sequence<Counts...>>;
  const cudax::coop::group_as mapping{input};

  static_assert(cuda::std::is_same_v<cuda::std::remove_const_t<decltype(mapping)>, ExpectedMapping>);

  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;

  const auto result = map_group_as<Exhaustive>(mapping, parent, prev);
  using Result      = decltype(result);

  // Check the static metadata for every subgroup, rather than only the current one.
  static_assert(Result::static_group_count() == sizeof...(Counts));

  static_assert([] {
    constexpr cuda::std::array expected_counts{Counts...};

    for (cuda::std::size_t i = 0; i < expected_counts.size(); ++i)
    {
      if (Result::static_unit_count(i) != expected_counts[i])
      {
        return false;
      }
    }

    return true;
  }());

  static_assert(!Result::has_uniform_static_unit_count());
  static_assert(Result::is_always_exhaustive() == Exhaustive);
  static_assert(Result::is_always_contiguous());

  // Runtime queries must agree with the same independent reference used for dynamic inputs.
  const cuda::std::array counts{Counts...};

  check_mapping_result(counts, result, cuda::gpu_thread.rank_as<unsigned>(parent));
}

template <class Config>
__device__ void test_static_composition(Config config)
{
  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;

  // Repeating the partition across previous groups must repeat the static metadata.
  const auto halves = cudax::coop::group_by{cuda::std::cw<16>}.map(cuda::gpu_thread, parent, prev);
  const cudax::coop::group_as split{cuda::std::integer_sequence<int, 4, 12>{}};
  const auto repeated = split.map(cuda::gpu_thread, parent, halves);
  using Repeated      = decltype(repeated);

  static_assert(Repeated::static_group_count() == 4);
  static_assert(Repeated::static_unit_count(0) == 4);
  static_assert(Repeated::static_unit_count(1) == 12);
  static_assert(Repeated::static_unit_count(2) == 4);
  static_assert(Repeated::static_unit_count(3) == 12);

  CHECK(repeated.group_rank() == 2 * halves.group_rank() + (halves.unit_rank() >= 4));

  // A following mapping can consume the per-group static counts.
  const auto quarters = cudax::coop::group_by{cuda::std::cw<4>}.map(cuda::gpu_thread, parent, repeated);
  static_assert(decltype(quarters)::static_group_count() == 8);
  CHECK(quarters.group_rank() == cuda::gpu_thread.rank(parent) / 4);
}

template <class Config>
__device__ void test_static_group_as_special_cases(Config config)
{
  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;

  // Equal subgroup sizes also provide a uniform static unit count.
  const auto uniform_groups =
    cudax::coop::group_as{cuda::std::integer_sequence<short, 16, 16>{}}.map(cuda::gpu_thread, parent, prev);
  static_assert(decltype(uniform_groups)::has_uniform_static_unit_count());
  static_assert(decltype(uniform_groups)::static_unit_count() == 16);
  CHECK(uniform_groups.unit_rank() == cuda::gpu_thread.rank(parent) % 16);

  // A single subgroup retains all lanes in the parent warp.
  const auto single_group =
    cudax::coop::group_as{cuda::std::integer_sequence<unsigned long, 32>{}}.map(cuda::gpu_thread, parent, prev);
  static_assert(decltype(single_group)::static_group_count() == 1);
  static_assert(decltype(single_group)::static_unit_count() == 32);
  CHECK(single_group.lane_mask() == prev.lane_mask());

  // Empty subgroups retain their ranks; every thread belongs to the middle subgroup.
  const auto empty_subgroups =
    cudax::coop::group_as{cuda::std::integer_sequence<int, 0, 32, 0>{}}.map(cuda::gpu_thread, parent, prev);
  static_assert(decltype(empty_subgroups)::static_group_count() == 3);
  static_assert(decltype(empty_subgroups)::static_unit_count(0) == 0);
  CHECK(empty_subgroups.group_rank() == 1);
}

template <class Config>
__device__ void test_static_previous_mappings(Config config)
{
  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;

  // Explicit specializations require matching, nonnegative counts.
  using Split = cudax::coop::group_as<cuda::std::index_sequence<4, 12>>;
  static_assert(cuda::std::is_constructible_v<Split, const cuda::std::integer_sequence<int, 4, 12>&>);
  static_assert(!cuda::std::is_constructible_v<Split, const cuda::std::integer_sequence<int, 4, 13>&>);
  static_assert(!cuda::std::is_constructible_v<Split, const cuda::std::integer_sequence<int, -1, 12>&>);

  // Threads excluded by the previous mapping must stay excluded.
  const auto partial =
    cudax::coop::take{cuda::std::cw<16>}.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);
  const cudax::coop::group_as split{cuda::std::integer_sequence<int, 4, 12>{}};
  const auto result = split.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, partial);
  static_assert(decltype(result)::static_group_count() == 2);
  CHECK(result.is_valid() == (cuda::gpu_thread.rank(parent) < 16));

  // Dynamic previous group counts retain the existing runtime representation.
  const auto dynamic_prev   = cudax::coop::group_by{16}.map(cuda::gpu_thread, parent, prev);
  const auto dynamic_result = split.map(cuda::gpu_thread, parent, dynamic_prev);
  static_assert(decltype(dynamic_result)::static_group_count() == cuda::std::dynamic_extent);
  CHECK(dynamic_result.group_count() == 4);
  CHECK(dynamic_result.unit_count() == (dynamic_prev.unit_rank() < 4 ? 4 : 12));
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // Runtime counts: C arrays, cuda::std::array, and spans.
    test_dynamic_group_as<true, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
      config);
    test_dynamic_group_as<true, 2, 4, 8, 16, 2>(config);
    test_dynamic_group_as<true, 3, 5, 1, 1, 22>(config);
    test_dynamic_group_as<true, 31, 1>(config);
    test_dynamic_group_as<true, 32>(config);

    // Non-exhaustive mapping also covers partitions that leave trailing threads unassigned.
    test_dynamic_group_as<false, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
      config);
    test_dynamic_group_as<false, 2, 4, 8, 16, 2>(config);
    test_dynamic_group_as<false, 3, 5, 1, 1, 22>(config);
    test_dynamic_group_as<false, 31, 1>(config);
    test_dynamic_group_as<false, 32>(config);
    test_dynamic_group_as<false, 31>(config);
    test_dynamic_group_as<false, 4, 6, 8>(config);
    test_dynamic_group_as<false, 2, 2, 3, 1, 14>(config);

    // All static inputs normalize to the same mapping type for equal counts.
    using FullPartition    = cuda::std::index_sequence<4, 12, 16>;
    using PartialPartition = cuda::std::index_sequence<4, 6, 8>;

    test_static_group_as<true>(config, cuda::std::integer_sequence<int, 4, 12, 16>{}, FullPartition{});
    test_static_group_as<true>(config, cuda::std::integer_sequence<unsigned short, 4, 12, 16>{}, FullPartition{});
    test_static_group_as<true>(config, FullPartition{}, FullPartition{});
    test_static_group_as<false>(config, cuda::std::integer_sequence<unsigned, 4, 6, 8>{}, PartialPartition{});

#if _CCCL_STD_VER >= 2020
    // Class-valued cw inputs include both arrays and a forward-only range.
    test_static_group_as<true>(config, cuda::std::cw<static_counts>, FullPartition{});
    test_static_group_as<true>(config, cuda::std::cw<forward_counts{}>, FullPartition{});
    test_static_group_as<false>(config, cuda::std::cw<partial_static_counts>, PartialPartition{});
#endif // _CCCL_STD_VER >= 2020

    test_static_composition(config);
    test_static_group_as_special_cases(config);
    test_static_previous_mappings(config);
  }
};
} // namespace

C2H_TEST("Group-as mapping", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  // Run with both statically known and runtime block dimensions.
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8, 4>());
    cuda::launch(stream, config, TestKernel{});
  }

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{8, 4}));
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
