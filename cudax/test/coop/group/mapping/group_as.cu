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
#include <cuda/std/bit>
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

// Mapping reads the const view, whose size may differ from the mutable view's size.
struct ConstSizedCounts : cuda::std::ranges::view_base
{
  const unsigned* counts;

  __host__ __device__ explicit constexpr ConstSizedCounts(const unsigned* values)
      : counts{values}
  {}

  __host__ __device__ constexpr const unsigned* begin() const
  {
    return counts;
  }

  __host__ __device__ constexpr const unsigned* end() // NOLINT(readability-make-member-function-const)
  {
    return counts + 2;
  }

  __host__ __device__ constexpr const unsigned* end() const
  {
    return counts + 3;
  }

  __host__ __device__ constexpr cuda::std::size_t size() // NOLINT(readability-make-member-function-const)
  {
    return 2;
  }

  __host__ __device__ constexpr cuda::std::size_t size() const
  {
    return 3;
  }
};

// Exercise runtime and compile-time ranges whose iterators only model forward_iterator.
struct ForwardCounts
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
  constexpr ForwardCounts counts{};
  auto it        = counts.begin();
  const auto old = it++;

  return counts.size() == 3 && old != it && *old == 4 && *it == 12 && it != counts.end();
}());

static_assert(cuda::std::ranges::forward_range<const ForwardCounts>);
static_assert(!cuda::std::ranges::random_access_range<const ForwardCounts>);

namespace
{
template <class Mapping, class Parent, class Result, class = void>
inline constexpr bool can_map_group_as = false;

template <class Mapping, class Parent, class Result>
inline constexpr bool can_map_group_as<
  Mapping,
  Parent,
  Result,
  cuda::std::void_t<decltype(cuda::std::declval<Mapping&>().map(
                      cuda::gpu_thread, cuda::std::declval<const Parent&>(), cuda::std::declval<const Result&>())),
                    decltype(cuda::std::declval<Mapping&>().map(cudax::coop::non_exhaustive,
                                                                cuda::gpu_thread,
                                                                cuda::std::declval<const Parent&>(),
                                                                cuda::std::declval<const Result&>()))>> = true;

// Use the same checks for exhaustive and non-exhaustive mapping.
template <bool Exhaustive, class Mapping, class ParentGroup, class PrevMappingResult>
__device__ auto map_group_as(Mapping& mapping, const ParentGroup& parent, const PrevMappingResult& prev)
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
template <class Range, class Result, class PrevMappingResult>
__device__ void check_mapping_result(const Range& counts, const Result& result, const PrevMappingResult& prev)
{
  if (!prev.is_valid())
  {
    CHECK(!result.is_valid());
    return;
  }

  const auto rank        = prev.unit_rank();
  const auto total_units = cuda::std::accumulate(cuda::std::ranges::begin(counts), cuda::std::ranges::end(counts), 0u);
  const bool is_valid    = rank < total_units;

  CHECK(result.is_valid() == is_valid);

  // Excluded threads cannot query group ranks, sizes, or lane masks.
  if (!is_valid)
  {
    return;
  }

  const auto first_lane  = cuda::std::countr_zero(prev.lane_mask().value());
  const auto group_count = static_cast<unsigned>(cuda::std::ranges::size(counts));

  // Find the subgroup containing this thread by walking the requested counts.
  unsigned group_rank  = 0;
  unsigned group_start = 0;
  for (const auto count : counts)
  {
    const auto unit_count = static_cast<unsigned>(count);

    if (rank < group_start + unit_count)
    {
      const auto lane_mask = (unit_count < 32) ? ((1u << unit_count) - 1) << (first_lane + group_start) : ~0u;

      CHECK(result.group_count() == prev.group_count() * group_count);
      CHECK(result.group_rank() == prev.group_rank() * group_count + group_rank);

      CHECK(result.unit_count() == unit_count);
      CHECK(result.unit_rank() == rank - group_start);

      CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask});

      return;
    }

    group_start += unit_count;
    ++group_rank;
  }
}

// Runtime ranges preserve their view type and expose static group counts when their size is known.
template <bool Exhaustive, cuda::std::size_t ExpectedGroupCount, class Config, class Input, class Range>
__device__ void test_range_input(Config config, Input&& input, const Range& counts)
{
  using Mapping = cudax::coop::group_as<cuda::std::views::all_t<Input>>;

  static_assert(!cuda::std::is_default_constructible_v<Mapping>);
  static_assert(cuda::std::is_constructible_v<Mapping, Input>);

  using QualifiedMapping =
    cuda::std::conditional_t<cuda::std::ranges::range<const cuda::std::views::all_t<Input>>, const Mapping, Mapping>;
  QualifiedMapping mapping{cuda::std::forward<Input>(input)};
  static_assert(cuda::std::is_same_v<cuda::std::remove_const_t<decltype(mapping)>, Mapping>);

  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;

  static_assert(can_map_group_as<const Mapping, decltype(parent), ThreadsInWarpMappingResult>
                == cuda::std::ranges::range<const cuda::std::views::all_t<Input>>);

  const auto result = map_group_as<Exhaustive>(mapping, parent, prev);
  using Result      = decltype(result);

  static_assert(cudax::coop::__group_mapping_result<Result>);
  static_assert(Result::static_group_count() == ExpectedGroupCount);
  static_assert(Result::static_group_rank() == (ExpectedGroupCount == 1 ? 0 : cuda::std::dynamic_extent));
  static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);

  static_assert(Result::is_always_exhaustive() == Exhaustive);
  static_assert(Result::is_always_contiguous());

  check_mapping_result(counts, result, prev);
}

template <bool Exhaustive, cuda::std::size_t ExpectedGroupCount, class Config, class Range>
__device__ void test_range_input(Config config, const Range& counts)
{
  test_range_input<Exhaustive, ExpectedGroupCount>(config, counts, counts);
}

template <bool Exhaustive, cuda::std::uint32_t... Counts, class Config>
__device__ void test_runtime_counts(Config config)
{
  const cuda::std::uint32_t c_array[]{Counts...};
  const cuda::std::array array{Counts...};

  // Array wrappers and fixed-extent spans preserve the static size; dynamic-extent spans do not.
  test_range_input<Exhaustive, sizeof...(Counts)>(config, c_array);
  test_range_input<Exhaustive, sizeof...(Counts)>(config, array);
  test_range_input<Exhaustive, sizeof...(Counts)>(config, cuda::std::span{c_array});
  test_range_input<Exhaustive, cuda::std::dynamic_extent>(config, cuda::std::span{c_array, sizeof...(Counts)});

  using MutableView = NonConstView<const cuda::std::uint32_t>;
  static_assert(cuda::std::ranges::sized_range<MutableView>);
  static_assert(!cuda::std::ranges::range<const MutableView>);
  test_range_input<Exhaustive, cuda::std::dynamic_extent>(config, MutableView{c_array, sizeof...(Counts)}, c_array);

  // Rvalue arrays are owned; use the separate C array as the independent reference.
  test_range_input<Exhaustive, sizeof...(Counts)>(config, cuda::std::array{Counts...}, c_array);
}

// A fixed-size partition multiplies the previous static group count, while either dynamic input keeps it dynamic.
template <bool Exhaustive, cuda::std::size_t ExpectedGroupCount, class Config, class Range, class PrevMappingResult>
__device__ void test_runtime_composition(Config config, Range counts, const PrevMappingResult& prev)
{
  const cudax::coop::this_warp parent{config};
  const cudax::coop::group_as mapping{counts};
  const auto result = map_group_as<Exhaustive>(mapping, parent, prev);
  using Result      = decltype(result);

  static_assert(Result::static_group_count() == ExpectedGroupCount);
  static_assert(Result::static_group_rank() == (ExpectedGroupCount == 1 ? 0 : cuda::std::dynamic_extent));
  static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
  static_assert(Result::has_uniform_static_unit_count());
  static_assert(Result::is_always_exhaustive() == Exhaustive);

  check_mapping_result(counts, result, prev);
}

template <bool Exhaustive, class Config>
__device__ void test_runtime_composition(Config config, unsigned last_count)
{
  const cudax::coop::this_warp parent{config};
  const ThreadsInWarpMappingResult prev;
  const auto static_prev  = cudax::coop::group_by{cuda::std::cw<16>}.map(cuda::gpu_thread, parent, prev);
  const auto dynamic_prev = cudax::coop::group_by{16}.map(cuda::gpu_thread, parent, prev);
  const unsigned counts[]{4, last_count};

  test_runtime_composition<Exhaustive, 4>(config, cuda::std::span{counts}, static_prev);
  test_runtime_composition<Exhaustive, cuda::std::dynamic_extent>(config, cuda::std::span{counts}, dynamic_prev);
  test_runtime_composition<Exhaustive, cuda::std::dynamic_extent>(config, cuda::std::span{counts, 2}, static_prev);
  test_runtime_composition<Exhaustive, cuda::std::dynamic_extent>(config, cuda::std::span{counts, 2}, dynamic_prev);

  if constexpr (!Exhaustive)
  {
    // Excluded threads from the previous mapping must stay excluded.
    const auto partial =
      cudax::coop::take{cuda::std::cw<12>}.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, static_prev);
    const unsigned partial_counts[]{4, 8};
    test_runtime_composition<false, 4>(config, cuda::std::span{partial_counts}, partial);
  }
}

#if _CCCL_STD_VER >= 2020
// Class-valued template arguments allow array partitions to be passed through cw.
inline constexpr cuda::std::array static_counts{4, 12, 16};
inline constexpr cuda::std::array partial_static_counts{4u, 6u, 8u};
#endif // _CCCL_STD_VER >= 2020

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
  static_assert(Result::static_group_rank() == (sizeof...(Counts) == 1 ? 0 : cuda::std::dynamic_extent));

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

  check_mapping_result(counts, result, prev);
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
  static_assert(Repeated::static_group_rank() == cuda::std::dynamic_extent);
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

template <bool Exhaustive, class Config>
__device__ void test_runtime_ranges(Config config)
{
  test_runtime_counts<Exhaustive, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
    config);
  test_runtime_counts<Exhaustive, 2, 4, 8, 16, 2>(config);
  test_runtime_counts<Exhaustive, 3, 5, 1, 1, 22>(config);
  test_runtime_counts<Exhaustive, 31, 1>(config);
  test_runtime_counts<Exhaustive, 32>(config);

  const ForwardCounts full_counts{};
  test_range_input<Exhaustive, 3>(config, full_counts);

  // An adaptor's runtime size must not be replaced with its base range's static size.
  const unsigned counts[]{16, 16, 0};
  const auto partition = cuda::std::views::take(cuda::std::span{counts}, 2);
  test_range_input<Exhaustive, cuda::std::dynamic_extent>(config, partition);

  const unsigned const_counts[]{4, 12, 16};
  const ConstSizedCounts const_partition{const_counts};
  static_assert(cuda::std::ranges::__static_size_v<ConstSizedCounts> == 2);
  static_assert(cuda::std::ranges::__static_size_v<const ConstSizedCounts> == 3);
  test_range_input<Exhaustive, 3>(config, const_partition);

  if constexpr (!Exhaustive)
  {
    test_runtime_counts<false, 31>(config);
    test_runtime_counts<false, 4, 6, 8>(config);
    test_runtime_counts<false, 2, 2, 3, 1, 14>(config);
    const ForwardCounts partial_counts{{4, 6, 8}};
    test_range_input<false, 3>(config, partial_counts);
  }
}

template <class Config>
__device__ void test_compile_time_ranges(Config config)
{
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
  test_static_group_as<true>(config, cuda::std::cw<ForwardCounts{}>, FullPartition{});
  test_static_group_as<false>(config, cuda::std::cw<partial_static_counts>, PartialPartition{});
#endif // _CCCL_STD_VER >= 2020
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_runtime_ranges<true>(config);
    test_runtime_ranges<false>(config);
    test_compile_time_ranges(config);
    test_runtime_composition<true>(config, 12);
    test_runtime_composition<false>(config, 8);
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
