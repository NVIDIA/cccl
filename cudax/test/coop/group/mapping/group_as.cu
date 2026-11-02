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
template <cuda::std::size_t NGroups, class Config, class Range>
__device__ void test_group_as(Config config, Range&& range)
{
  using Mapping = cudax::coop::group_as<cuda::std::views::all_t<Range>>;

  cuda::std::uint32_t group_starts[NGroups];
  cuda::std::exclusive_scan(cuda::std::ranges::begin(range), cuda::std::ranges::end(range), group_starts, 0u);

  // Test default constructor.
  static_assert(!cuda::std::is_default_constructible_v<Mapping>);

  // Test the mapping is constructible from the Ns array.
  {
    static_assert(cuda::std::is_constructible_v<Mapping, Range>);

    // NOLINTNEXTLINE(misc-const-correctness): decltype must not be const-qualified
    cudax::coop::group_as mapping{range};
    static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
  }

  // Test map(...).
  {
    const cudax::coop::this_warp parent_group{config};
    const ThreadsInWarpMappingResult prev_mapping_result;

    static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                    cuda::gpu_thread, parent_group, prev_mapping_result))>);
    static_assert(
      noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group, prev_mapping_result)));

    const Mapping mapping{range};
    auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
    using Result = decltype(result);

    const auto rank_in_warp = cuda::gpu_thread.rank_as<unsigned>(parent_group);

    unsigned group_rank_ref = NGroups - 1;
    unsigned rank_ref       = rank_in_warp - group_starts[group_rank_ref];
    for (unsigned i = 1; i < NGroups; ++i)
    {
      if (rank_in_warp < group_starts[i])
      {
        group_rank_ref = i - 1;
        rank_ref       = rank_in_warp - group_starts[i - 1];
        break;
      }
    }

    static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
    CHECK(result.group_count() == NGroups);
    CHECK(result.group_rank() == group_rank_ref);

    static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
    CHECK(result.unit_count() == *cuda::std::ranges::next(cuda::std::ranges::begin(range), group_rank_ref));
    CHECK(result.unit_rank() == rank_ref);

    const auto lane_mask_ref =
      (result.unit_count() < 32) ? ((1u << result.unit_count()) - 1) << group_starts[group_rank_ref] : ~0;
    CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});

    CHECK(result.is_valid());
    static_assert(Result::is_always_exhaustive());
    static_assert(Result::is_always_contiguous());
  }
}

template <cuda::std::uint32_t... Ns, class Config>
__device__ void test_group_as(Config config)
{
  constexpr auto NGroups = sizeof...(Ns);

  const cuda::std::uint32_t c_array[]{Ns...};
  const cuda::std::array array{Ns...};
  test_group_as<NGroups>(config, c_array);
  test_group_as<NGroups>(config, array);
  test_group_as<NGroups>(config, cuda::std::span{c_array, NGroups});
}

template <cuda::std::size_t NGroups, class Config, class Range>
__device__ void test_group_as_non_exhaustive(Config config, Range&& range)
{
  cuda::std::uint32_t group_starts[NGroups];
  cuda::std::exclusive_scan(cuda::std::ranges::begin(range), cuda::std::ranges::end(range), group_starts, 0u);

  const auto ns_sum = cuda::std::accumulate(cuda::std::ranges::begin(range), cuda::std::ranges::end(range), 0u);

  using Mapping = cudax::coop::group_as<cuda::std::views::all_t<Range>>;

  // Test default constructor.
  static_assert(!cuda::std::is_default_constructible_v<Mapping>);

  // Test the mapping is constructible from the Ns array.
  {
    static_assert(cuda::std::is_constructible_v<Mapping, Range>);

    // NOLINTNEXTLINE(misc-const-correctness): decltype must not be const-qualified
    cudax::coop::group_as mapping{range};
    static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
  }

  // Test map(...).
  {
    const cudax::coop::this_warp parent_group{config};
    const ThreadsInWarpMappingResult prev_mapping_result;

    static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                    cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result))>);
    static_assert(noexcept(cuda::std::declval<const Mapping>().map(
      cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result)));

    const Mapping mapping{range};
    auto result  = mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result);
    using Result = decltype(result);

    const auto rank_in_warp = cuda::gpu_thread.rank(parent_group);
    const auto is_valid_ref = (rank_in_warp < ns_sum);

    static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
    static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
    static_assert(!Result::is_always_exhaustive());
    static_assert(Result::is_always_contiguous());

    CHECK(result.is_valid() == is_valid_ref);

    if (is_valid_ref)
    {
      unsigned group_rank_ref = NGroups - 1;
      unsigned rank_ref       = rank_in_warp - group_starts[group_rank_ref];
      for (unsigned i = 1; i < NGroups; ++i)
      {
        if (rank_in_warp < group_starts[i])
        {
          group_rank_ref = i - 1;
          rank_ref       = rank_in_warp - group_starts[i - 1];
          break;
        }
      }

      CHECK(result.group_count() == NGroups);
      CHECK(result.group_rank() == group_rank_ref);

      CHECK(result.unit_count() == *cuda::std::ranges::next(cuda::std::ranges::begin(range), group_rank_ref));
      CHECK(result.unit_rank() == rank_ref);

      const auto lane_mask_ref =
        (result.unit_count() < 32) ? ((1u << result.unit_count()) - 1) << group_starts[group_rank_ref] : ~0;
      CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});
    }
  }
}

template <cuda::std::uint32_t... Ns, class Config>
__device__ void test_group_as_non_exhaustive(Config config)
{
  constexpr auto NGroups = sizeof...(Ns);

  const cuda::std::uint32_t c_array[]{Ns...};
  const cuda::std::array array{Ns...};
  test_group_as_non_exhaustive<NGroups>(config, c_array);
  test_group_as_non_exhaustive<NGroups>(config, array);
  test_group_as_non_exhaustive<NGroups>(config, cuda::std::span{c_array, NGroups});
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_group_as<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
      config);
    test_group_as<2, 4, 8, 16, 2>(config);
    test_group_as<3, 5, 1, 1, 22>(config);
    test_group_as<31, 1>(config);
    test_group_as<32>(config);

    test_group_as_non_exhaustive<1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1>(
      config);
    test_group_as_non_exhaustive<2, 4, 8, 16, 2>(config);
    test_group_as_non_exhaustive<3, 5, 1, 1, 22>(config);
    test_group_as_non_exhaustive<31, 1>(config);
    test_group_as_non_exhaustive<32>(config);
    test_group_as_non_exhaustive<31>(config);
    test_group_as_non_exhaustive<4, 6, 8>(config);
    test_group_as_non_exhaustive<2, 2, 3, 1, 14>(config);
  }
};
} // namespace

C2H_TEST("Group-as mapping", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

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
