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
#include <cuda/std/cstddef>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

namespace
{
template <cuda::std::size_t N, class Config>
__device__ void test_take(Config config)
{
  constexpr auto n = static_cast<cuda::std::uint32_t>(N);

  // Test static N.
  {
    using Mapping = cudax::coop::take<N>;

    // Test that the mapping is empty.
    static_assert(cuda::std::is_empty_v<Mapping>);

    // Test default constructor is deleted.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is constructible from uint32_t.
    static_assert(cuda::std::is_nothrow_constructible_v<Mapping, cuda::std::uint32_t>);

    // Test the mapping is constructible and deducible from integral_constant<size_t, N>.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, cuda::std::integral_constant<cuda::std::size_t, N>>);

      cudax::coop::take mapping{cuda::std::integral_constant<cuda::std::size_t, N>{}};
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

      const cudax::coop::take mapping{cuda::std::integral_constant<cuda::std::size_t, N>{}};
      auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == ThreadsInWarpMappingResult::static_group_count());
      static_assert(Result::static_unit_count() == N);
      static_assert(Result::is_always_exhaustive()
                    == (Result::static_unit_count() == ThreadsInWarpMappingResult::static_unit_count()));
      static_assert(Result::is_always_contiguous());

      const auto is_valid_ref = cuda::std::cmp_less(cuda::gpu_thread.rank(cuda::warp), n);
      CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        CHECK(result.group_count() == prev_mapping_result.group_count());
        CHECK(result.group_rank() == prev_mapping_result.group_rank());

        CHECK(result.unit_count() == n);
        CHECK(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp));

        const auto lane_mask_ref = ((N < 32) ? ((1u << N) - 1) : ~0u);
        CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});

        CHECK(result.is_valid());
      }
    }
  }

  // Test dynamic Ns.
  {
    using Mapping = cudax::coop::take<cuda::std::dynamic_extent>;

    // Test default constructor is deleted.
    static_assert(!cuda::std::is_default_constructible_v<Mapping>);

    // Test the mapping is constructible and deducible from uint32_t.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, cuda::std::uint32_t>);

      cudax::coop::take mapping{n};
      static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
    }

    // Test the mapping is constructible from integral_constant<size_t, N>.
    static_assert(cuda::std::is_nothrow_constructible_v<Mapping, cuda::std::integral_constant<cuda::std::size_t, N>>);

    // Test map(...).
    {
      const cudax::coop::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cuda::gpu_thread, parent_group, prev_mapping_result))>);
      static_assert(
        noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group, prev_mapping_result)));

      const Mapping mapping{n};
      auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == ThreadsInWarpMappingResult::static_group_count());
      static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
      static_assert(!Result::is_always_exhaustive());
      static_assert(Result::is_always_contiguous());

      const auto is_valid_ref = cuda::std::cmp_less(cuda::gpu_thread.rank(cuda::warp), n);
      CHECK(result.is_valid() == is_valid_ref);

      if (is_valid_ref)
      {
        CHECK(result.group_count() == prev_mapping_result.group_count());
        CHECK(result.group_rank() == prev_mapping_result.group_rank());

        CHECK(result.unit_count() == n);
        CHECK(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp));

        const auto lane_mask_ref = ((N < 32) ? ((1u << N) - 1) : ~0u);
        CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});

        CHECK(result.is_valid());
      }
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_take<0>(config);
    test_take<1>(config);
    test_take<2>(config);
    test_take<3>(config);
    test_take<4>(config);
    test_take<14>(config);
    test_take<16>(config);
    test_take<30>(config);
    test_take<32>(config);
  }
};
} // namespace

C2H_TEST("Take mapping", "[group]")
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
