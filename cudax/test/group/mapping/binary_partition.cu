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
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
struct AlwaysTruePredFn
{
  template <class MappingResult>
  __device__ bool operator()(MappingResult mapping_result)
  {
    return true;
  }
};

struct AlwaysFalsePredFn
{
  template <class MappingResult>
  __device__ bool operator()(MappingResult mapping_result) noexcept
  {
    return false;
  }
};

struct IsEvenPredFn
{
  template <class MappingResult>
  __device__ bool operator()(MappingResult mapping_result)
  {
    return mapping_result.unit_rank() % 2 == 0;
  }
};

template <class Config>
__device__ void test_binary_partition(Config config)
{
  // Always true predicate.
  {
    using Pred    = AlwaysTruePredFn;
    using Mapping = cudax::binary_partition<Pred>;

    // Test constructor from pred_fn.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, Pred>);
      cudax::binary_partition mapping{Pred{}};
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cuda::gpu_thread, parent_group, prev_mapping_result))>);
      static_assert(
        !noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group, prev_mapping_result)));

      const Mapping mapping{Pred{}};
      auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == 2);
      REQUIRE(result.group_count() == 2);
      REQUIRE(result.group_rank() == 1);

      static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
      REQUIRE(result.unit_count() == cuda::gpu_thread.count(cuda::warp));
      REQUIRE(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp));

      REQUIRE(result.lane_mask() == cuda::device::lane_mask::all());

      REQUIRE(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(!Result::is_always_contiguous());
    }
  }

  // Always false predicate.
  {
    using Pred    = AlwaysFalsePredFn;
    using Mapping = cudax::binary_partition<Pred>;

    // Test constructor from pred_fn.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, Pred>);
      cudax::binary_partition mapping{Pred{}};
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cuda::gpu_thread, parent_group, prev_mapping_result))>);
      static_assert(
        noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group, prev_mapping_result)));

      const Mapping mapping{Pred{}};
      auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == 2);
      REQUIRE(result.group_count() == 2);
      REQUIRE(result.group_rank() == 0);

      static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
      REQUIRE(result.unit_count() == cuda::gpu_thread.count(cuda::warp));
      REQUIRE(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp));

      REQUIRE(result.lane_mask() == cuda::device::lane_mask::all());

      REQUIRE(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(!Result::is_always_contiguous());
    }
  }

  // True for even ranks predicate.
  {
    using Pred    = IsEvenPredFn;
    using Mapping = cudax::binary_partition<Pred>;

    // Test constructor from pred_fn.
    {
      static_assert(cuda::std::is_nothrow_constructible_v<Mapping, Pred>);
      cudax::binary_partition mapping{Pred{}};
    }

    // Test map(...).
    {
      const cudax::this_warp parent_group{config};
      const ThreadsInWarpMappingResult prev_mapping_result;

      static_assert(cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                      cuda::gpu_thread, parent_group, prev_mapping_result))>);
      static_assert(
        !noexcept(cuda::std::declval<const Mapping>().map(cuda::gpu_thread, parent_group, prev_mapping_result)));

      const Mapping mapping{Pred{}};
      auto result  = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);
      using Result = decltype(result);

      static_assert(Result::static_group_count() == 2);
      REQUIRE(result.group_count() == 2);
      REQUIRE(result.group_rank() == (cuda::gpu_thread.rank(cuda::warp) % 2 == 0));

      static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
      REQUIRE(result.unit_count() == cuda::gpu_thread.count(cuda::warp) / 2);
      REQUIRE(result.unit_rank() == cuda::gpu_thread.rank(cuda::warp) / 2);

      const auto lane_mask_ref =
        (cuda::gpu_thread.rank(cuda::warp) % 2 == 0)
          ? cuda::device::lane_mask{0x5555'5555u}
          : cuda::device::lane_mask(0xaaaa'aaaau);
      REQUIRE(result.lane_mask() == lane_mask_ref);

      REQUIRE(result.is_valid());
      static_assert(Result::is_always_exhaustive());
      static_assert(!Result::is_always_contiguous());
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_binary_partition(config);
  }
};
} // namespace

C2H_TEST("Binary partition mapping", "[group]")
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
