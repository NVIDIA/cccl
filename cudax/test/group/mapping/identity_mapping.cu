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
template <class Config>
__device__ void test_identity_mapping(Config config)
{
  using Mapping = cudax::identity_mapping;

  // Test default constructor.
  {
    static_assert(cuda::std::is_trivially_default_constructible_v<Mapping>);
    static_assert(cuda::std::is_empty_v<Mapping>);

    [[maybe_unused]] cudax::identity_mapping mapping;
  }

  // Test map(...).
  {
    const cudax::this_warp parent_group{config};
    const ThreadsInWarpMappingResult prev_mapping_result;

    static_assert(cudax::__group_mapping_result<decltype(cuda::std::declval<const Mapping>().map(
                    parent_group, prev_mapping_result))>);
    static_assert(noexcept(cuda::std::declval<const Mapping>().map(parent_group, prev_mapping_result)));

    const Mapping mapping;
    auto result  = mapping.map(parent_group, prev_mapping_result);
    using Result = decltype(result);

    static_assert(cuda::std::is_same_v<Result, ThreadsInWarpMappingResult>);

    static_assert(Result::static_group_count() == ThreadsInWarpMappingResult::static_group_count());
    CUDAX_CHECK(result.group_count() == prev_mapping_result.group_count());
    CUDAX_CHECK(result.group_rank() == prev_mapping_result.group_rank());

    static_assert(Result::static_count() == ThreadsInWarpMappingResult::static_count());
    CUDAX_CHECK(result.count() == prev_mapping_result.count());
    CUDAX_CHECK(result.rank() == prev_mapping_result.rank());

    CUDAX_CHECK(result.is_valid() == prev_mapping_result.is_valid());
    static_assert(Result::is_always_exhaustive() == ThreadsInWarpMappingResult::is_always_exhaustive());
    static_assert(Result::is_always_contiguous() == ThreadsInWarpMappingResult::is_always_contiguous());
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_identity_mapping(config);
    test_identity_mapping(config);
    test_identity_mapping(config);
    test_identity_mapping(config);
    test_identity_mapping(config);
  }
};
} // namespace

C2H_TEST("Identity mapping", "[group]")
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
