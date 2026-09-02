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
#include <cuda/stream>
#include <cuda/type_traits>

#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

#include "testing.cuh"

template <class T>
__device__ T make_instance_for(unsigned rank)
{
  if constexpr (cuda::is_vector_type_v<T>)
  {
    using U = cuda::scalar_type_t<T>;
    return T{static_cast<U>(rank), static_cast<U>(rank), static_cast<U>(rank)};
  }
  else
  {
    return static_cast<T>(rank);
  }
}

template <class T, class Group>
__device__ void test_group(const Group& group)
{
  // Exit all threads that are not part of the group.
  if (!cuda::gpu_thread.is_part_of(group))
  {
    return;
  }

  const auto my_rank  = cuda::gpu_thread.rank_as<unsigned>(group);
  const auto my_value = make_instance_for<T>(my_rank);

  // Test identity.
  REQUIRE(cudax::coop::shuffle_up(group, my_value, 0) == my_value);

  // Test getting value from the previous rank.
  {
    const auto offset     = 1;
    const auto other_rank = cuda::gpu_thread.rank(group) - offset;
    const auto ref        = (other_rank < cuda::gpu_thread.count(group))
                            ? cuda::std::optional{make_instance_for<T>(other_rank)}
                            : cuda::std::nullopt;
    REQUIRE(cudax::coop::shuffle_up(group, my_value, offset) == ref);
  }

  // Test getting value from the this + 2 rank.
  {
    const auto offset     = 2;
    const auto other_rank = cuda::gpu_thread.rank(group) - offset;
    const auto ref        = (other_rank < cuda::gpu_thread.count(group))
                            ? cuda::std::optional{make_instance_for<T>(other_rank)}
                            : cuda::std::nullopt;
    REQUIRE(cudax::coop::shuffle_up(group, my_value, offset) == ref);
  }

  // Test getting value from the first rank.
  {
    const auto other_rank = 0;
    const auto offset     = cuda::gpu_thread.rank(group) - other_rank;
    REQUIRE(cudax::coop::shuffle_up(group, my_value, offset) == make_instance_for<T>(other_rank));
  }

  // Test getting value from the first rank - 1.
  {
    const auto offset = cuda::gpu_thread.rank(group) + 1;
    REQUIRE(cudax::coop::shuffle_up(group, my_value, offset) == cuda::std::nullopt);
  }

  // Test getting value from out of range offset.
  REQUIRE(cudax::coop::shuffle_up(group, my_value, ~0u) == cuda::std::nullopt);
}

struct CustomBinaryPartition
{
  template <class MappingResult>
  __device__ bool operator()(MappingResult mapping_result)
  {
    switch (mapping_result.unit_rank())
    {
      case 1:
      case 5:
      case 14:
      case 15:
      case 31:
        return true;
      default:
        return false;
    }
  }
};

template <class T, class Config>
__device__ void test_type(const Config& config)
{
  const cudax::this_warp warp{config};
  test_group<T>(cudax::group{cuda::gpu_thread, warp, cudax::identity_mapping{}, cudax::lane_synchronizer{}});
  test_group<T>(cudax::group{cuda::gpu_thread, warp, cudax::group_by<4>{}, cudax::lane_synchronizer{}});
  test_group<T>(cudax::group{cuda::gpu_thread, warp, cudax::group_by{1}, cudax::lane_synchronizer{}});
  test_group<T>(
    cudax::group{cuda::gpu_thread, warp, cudax::group_by{3, cudax::non_exhaustive}, cudax::lane_synchronizer{}});
  test_group<T>(
    cudax::group{cuda::gpu_thread, warp, cudax::binary_partition{CustomBinaryPartition{}}, cudax::lane_synchronizer{}});
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_type<signed char>(config);
    test_type<unsigned>(config);
    test_type<unsigned long long>(config);
    test_type<longlong3>(config);
  }
};

C2H_TEST("shuffle/threads_within_warp", "[shuffle][threads_within_warp]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>());
  cuda::launch(stream, config, TestKernel{});

  stream.sync();
}
