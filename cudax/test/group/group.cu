//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "testing.cuh"

namespace
{
__device__ unsigned global_var = 0;

template <class Unit, class Level, class Hierarchy, class Group>
__device__ void test_common_properties(const Hierarchy&, Group& group)
{
  // Assert that Group satisfies the group concept.
  static_assert(cudax::is_group<Group>);

  // Test types
  static_assert(cuda::std::is_same_v<Unit, typename Group::unit_type>);
  static_assert(cuda::std::is_same_v<Level, typename Group::level_type>);

  // Test that the group can be queried for it's hierarchy.
  {
    decltype(auto) hierarchy = cuda::std::as_const(group).hierarchy();
    static_assert(cuda::std::is_same_v<decltype(hierarchy), const Hierarchy&>);
  }

  // Test that the group can be synchronized using .sync() method.
  {
    static_assert(cuda::std::is_same_v<void, decltype(group.sync())>);
    static_assert(noexcept(group.sync()));

    // .sync() method must support calls from different branches. Add some dummy work to make sure the branches are not
    // collided.
    cuda::atomic_ref<unsigned, cuda::thread_scope_device> atomic{global_var};
    if ((threadIdx.x + threadIdx.y + threadIdx.z) % 2 == 0)
    {
      atomic++;
      group.sync();
      atomic--;
    }
    else
    {
      atomic--;
      group.sync();
      atomic++;
    }
  }

  // Test that the group can be synchronized using .sync_aligned() method.
  {
    static_assert(cuda::std::is_same_v<void, decltype(group.sync_aligned())>);
    static_assert(noexcept(group.sync_aligned()));

    // .sync_aligned() method must be called by all threads in the group uniformly in one place.
    group.sync_aligned();
  }
}

template <class ParentGroup, cuda::std::size_t N, class Synchronizer>
__device__ void
test_queries(const cudax::group<cuda::thread_level, ParentGroup, cudax::group_by<N>, Synchronizer>& group)
{
  // todo(dabayer): These queries end up in `error: expression must have a constant value`, when group is taken by
  // reference. Can we find a solution that works without copying the group?
  // static_assert(cuda::gpu_thread.static_count(group) == N);

  using Group = cuda::std::remove_cvref_t<decltype(group)>;
  using Level = typename Group::level_type;

  const auto count_ref = group.mapping().count();
  const auto rank_ref  = cuda::gpu_thread.rank(Level{}, group.hierarchy()) % count_ref;

  CUDAX_REQUIRE(cuda::gpu_thread.count(group) == count_ref);
  CUDAX_REQUIRE(cuda::gpu_thread.rank(group) == rank_ref);
  CUDAX_REQUIRE(cuda::gpu_thread.is_root_rank(group) == (rank_ref == 0));
  CUDAX_REQUIRE(cuda::gpu_thread.is_part_of(group));
}

template <class T>
__device__ T global_barrier_storage;

template <cuda::std::size_t N, class Unit, class Level, class Config>
__device__ void test_group_by_group(const Unit& unit, const Level& level, const Config& config)
{
  constexpr cuda::std::size_t nbarriers = 128 / N;
  constexpr auto use_global_barrier =
    (cuda::std::is_same_v<Level, cuda::cluster_level> || cuda::std::is_same_v<Level, cuda::grid_level>);
  constexpr auto barrier_scope = (use_global_barrier) ? cuda::thread_scope_device : cuda::thread_scope_block;

  using Barrier         = cuda::barrier<barrier_scope>;
  using BarriersStorage = cuda::std::aligned_storage_t<nbarriers * sizeof(Barrier), alignof(Barrier)>;
  __shared__ BarriersStorage shared_barriers_storage;

  auto& barriers = reinterpret_cast<Barrier(&)[nbarriers]>(
    (use_global_barrier) ? global_barrier_storage<BarriersStorage> : shared_barriers_storage);

  auto parent_group = cudax::make_this_group(level, config);
  cudax::barrier_synchronizer synchronizer{barriers};

  {
    cudax::group_by<N> mapping{};
    cudax::group group{unit, parent_group, mapping, synchronizer};

    static_assert(
      cuda::std::is_same_v<cudax::group<Unit, decltype(parent_group), decltype(mapping), decltype(synchronizer)>,
                           decltype(group)>);
    test_common_properties<Unit, Level>(config.hierarchy(), group);
    group.sync();
  }
  {
    cudax::group_by mapping{N};
    cudax::group group{unit, parent_group, mapping, synchronizer};

    static_assert(
      cuda::std::is_same_v<cudax::group<Unit, decltype(parent_group), decltype(mapping), decltype(synchronizer)>,
                           decltype(group)>);
    test_common_properties<Unit, Level>(config.hierarchy(), group);
    group.sync();
  }
}

template <class Unit, class Level, class Config>
__device__ void test_group_by_group(const Unit& unit, const Level& level, const Config& config)
{
  // powers of 2
  test_group_by_group<1>(unit, level, config);
  test_group_by_group<4>(unit, level, config);
  test_group_by_group<16>(unit, level, config);
  test_group_by_group<32>(unit, level, config);

  if constexpr (!cuda::std::is_same_v<Level, cuda::warp_level>)
  {
    test_group_by_group<64>(unit, level, config);
    test_group_by_group<128>(unit, level, config);
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // test_group_by_group(cuda::gpu_thread, cuda::warp, config);
    test_group_by_group(cuda::gpu_thread, cuda::block, config);
    // test_group_by_group(cuda::gpu_thread, cuda::warp, config);
    // test_group_by_group(cuda::gpu_thread, cuda::warp, config);
  }
};
} // namespace

C2H_TEST("Group", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
  cuda::launch(stream, config, TestKernel{});

  if (cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    const auto config_cluster = cuda::make_config(
      cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config_cluster, TestKernel{});
  }

  stream.sync();
}
