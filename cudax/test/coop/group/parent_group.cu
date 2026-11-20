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
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

namespace
{
template <class Group, class Parent>
__device__ void test_queries(const Group& group, const Parent& parent)
{
  constexpr auto parent_group = cudax::coop::parent_group;
  using MappingResult         = typename Group::__mapping_result_type;

  // Static queries preserve their result types and compile-time values.
  static_assert(cuda::std::is_same_v<decltype(Group::static_count(parent_group)), cuda::std::size_t>);
  static_assert(cuda::std::is_same_v<decltype(Group::static_rank(parent_group)), cuda::std::size_t>);
  static_assert(Group::static_count(parent_group) == MappingResult::static_group_count());
  static_assert(Group::static_rank(parent_group) == MappingResult::static_group_rank());

  // Runtime queries match queries using the actual parent object.
  static_assert(cuda::std::is_same_v<decltype(group.count(parent_group)), decltype(group.count(parent))>);
  static_assert(cuda::std::is_same_v<decltype(group.rank(parent_group)), decltype(group.rank(parent))>);
  REQUIRE(group.count(parent_group) == group.count(parent));
  REQUIRE(group.rank(parent_group) == group.rank(parent));

  // Explicit result types work for both unsigned and signed integers.
  const auto count = group.template count_as<cuda::std::uint64_t>(parent_group);
  const auto rank  = group.template rank_as<int>(parent_group);
  static_assert(cuda::std::is_same_v<decltype(count), const cuda::std::uint64_t>);
  static_assert(cuda::std::is_same_v<decltype(rank), const int>);
  REQUIRE(count == group.template count_as<cuda::std::uint64_t>(parent));
  REQUIRE(rank == group.template rank_as<int>(parent));
}

template <class Group, class Parent>
__device__ void test_group_and_views(const Group& group, const Parent& parent)
{
  test_queries(group, parent);

  const cudax::coop::group_view view{group};
  test_queries(view, parent);

  // Changing the view's unit does not change its parent.
  const cudax::coop::group_view thread_view{cuda::gpu_thread, view};
  test_queries(thread_view, parent);

  // Copying a view preserves the underlying group type.
  const cudax::coop::group_view copied_view{view};
  static_assert(cuda::std::is_same_v<decltype(view), decltype(copied_view)>);
  test_queries(copied_view, parent);
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config, unsigned units_per_group) const
  {
    const cudax::coop::this_block block{config};
    const cudax::coop::this_warp warp{config};

    // Virtual groups with static and dynamic subgroup sizes.
    {
      const cudax::coop::virtual_group group{cuda::warp, block, cudax::coop::group_by{cuda::std::cw<2>}};
      test_group_and_views(group, block);
    }
    {
      const cudax::coop::virtual_group group{cuda::warp, block, cudax::coop::group_by{units_per_group}};
      test_group_and_views(group, block);
    }

    // A single derived group has both a static rank and a static count.
    {
      const cudax::coop::virtual_group group{cuda::gpu_thread, warp, cudax::coop::identity_mapping{}};
      static_assert(group.static_count(cudax::coop::parent_group) == 1);
      static_assert(group.static_rank(cudax::coop::parent_group) == 0);
      test_group_and_views(group, warp);
    }

    // Generic groups with static and dynamic subgroup sizes.
    const cudax::coop::generic_group parent{
      cuda::gpu_thread, warp, cudax::coop::group_by{cuda::std::cw<8>}, cudax::coop::lane_synchronizer{}};
    test_group_and_views(parent, warp);
    {
      const cudax::coop::generic_group group{
        cuda::gpu_thread, warp, cudax::coop::group_by{units_per_group}, cudax::coop::lane_synchronizer{}};
      test_group_and_views(group, warp);
    }

    // Stacked groups refer to their immediate parent.
    {
      const cudax::coop::virtual_group child{cuda::gpu_thread, parent, cudax::coop::group_by{cuda::std::cw<2>}};
      test_group_and_views(child, parent);
    }

    // Constructing a child from a view preserves the underlying parent.
    {
      const cudax::coop::group_view parent_view{parent};
      const cudax::coop::generic_group child{
        cuda::gpu_thread, parent_view, cudax::coop::group_by{cuda::std::cw<2>}, cudax::coop::lane_synchronizer{}};
      test_group_and_views(child, parent);
    }
  }
};
} // namespace

C2H_TEST("Parent group placeholder", "[parent_group]")
{
  const auto device = cuda::devices[0];
  const cuda::stream stream{device};
  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>());
  cuda::launch(stream, config, TestKernel{}, 2u);
  stream.sync();
}
