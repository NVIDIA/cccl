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
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

template <class Group, class View>
__device__ void check_mapping(const Group& group, const View& view)
{
  using Original = typename Group::__mapping_result_type;
  using Mapping  = typename View::__mapping_result_type;
  using Unit     = typename Group::unit_type;

  // Specializing the unit count must preserve the hierarchy and mapping guarantees.
  static_assert(cuda::std::is_same_v<typename View::unit_type, Unit>);
  static_assert(cuda::std::is_same_v<typename View::hierarchy_type, typename Group::hierarchy_type>);
  static_assert(Mapping::static_group_count() == Original::static_group_count());
  static_assert(Mapping::is_always_exhaustive() == Original::is_always_exhaustive());
  static_assert(View::is_always_contiguous() == Group::is_always_contiguous());

  // Group membership and ranks must agree with the source group at runtime.
  const auto original = group.__mapping_result();
  const auto mapping  = view.__mapping_result();

  CHECK(mapping.is_valid());
  CHECK(mapping.group_count() == original.group_count());
  CHECK(mapping.group_rank() == original.group_rank());
  CHECK(mapping.unit_count() == original.unit_count());
  CHECK(mapping.unit_rank() == original.unit_rank());
  CHECK(mapping.lane_mask() == original.lane_mask());

  // Public queries must observe the same mapping as the internal getters.
  CHECK(Unit{}.count(view) == Unit{}.count(group));
  CHECK(Unit{}.rank(view) == Unit{}.rank(group));
}

template <class Group>
__device__ void test_uniform(const Group& group)
{
  using Original = typename Group::__mapping_result_type;
  using Unit     = typename Group::unit_type;
  static_assert(Original::has_uniform_static_unit_count());

  // Uniform counts, including dynamic_extent, need no per-group specialization.
  unsigned calls   = 0;
  const auto count = group.with_static_unit_count([&](auto view) {
    ++calls;

    constexpr auto unit         = Unit{};
    constexpr auto static_count = unit.static_count(view);
    static_assert(static_count == Original::static_unit_count());

    check_mapping(group, view);

    // The view must retain a usable synchronization mechanism.
    view.sync();

    return unit.count(view);
  });

  // Dispatch invokes the callable exactly once and returns its result.
  CHECK(calls == 1);
  CHECK(count == Unit{}.count(group));
}

template <class Config, class Level>
__device__ void test_hardware_group(Config config, Level level)
{
  const auto group = cudax::coop::make_this_group(level, config);
  test_uniform(group);

  // Changing to thread units must preserve any available static count.
  const cudax::coop::group_view threads{cuda::gpu_thread, group};
  test_uniform(threads);
}

struct UniformKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    // Exercise every hardware group, both directly and through a thread view.
    test_hardware_group(config, cuda::gpu_thread);
    test_hardware_group(config, cuda::warp);
    test_hardware_group(config, cuda::block);
    test_hardware_group(config, cuda::cluster);
    test_hardware_group(config, cuda::grid);

    const cudax::coop::this_block parent{config};

    // Virtual groups reuse the parent block synchronizer.
    {
      const cudax::coop::virtual_group group{cuda::warp, parent, cudax::coop::group_by{cuda::std::cw<2>}};
      test_uniform(group);

      // Changing to thread units must preserve any available static count.
      const cudax::coop::group_view threads{cuda::gpu_thread, group};
      test_uniform(threads);
    }

    {
      // A runtime group size must remain dynamic in the callable.
      const cudax::coop::virtual_group group{cuda::warp, parent, cudax::coop::group_by{2}};
      test_uniform(group);
    }

    // Generic groups own a barrier for each of the two groups of warps.
    auto& barriers = get_barriers<2>(cuda::warp);

    {
      const cudax::coop::generic_group group{
        cuda::warp, parent, cudax::coop::group_by{cuda::std::cw<2>}, cudax::coop::barrier_synchronizer{barriers}};
      test_uniform(group);

      // Changing to thread units must preserve any available static count.
      const cudax::coop::group_view threads{cuda::gpu_thread, group};
      test_uniform(threads);
    }

    // Wait for both groups to destroy their barriers before reusing the storage.
    parent.sync();

    // Repeat with a runtime unit count and the same synchronization mechanism.
    {
      const cudax::coop::generic_group group{
        cuda::warp, parent, cudax::coop::group_by{2}, cudax::coop::barrier_synchronizer{barriers}};
      test_uniform(group);
    }
  }
};

C2H_TEST("Static unit count dispatch for uniform groups", "[with_static_unit_count]")
{
  const auto device = cuda::devices[0];
  const cuda::stream stream{device};

  // Static launch dimensions allow thread views to retain compile-time counts.
  cuda::launch(stream,
               cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{}),
               UniformKernel{});

  // Runtime block dimensions exercise the dynamic fallback for thread views.
  cuda::launch(stream,
               cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims(128), cuda::cooperative_launch{}),
               UniformKernel{});

  // Also exercise hardware cluster synchronization when supported.
  if (cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    cuda::launch(stream,
                 cuda::make_config(
                   cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<128>(), cuda::cooperative_launch{}),
                 UniformKernel{});
  }

  stream.sync();
}

// Split four warps into groups of one, two, and one warp, exercising all dispatch
// positions and two groups that share the same static unit count.
template <bool Exhaustive, bool Contiguous>
struct UnequalStaticCounts
{
  template <class Unit, class Parent, class Prev>
  __device__ auto map(const Unit&, const Parent&, const Prev& prev) const
  {
    using Mapping   = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<1, 2, 1>, Exhaustive, Contiguous>;
    const auto rank = prev.unit_rank();

    return Mapping{rank == 0 ? 0u : (rank == 3 ? 2u : 1u), rank == 2 ? 1u : 0u, prev.lane_mask()};
  }
};

template <class Group>
__device__ void test_nonuniform(const Group& group, unsigned units_per_warp)
{
  using Original = typename Group::__mapping_result_type;
  using Unit     = typename Group::unit_type;
  static_assert(!Original::has_uniform_static_unit_count());

  // The runtime count identifies which compile-time specialization must be selected.
  const auto expected = group.__mapping_result().unit_count();
  unsigned calls      = 0;
  auto callback       = [&](auto view) -> unsigned& {
    ++calls;

    constexpr auto unit  = Unit{};
    constexpr auto count = unit.static_count(view);
    static_assert(count == Original::static_unit_count(0) || count == Original::static_unit_count(1));
    CHECK(count == expected);

    check_mapping(group, view);

    // Copy deduction must retain the specialized view type.
    const cudax::coop::group_view copy{view};
    static_assert(cuda::std::is_same_v<decltype(view), cuda::std::remove_cv_t<decltype(copy)>>);
    static_assert(unit.static_count(copy) == count);

    // Converting the specialized view to threads must scale its static count.
    const cudax::coop::group_view threads{cuda::gpu_thread, view};
    constexpr auto thread_count = cuda::gpu_thread.static_count(threads);
    CHECK(cuda::gpu_thread.count(threads) == expected * 32 / units_per_warp);
    CHECK(thread_count == expected * 32 / units_per_warp);
    CHECK(cuda::gpu_thread.rank(threads) == cuda::gpu_thread.rank(group));

    // Dispatching an already specialized view must keep the selected count.
    view.with_static_unit_count([&](auto again) {
      static_assert(Unit{}.static_count(again) == count);
      check_mapping(group, again);
    });

    return calls;
  };

  // An lvalue callable and its reference return must pass through without copying.
  static_assert(cuda::std::is_same_v<decltype(group.with_static_unit_count(callback)), unsigned&>);
  CHECK(&group.with_static_unit_count(callback) == &calls);
  CHECK(calls == 1);
}

template <bool Exhaustive, bool Contiguous, class Config>
__device__ void test_nonuniform_groups(Config config)
{
  const cudax::coop::this_block parent{config};

  // Check nonuniform mappings while retaining the parent block synchronizer.
  const cudax::coop::virtual_group virtual_group{cuda::warp, parent, UnequalStaticCounts<Exhaustive, Contiguous>{}};
  test_nonuniform(virtual_group, 1);
  CHECK(virtual_group.count(parent) == 3);

  const cudax::coop::group_view threads{cuda::gpu_thread, virtual_group};
  test_nonuniform(threads, 32);

  // Repeat with independent barriers for the three groups.
  auto& barriers = get_barriers<3>(cuda::warp);
  const cudax::coop::generic_group group{
    cuda::warp, parent, UnequalStaticCounts<Exhaustive, Contiguous>{}, cudax::coop::barrier_synchronizer{barriers}};
  test_nonuniform(group, 1);
  CHECK(group.count(parent) == 3);

  const cudax::coop::group_view group_threads{cuda::gpu_thread, group};
  test_nonuniform(group_threads, 32);

  // Super-group queries and synchronization must survive specialization too.
  group.with_static_unit_count([&](auto view) {
    CHECK(view.count(parent) == group.count(parent));
    CHECK(view.rank(parent) == group.rank(parent));
    view.sync();
  });
}

struct NonuniformKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    // Preserve both strong and conservative mapping guarantees.
    test_nonuniform_groups<true, true>(config);

    // All groups must finish before the next case reuses their barrier storage.
    cudax::coop::this_block{config}.sync();

    test_nonuniform_groups<false, false>(config);
  }
};

C2H_TEST("Static unit count dispatch for nonuniform groups", "[with_static_unit_count]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>()), NonuniformKernel{});
  stream.sync();
}

// This callable cannot be copied and can only be invoked as an rvalue.
struct ForwardedCallable
{
  unsigned& calls;

  __device__ explicit ForwardedCallable(unsigned& count)
      : calls{count}
  {}

  ForwardedCallable(const ForwardedCallable&) = delete;

  template <class Group>
  __device__ unsigned& operator()(Group group) &&
  {
    CHECK(cuda::gpu_thread.count(group) > 0);
    ++calls;

    return calls;
  }
};

struct CoalescedKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    const auto rank = cuda::gpu_thread.rank(cuda::warp);

    // A noncontiguous active subset still has a dynamic static unit count.
    if (rank % 2 == 0)
    {
      const cudax::coop::coalesced_group group{config};
      static_assert(cuda::gpu_thread.static_count(group) == cuda::std::dynamic_extent);
      test_uniform(group);

      const cudax::coop::group_view view{group};
      test_uniform(view);

      // Forward an rvalue callable and preserve the identity of its reference result.
      unsigned calls = 0;
      ForwardedCallable callable{calls};
      static_assert(cuda::std::is_same_v<decltype(group.with_static_unit_count(cuda::std::move(callable))), unsigned&>);

      unsigned& result = group.with_static_unit_count(cuda::std::move(callable));
      CHECK(&result == &calls);
      CHECK(calls == 1);
    }
  }
};

C2H_TEST("Static unit count dispatch for coalesced groups and callable forwarding", "[with_static_unit_count]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>()), CoalescedKernel{});
  stream.sync();
}
