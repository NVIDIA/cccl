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
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

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

template <class Group, class View>
__device__ void check_mapping(const Group& group, const View& view)
{
  using Original = typename Group::__mapping_result_type;
  using Mapping  = typename View::__mapping_result_type;
  using Unit     = typename Group::unit_level;

  // Specializing the unit count must preserve the hierarchy and mapping guarantees.
  static_assert(cuda::std::is_same_v<typename View::unit_level, Unit>);
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

template <class Group, class Policy>
__device__ void test_uniform(const Group& group, Policy policy)
{
  using Original = typename Group::__mapping_result_type;
  using Unit     = typename Group::unit_level;
  static_assert(Original::has_uniform_static_unit_count());

  // Unit count specialization preserves uniform counts, including dynamic_extent.
  unsigned calls   = 0;
  const auto count = group.dispatch(policy, [&](auto view) {
    ++calls;

    constexpr auto unit         = Unit{};
    constexpr auto static_count = unit.static_count(view);
    static_assert(static_count == Original::static_unit_count());

    using Mapping = typename decltype(view)::__mapping_result_type;
    if constexpr (cuda::std::is_same_v<Policy, cuda::std::remove_cv_t<decltype(cudax::coop::by_group_rank)>>
                  && Original::static_group_count() != cuda::std::dynamic_extent)
    {
      static_assert(Mapping::static_group_rank() < Original::static_group_count());
      CHECK(Mapping::static_group_rank() == group.__mapping_result().group_rank());
    }
    else
    {
      static_assert(Mapping::static_group_rank() == Original::static_group_rank());
    }

    check_mapping(group, view);

    // The view must retain a usable synchronization mechanism.
    view.sync();

    return unit.count(view);
  });

  // Dispatch invokes the callable exactly once and returns its result.
  CHECK(calls == 1);
  CHECK(count == Unit{}.count(group));
}

template <class Group>
__device__ void test_uniform(const Group& group)
{
  test_uniform(group, cudax::coop::by_group_rank);
  test_uniform(group, cudax::coop::by_unit_count);
}

template <class Config, class Level>
__device__ void test_hardware_group(Config config, Level level)
{
  const auto group = cudax::coop::make_this_group(level, config);
  static_assert(group.static_rank(level) == 0);
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
    const cudax::coop::group_as static_mapping{cuda::std::integer_sequence<int, 2, 2>{}};

    // Virtual groups reuse the parent block synchronizer.
    {
      const cudax::coop::virtual_group group{cuda::warp, parent, static_mapping};
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

    {
      // A statically known number of groups allows rank specialization even when their sizes are dynamic.
      const cuda::std::array counts{2, 2};
      const cudax::coop::virtual_group group{cuda::warp, parent, cudax::coop::group_as{counts}};
      static_assert(group.static_count(cudax::coop::parent_group) == 2);
      static_assert(cuda::warp.static_count(group) == cuda::std::dynamic_extent);
      test_uniform(group);
    }

    // Generic groups own a barrier for each of the two groups of warps.
    auto& barriers = get_barriers<2>(cuda::warp);

    {
      const cudax::coop::generic_group group{
        cuda::warp, parent, static_mapping, cudax::coop::barrier_synchronizer{barriers}};
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

C2H_TEST("Dispatch for uniform groups", "[dispatch]")
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

template <class Group, class Parent, class Policy>
__device__ void test_nonuniform(const Group& group, const Parent& parent, unsigned units_per_warp, Policy policy)
{
  using Original = typename Group::__mapping_result_type;
  using Unit     = typename Group::unit_level;
  static_assert(!Original::has_uniform_static_unit_count());

  // The runtime count identifies which compile-time specialization must be selected.
  const auto expected = group.__mapping_result().unit_count();
  unsigned calls      = 0;

  // Capture the address to avoid older NVCC's false dangling-reference diagnostic.
  auto callback = [&, calls_ptr = &calls](auto view) -> unsigned& {
    ++*calls_ptr;

    constexpr auto unit  = Unit{};
    constexpr auto count = unit.static_count(view);
    static_assert(count == Original::static_unit_count(0) || count == Original::static_unit_count(1));
    CHECK(count == expected);

    using Mapping = typename decltype(view)::__mapping_result_type;
    static_assert(cudax::coop::__group_mapping_result<Mapping>);

    if constexpr (cuda::std::is_same_v<Policy, cuda::std::remove_cv_t<decltype(cudax::coop::by_group_rank)>>)
    {
      constexpr auto group_rank = view.static_rank(cudax::coop::parent_group);
      static_assert(count == Original::static_unit_count(group_rank));
      CHECK(group_rank == group.rank(parent));
      CHECK(view.rank(parent) == group_rank);
      CHECK(view.template rank_as<unsigned>(parent) == group_rank);
    }
    else
    {
      static_assert(view.static_rank(cudax::coop::parent_group) == cuda::std::dynamic_extent);
      CHECK(view.rank(parent) == group.rank(parent));
      CHECK(view.template rank_as<unsigned>(parent) == group.rank(parent));
    }

    check_mapping(group, view);

    // Copy deduction must retain the specialized view type.
    const cudax::coop::group_view copy{view};
    static_assert(cuda::std::is_same_v<decltype(view), cuda::std::remove_cv_t<decltype(copy)>>);
    static_assert(unit.static_count(copy) == count);

    // Converting the specialized view to threads scales its static count and preserves its group rank.
    const cudax::coop::group_view threads{cuda::gpu_thread, view};
    static_assert(threads.static_rank(cudax::coop::parent_group) == view.static_rank(cudax::coop::parent_group));
    constexpr auto thread_count = cuda::gpu_thread.static_count(threads);
    CHECK(cuda::gpu_thread.count(threads) == expected * 32 / units_per_warp);
    CHECK(thread_count == expected * 32 / units_per_warp);
    CHECK(cuda::gpu_thread.rank(threads) == cuda::gpu_thread.rank(group));

    // Repeated specialization and changing the specialization key preserve the view's values.
    view.dispatch(cudax::coop::by_unit_count, [&](auto again) {
      static_assert(cuda::std::is_same_v<decltype(view), decltype(again)>);
      static_assert(Unit{}.static_count(again) == count);
      static_assert(decltype(again)::static_rank(cudax::coop::parent_group)
                    == decltype(view)::static_rank(cudax::coop::parent_group));
      check_mapping(group, again);
    });
    view.dispatch(cudax::coop::by_group_rank, [&](auto again) {
      if constexpr (cuda::std::is_same_v<Policy, cuda::std::remove_cv_t<decltype(cudax::coop::by_group_rank)>>)
      {
        static_assert(cuda::std::is_same_v<decltype(view), decltype(again)>);
      }
      static_assert(Unit{}.static_count(again) == count);
      static_assert(again.static_rank(cudax::coop::parent_group) < Original::static_group_count());
      CHECK(again.static_rank(cudax::coop::parent_group) == group.rank(parent));
      check_mapping(group, again);
    });

    return *calls_ptr;
  };

  // An lvalue callable and its reference return must pass through without copying.
  static_assert(cuda::std::is_same_v<decltype(group.dispatch(policy, callback)), unsigned&>);
  CHECK(&group.dispatch(policy, callback) == &calls);
  CHECK(calls == 1);

  ForwardedCallable forwarded{calls};
  static_assert(cuda::std::is_same_v<decltype(group.dispatch(policy, cuda::std::move(forwarded))), unsigned&>);
  CHECK(&group.dispatch(policy, cuda::std::move(forwarded)) == &calls);
  CHECK(calls == 2);
}

template <class Group, class Parent>
__device__ void test_nonuniform(const Group& group, const Parent& parent, unsigned units_per_warp)
{
  test_nonuniform(group, parent, units_per_warp, cudax::coop::by_group_rank);
  test_nonuniform(group, parent, units_per_warp, cudax::coop::by_unit_count);
}

template <class View>
__device__ const void* view_type_id()
{
  __shared__ int identity;
  return &identity;
}

template <class Group, class Parent, class Policy>
__device__ void test_dispatch_types(const Group& group, const Parent& parent, Policy policy)
{
  __shared__ const void* identities[3];

  group.dispatch(policy, [&](auto view) {
    if (cuda::gpu_thread.rank(view) == 0)
    {
      identities[view.rank(parent)] = view_type_id<decltype(view)>();
    }
  });

  parent.sync();
  if constexpr (cuda::std::is_same_v<Policy, cuda::std::remove_cv_t<decltype(cudax::coop::by_group_rank)>>)
  {
    CHECK(identities[0] != identities[2]);
  }
  else
  {
    CHECK(identities[0] == identities[2]);
  }
  CHECK(identities[0] != identities[1]);
  CHECK(identities[1] != identities[2]);
  parent.sync();
}

// Split four warps into groups of one, two, and one warp, exercising all dispatch
// positions and two groups that share the same static unit count.
template <class Config, class Counts>
__device__ void test_nonuniform_groups(Config config, Counts counts)
{
  const cudax::coop::this_block parent{config};
  const cudax::coop::group_as mapping{counts};

  // Check nonuniform mappings while retaining the parent block synchronizer.
  const cudax::coop::virtual_group virtual_group{cuda::warp, parent, mapping};
  test_nonuniform(virtual_group, parent, 1);
  test_dispatch_types(virtual_group, parent, cudax::coop::by_group_rank);
  test_dispatch_types(virtual_group, parent, cudax::coop::by_unit_count);
  virtual_group.dispatch(cudax::coop::by_group_rank, [&](auto view) {
    constexpr auto rank = view.static_rank(cudax::coop::parent_group);
    static_assert(rank < 3);
    CHECK(view.rank(parent) == rank);
    CHECK(view.template rank_as<unsigned>(parent) == rank);

    CHECK(view.rank(parent) == virtual_group.rank(parent));
  });
  CHECK(virtual_group.count(parent) == 3);

  const cudax::coop::group_view threads{cuda::gpu_thread, virtual_group};
  test_nonuniform(threads, parent, 32);

  // Repeat with independent barriers for the three groups.
  auto& barriers = get_barriers<3>(cuda::warp);
  const cudax::coop::generic_group group{cuda::warp, parent, mapping, cudax::coop::barrier_synchronizer{barriers}};
  test_nonuniform(group, parent, 1);
  CHECK(group.count(parent) == 3);

  const cudax::coop::group_view group_threads{cuda::gpu_thread, group};
  test_nonuniform(group_threads, parent, 32);

  // Super-group queries and synchronization must survive specialization too.
  group.dispatch(cudax::coop::by_group_rank, [&](auto view) {
    CHECK(view.count(parent) == group.count(parent));

    constexpr auto rank = view.static_rank(cudax::coop::parent_group);
    static_assert(rank < 3);
    CHECK(view.rank(parent) == rank);
    CHECK(view.template rank_as<unsigned>(parent) == rank);

    CHECK(view.rank(parent) == group.rank(parent));
    view.sync();
  });
}

struct NonuniformKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    // Integer sequences retain per-group static counts in C++17 and newer.
    test_nonuniform_groups(config, cuda::std::integer_sequence<int, 1, 2, 1>{});

#if _CCCL_STD_VER >= 2020
    // All groups must finish before the next case reuses their barrier storage.
    cudax::coop::this_block{config}.sync();

    // Compile-time ranges must provide the same dispatch through cw.
    test_nonuniform_groups(config, cuda::std::cw<cuda::std::array{1, 2, 1}>);
#endif // _CCCL_STD_VER >= 2020
  }
};

C2H_TEST("Dispatch for nonuniform groups", "[dispatch]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>()), NonuniformKernel{});
  stream.sync();
}

struct DistinctCountsKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    const cudax::coop::this_block parent{config};
    const cudax::coop::virtual_group group{
      cuda::warp, parent, cudax::coop::group_as{cuda::std::integer_sequence<int, 3, 1, 2, 1>{}}};
    constexpr unsigned counts[]{3, 1, 2, 1};
    const auto rank = group.rank(parent);
    unsigned calls  = 0;

    // Unordered, repeated sizes exercise all branches of dispatch over three distinct counts.
    const auto result = group.dispatch(cudax::coop::by_unit_count, [&](auto view) {
      ++calls;
      constexpr auto count = cuda::warp.static_count(view);
      static_assert(count == 1 || count == 2 || count == 3);
      static_assert(view.static_rank(cudax::coop::parent_group) == cuda::std::dynamic_extent);
      CHECK(count == counts[rank]);
      CHECK(view.rank(parent) == rank);
      CHECK(view.count(parent) == 4);
      check_mapping(group, view);

      view.dispatch(cudax::coop::by_unit_count, [&](auto again) {
        static_assert(cuda::std::is_same_v<decltype(view), decltype(again)>);
        static_assert(cuda::warp.static_count(again) == count);
        CHECK(again.rank(parent) == rank);
      });

      // Both one-warp groups must promote -1 to double without an intermediate conversion to unsigned.
      if constexpr (count == 1)
      {
        return -1;
      }
      else if constexpr (count == 2)
      {
        return 2.5;
      }
      else
      {
        return 3u;
      }
    });

    static_assert(cuda::std::is_same_v<decltype(result), const double>);
    constexpr double expected[]{3.0, -1.0, 2.5, -1.0};
    CHECK(result == expected[rank]);
    CHECK(calls == 1);
  }
};

C2H_TEST("Dispatch by three distinct unit counts", "[dispatch]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<224>()), DistinctCountsKernel{});
  stream.sync();
}

struct MoveOnlyDispatchResult
{
  unsigned value;

  __device__ explicit MoveOnlyDispatchResult(unsigned result)
      : value{result}
  {}

  MoveOnlyDispatchResult(const MoveOnlyDispatchResult&) = delete;

  __device__ MoveOnlyDispatchResult(MoveOnlyDispatchResult&& other) noexcept
      : value{other.value}
  {}
};

template <class Group, class Policy>
__device__ void test_dispatch_results(const Group& group, Policy policy)
{
  using Unit = typename Group::unit_level;

  // Arithmetic results from different leaves retain the conditional operator's common type.
  unsigned calls        = 0;
  const auto arithmetic = [&](auto view) {
    ++calls;
    constexpr auto count = Unit{}.static_count(view);
    if constexpr (count % 2 == 0)
    {
      return static_cast<int>(count);
    }
    else
    {
      return static_cast<double>(count) + 0.5;
    }
  };
  static_assert(cuda::std::is_same_v<decltype(group.dispatch(policy, arithmetic)), double>);
  const auto count = Unit{}.count(group);
  CHECK(group.dispatch(policy, arithmetic) == static_cast<double>(count) + (count % 2 == 0 ? 0.0 : 0.5));
  CHECK(calls == 1);

  const auto move_only = [&](auto view) {
    ++calls;
    return MoveOnlyDispatchResult{Unit{}.count(view)};
  };
  static_assert(cuda::std::is_same_v<decltype(group.dispatch(policy, move_only)), MoveOnlyDispatchResult>);
  const auto result = group.dispatch(policy, move_only);
  CHECK(result.value == count);
  CHECK(calls == 2);
}

struct ZeroCountsKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    const cudax::coop::this_block parent{config};
    const cudax::coop::virtual_group group{
      cuda::warp, parent, cudax::coop::group_as{cuda::std::integer_sequence<int, 0, 3, 0, 1>{}}};
    const auto rank = group.rank(parent);
    CHECK(rank == 1 || rank == 3);
    CHECK(group.count(parent) == 4);

    unsigned calls = 0;
    group.dispatch(cudax::coop::by_unit_count, [&](auto view) {
      ++calls;
      constexpr auto count = cuda::warp.static_count(view);
      static_assert(count == 0 || count == 1 || count == 3);
      CHECK(count > 0);
      CHECK(count == (rank == 1 ? 3 : 1));
      CHECK(view.rank(parent) == rank);
      check_mapping(group, view);
    });
    CHECK(calls == 1);

    test_dispatch_results(group, cudax::coop::by_group_rank);
    test_dispatch_results(group, cudax::coop::by_unit_count);
  }
};

struct ManyDistinctCountsKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    const cudax::coop::this_block parent{config};
    const cudax::coop::virtual_group group{
      cuda::gpu_thread, parent, cudax::coop::group_as{cuda::std::integer_sequence<int, 8, 7, 6, 5, 4, 3, 2, 1>{}}};
    const auto rank = group.rank(parent);

    unsigned calls = 0;
    group.dispatch(cudax::coop::by_unit_count, [&](auto view) {
      ++calls;
      constexpr auto count = cuda::gpu_thread.static_count(view);
      static_assert(count >= 1 && count <= 8);
      CHECK(count == 8 - rank);
      CHECK(view.rank(parent) == rank);
      CHECK(view.count(parent) == 8);
      check_mapping(group, view);
    });
    CHECK(calls == 1);

    test_dispatch_results(group, cudax::coop::by_group_rank);
    test_dispatch_results(group, cudax::coop::by_unit_count);
  }
};

C2H_TEST("Dispatch preserves empty groups and result types", "[dispatch]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>()), ZeroCountsKernel{});
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<36>()), ManyDistinctCountsKernel{});
  stream.sync();
}

// Dispatches a block of threads grouped by the given unit counts and checks that every group reaches the
// specialization for its own unit count.
template <class Config, unsigned... Counts>
__device__ void test_unit_count_layout(Config config, cuda::std::integer_sequence<unsigned, Counts...> layout)
{
  const cudax::coop::this_block parent{config};
  const cudax::coop::virtual_group group{cuda::gpu_thread, parent, cudax::coop::group_as{layout}};
  constexpr unsigned counts[]{Counts...};
  const auto rank = group.rank(parent);
  __shared__ const void* identities[sizeof...(Counts)];
  unsigned calls = 0;

  const auto selected_count = group.dispatch(cudax::coop::by_unit_count, [&](auto view) {
    ++calls;
    constexpr auto count = cuda::gpu_thread.static_count(view);
    static_assert(((count == Counts) || ...));
    static_assert(view.static_rank(cudax::coop::parent_group) == cuda::std::dynamic_extent);
    CHECK(count == counts[rank]);
    CHECK(view.rank(parent) == rank);
    CHECK(view.count(parent) == sizeof...(Counts));
    check_mapping(group, view);

    if (cuda::gpu_thread.rank(view) == 0)
    {
      identities[rank] = view_type_id<decltype(view)>();
    }
    return static_cast<unsigned>(count);
  });

  CHECK(calls == 1);
  CHECK(selected_count == counts[rank]);

  // Groups share a specialization exactly if their unit counts are equal.
  parent.sync();
  CHECK((identities[rank] == identities[0]) == (counts[rank] == counts[0]));
}

template <class Layout>
struct UnitCountLayoutKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    test_unit_count_layout(config, Layout{});
  }
};

template <unsigned... Counts>
void launch_unit_count_layout(const cuda::stream& stream, cuda::std::integer_sequence<unsigned, Counts...> layout)
{
  cuda::launch(stream,
               cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<(Counts + ...)>()),
               UnitCountLayoutKernel<decltype(layout)>{});
}

// A layout of GroupCount groups whose unit counts cycle through Pattern.
template <cuda::std::size_t GroupCount, unsigned... Pattern>
struct CyclicLayout
{
  // MSVC 19.52 rejects a subscript expanded directly in make's return type.
  template <cuda::std::size_t Rank>
  static constexpr unsigned value_at()
  {
    constexpr unsigned values[]{Pattern...};
    return values[Rank % sizeof...(Pattern)];
  }

  template <cuda::std::size_t... Ranks>
  static auto make(cuda::std::index_sequence<Ranks...>) -> cuda::std::integer_sequence<unsigned, value_at<Ranks>()...>;

  using type = decltype(make(cuda::std::make_index_sequence<GroupCount>{}));
};

template <cuda::std::size_t GroupCount, unsigned... Pattern>
using cyclic_layout = typename CyclicLayout<GroupCount, Pattern...>::type;

C2H_TEST("Dispatch by unit count across repeated group sizes", "[dispatch]")
{
  static_assert(
    cuda::std::is_same_v<cyclic_layout<7, 2, 1, 1>, cuda::std::integer_sequence<unsigned, 2, 1, 1, 2, 1, 1, 2>>);
  const cuda::stream stream{cuda::devices[0]};

  // Adjacent groups with the same size share a specialization, including long runs at either end.
  launch_unit_count_layout(stream, cuda::std::integer_sequence<unsigned, 2, 2, 2, 1, 1, 3>{});
  launch_unit_count_layout(stream, cuda::std::integer_sequence<unsigned, 1, 1, 1, 1, 1, 1, 2, 3, 4, 4, 4, 4, 4, 4>{});

  // The same sizes can also recur at nonadjacent ranks throughout a larger layout.
  launch_unit_count_layout(stream, cyclic_layout<32, 2, 1, 1>{});
  launch_unit_count_layout(stream, cyclic_layout<35, 2, 1, 1>{});
  launch_unit_count_layout(stream, cyclic_layout<19, 3, 1, 2, 1, 2>{});
  launch_unit_count_layout(stream, cyclic_layout<11, 1, 2, 3, 4, 5, 1, 3>{});
  launch_unit_count_layout(stream, cyclic_layout<64, 1, 2, 3, 4, 5>{});
  launch_unit_count_layout(stream, cyclic_layout<65, 1, 2, 3, 4, 5>{});

  stream.sync();
}

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
      static_assert(
        cuda::std::is_same_v<decltype(group.dispatch(cudax::coop::by_group_rank, cuda::std::move(callable))), unsigned&>);

      unsigned& result = group.dispatch(cudax::coop::by_group_rank, cuda::std::move(callable));
      CHECK(&result == &calls);
      CHECK(calls == 1);

      static_assert(
        cuda::std::is_same_v<decltype(group.dispatch(cudax::coop::by_unit_count, cuda::std::move(callable))), unsigned&>);
      unsigned& count_result = group.dispatch(cudax::coop::by_unit_count, cuda::std::move(callable));
      CHECK(&count_result == &calls);
      CHECK(calls == 2);
    }
  }
};

C2H_TEST("Dispatch for coalesced groups and callable forwarding", "[dispatch]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>()), CoalescedKernel{});
  stream.sync();
}
