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

template <class View>
__device__ void test_view_synchronization(const View& view)
{
  view.sync();
  view.wait(view.arrive());
  view.sync_aligned();
  view.wait_aligned(view.arrive_aligned());
}

template <class View, class Group, class Level>
__device__ void check_group_view(const View& view, const Group& group, Level level)
{
  if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    REQUIRE(cuda::gpu_thread.count(view) == 1);
    REQUIRE(cuda::gpu_thread.rank(view) == 0);
  }
  else
  {
    REQUIRE(cuda::gpu_thread.count(view) == cuda::gpu_thread.count(level));
    REQUIRE(cuda::gpu_thread.rank(view) == cuda::gpu_thread.rank(level));
  }

  static_assert(View::static_count(Level{}) == 1);
  static_assert(View::static_rank(Level{}) == 0);
  REQUIRE(view.count(level) == 1);
  REQUIRE(view.template count_as<unsigned>(level) == 1);
  REQUIRE(view.rank(level) == 0);
  REQUIRE(view.template rank_as<unsigned>(level) == 0);
  static_assert(View::is_always_contiguous() == Group::is_always_contiguous());

  test_view_synchronization(view);
}

template <class Config, class Level>
__device__ void test_group_view(Config config, Level level)
{
  const auto g = cudax::coop::make_this_group(level, config);
  using View   = decltype(cudax::coop::group_view{g});
  static_assert(!cuda::std::is_default_constructible_v<View>);
  static_assert(cuda::std::is_copy_constructible_v<View>);
  static_assert(cuda::std::is_copy_assignable_v<View>);
  static_assert(cuda::std::is_move_assignable_v<View>);

  // Exercise each construction path with the same query and synchronization checks.
  const auto check = [&](const auto& view) {
    check_group_view(view, g, level);
    const cudax::coop::group_view copy{view};
    check_group_view(copy, g, level);
  };

  cudax::coop::group_view view{g};
  check(view);

  View other{g};
  other = view;
  view  = cuda::std::move(other);
  check(view);

  const cudax::coop::group_view copy{view};
  static_assert(cuda::std::is_same_v<cuda::std::remove_cv_t<decltype(copy)>, View>);
  check(copy);

  const cudax::coop::group_view threads{cuda::gpu_thread, g};
  check(threads);
  check(cudax::coop::group_view{cuda::gpu_thread, view});

  if constexpr (!cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    const cudax::coop::group_view warps{cuda::warp, g};
    check(cudax::coop::group_view{cuda::gpu_thread, warps});
  }

  // Test a view of a generic group.
  if constexpr (cuda::std::is_same_v<Level, cuda::block_level>)
  {
    constexpr auto n = 2;
    auto& barriers   = get_barriers<cuda::warp.static_count(level, config) / n>(cuda::warp);

    const cudax::coop::generic_group g2{
      cuda::warp, g, cudax::coop::group_by{cuda::std::cw<n>}, cudax::coop::barrier_synchronizer{barriers}};
    REQUIRE(cuda::warp.count(g2) == n);
    REQUIRE(g2.count(g) == cuda::warp.count(level) / n);

    const cudax::coop::group_view g2_view{g2};
    REQUIRE(cuda::warp.count(g2_view) == n);
    REQUIRE(g2_view.count(g) == cuda::warp.count(level) / n);
    static_assert(g2_view.is_always_contiguous());
    test_view_synchronization(g2_view);

    const cudax::coop::group_view g2_view_threads{cuda::gpu_thread, g2_view};
    REQUIRE(cuda::gpu_thread.count(g2_view_threads) == n * cuda::gpu_thread.count(cuda::warp));
    REQUIRE(g2_view_threads.count(g) == cuda::warp.count(level) / n);
    static_assert(g2_view_threads.is_always_contiguous());
    test_view_synchronization(g2_view_threads);
  }
}

template <class Group, class Parent, class = void>
inline constexpr bool can_static_count_parent = false;

template <class Group, class Parent>
inline constexpr bool can_static_count_parent<
  Group,
  Parent,
  cuda::std::void_t<decltype(cuda::std::declval<const Group&>().static_count(cuda::std::declval<const Parent&>()))>> =
  true;

template <class Group, class Parent, class = void>
inline constexpr bool can_static_rank_parent = false;

template <class Group, class Parent>
inline constexpr bool can_static_rank_parent<
  Group,
  Parent,
  cuda::std::void_t<decltype(cuda::std::declval<const Group&>().static_rank(cuda::std::declval<const Parent&>()))>> =
  true;

template <class Group, class Parent, class = void>
inline constexpr bool can_count_parent = false;

template <class Group, class Parent>
inline constexpr bool can_count_parent<
  Group,
  Parent,
  cuda::std::void_t<decltype(cuda::std::declval<const Group&>().count(cuda::std::declval<const Parent&>()))>> = true;

template <class Group, class Parent, class = void>
inline constexpr bool can_rank_parent = false;

template <class Group, class Parent>
inline constexpr bool can_rank_parent<
  Group,
  Parent,
  cuda::std::void_t<decltype(cuda::std::declval<const Group&>().rank(cuda::std::declval<const Parent&>()))>> = true;

template <class Group, class Parent, class = void>
inline constexpr bool can_count_as_parent = false;

template <class Group, class Parent>
inline constexpr bool
  can_count_as_parent<Group,
                      Parent,
                      cuda::std::void_t<decltype(cuda::std::declval<const Group&>().template count_as<unsigned>(
                        cuda::std::declval<const Parent&>()))>> = true;

template <class Group, class Parent, class = void>
inline constexpr bool can_rank_as_parent = false;

template <class Group, class Parent>
inline constexpr bool
  can_rank_as_parent<Group,
                     Parent,
                     cuda::std::void_t<decltype(cuda::std::declval<const Group&>().template rank_as<unsigned>(
                       cuda::std::declval<const Parent&>()))>> = true;

template <class Group, class Parent>
__device__ void check_invalid_parent_queries()
{
  static_assert(!can_static_count_parent<Group, Parent>);
  static_assert(!can_static_rank_parent<Group, Parent>);
  static_assert(!can_count_parent<Group, Parent>);
  static_assert(!can_rank_parent<Group, Parent>);
  static_assert(!can_count_as_parent<Group, Parent>);
  static_assert(!can_rank_as_parent<Group, Parent>);
}

template <class Group, class Parent>
__device__ void check_parent_view_queries(const Group& group, const Parent& parent)
{
  const auto check = [&](const auto& query_parent) {
    const cudax::coop::group_view view{query_parent};
    static_assert(Group::static_count(view) == Group::__mapping_result_type::static_group_count());
    static_assert(Group::static_rank(view) == Group::__mapping_result_type::static_group_rank());
    static_assert(cuda::std::is_same_v<decltype(group.count(query_parent)), decltype(group.count(parent))>);
    static_assert(cuda::std::is_same_v<decltype(group.rank(query_parent)), decltype(group.rank(parent))>);
    static_assert(
      cuda::std::is_same_v<decltype(group.template count_as<unsigned long long>(query_parent)), unsigned long long>);
    static_assert(
      cuda::std::is_same_v<decltype(group.template rank_as<unsigned long long>(query_parent)), unsigned long long>);
    CHECK(group.count(query_parent) == group.__mapping_result().group_count());
    CHECK(group.template count_as<unsigned long long>(query_parent) == group.__mapping_result().group_count());
    CHECK(group.rank(query_parent) == group.__mapping_result().group_rank());
    CHECK(group.template rank_as<unsigned long long>(query_parent) == group.__mapping_result().group_rank());
  };

  check(parent);
  check(cudax::coop::group_view{parent});
  const cudax::coop::group_view parent_view{parent};
  // Parent views must preserve the parent's unit.
  if constexpr (!cuda::std::is_same_v<typename Parent::unit_level, cuda::thread_level>)
  {
    using ChangedUnit = decltype(cudax::coop::group_view{cuda::gpu_thread, parent_view});
    check_invalid_parent_queries<Group, ChangedUnit>();
  }

  // A child group and its view are not the parent, even when they share a hierarchy.
  using Unrelated = cudax::coop::group_view<typename Group::unit_level, Group, typename Group::__mapping_result_type>;
  check_invalid_parent_queries<Group, Unrelated>();
}

struct ParentViewQueryKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    const cudax::coop::this_block parent{config};
    const cudax::coop::group_view parent_view{parent};
    const cudax::coop::virtual_group group{cuda::warp, parent, cudax::coop::group_by{cuda::std::cw<2>}};
    check_parent_view_queries(group, parent);
    check_parent_view_queries(cudax::coop::group_view{group}, parent);
    check_parent_view_queries(cudax::coop::group_view{cuda::gpu_thread, group}, parent);

    const cudax::coop::virtual_group single{cuda::warp, parent, cudax::coop::identity_mapping{}};
    check_parent_view_queries(single, parent);
    check_parent_view_queries(cudax::coop::group_view{single}, parent);

    // Construction from a view retains the underlying parent type.
    const cudax::coop::virtual_group from_view{cuda::warp, parent_view, cudax::coop::group_by{cuda::std::cw<2>}};
    static_assert(cuda::std::is_same_v<decltype(group), decltype(from_view)>);
    check_parent_view_queries(from_view, parent);
    check_parent_view_queries(cudax::coop::group_view{from_view}, parent);

    const cudax::coop::group_view thread_parent{cuda::gpu_thread, parent_view};
    const cudax::coop::virtual_group thread_group{cuda::gpu_thread, parent, cudax::coop::group_by{cuda::std::cw<32>}};
    const cudax::coop::virtual_group from_thread_view{
      cuda::gpu_thread, thread_parent, cudax::coop::group_by{cuda::std::cw<32>}};
    static_assert(cuda::std::is_same_v<decltype(thread_group), decltype(from_thread_view)>);
    check_parent_view_queries(from_thread_view, parent);
    check_parent_view_queries(cudax::coop::group_view{from_thread_view}, parent);

    auto& barriers = get_barriers<2>(cuda::warp);
    {
      const cudax::coop::generic_group synchronized{
        cuda::warp, parent, cudax::coop::group_by{cuda::std::cw<2>}, cudax::coop::barrier_synchronizer{barriers}};
      check_parent_view_queries(synchronized, parent);
      check_parent_view_queries(cudax::coop::group_view{synchronized}, parent);
      check_parent_view_queries(cudax::coop::group_view{cuda::gpu_thread, synchronized}, parent);

      using FromView = decltype(cudax::coop::generic_group{
        cuda::warp, parent_view, cudax::coop::group_by{cuda::std::cw<2>}, cudax::coop::barrier_synchronizer{barriers}});
      static_assert(cuda::std::is_same_v<cuda::std::remove_cv_t<decltype(synchronized)>, FromView>);
    }
    {
      const cudax::coop::generic_group synchronized{
        cuda::warp, parent_view, cudax::coop::group_by{cuda::std::cw<2>}, cudax::coop::barrier_synchronizer{barriers}};
      check_parent_view_queries(synchronized, parent);
      check_parent_view_queries(cudax::coop::group_view{synchronized}, parent);
    }
    {
      const cudax::coop::generic_group synchronized{
        cuda::gpu_thread,
        thread_parent,
        cudax::coop::group_by{cuda::std::cw<64>},
        cudax::coop::barrier_synchronizer{barriers}};
      using FromParent = decltype(cudax::coop::generic_group{
        cuda::gpu_thread,
        parent,
        cudax::coop::group_by{cuda::std::cw<64>},
        cudax::coop::barrier_synchronizer{barriers}});
      static_assert(cuda::std::is_same_v<cuda::std::remove_cv_t<decltype(synchronized)>, FromParent>);
      check_parent_view_queries(synchronized, parent);
      check_parent_view_queries(cudax::coop::group_view{synchronized}, parent);
    }
  }
};

C2H_TEST("Group queries accept parent views", "[group_view]")
{
  const cuda::stream stream{cuda::devices[0]};
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>()), ParentViewQueryKernel{});
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims(128)), ParentViewQueryKernel{});
  stream.sync();
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    test_group_view(config, cuda::gpu_thread);
    test_group_view(config, cuda::warp);
    test_group_view(config, cuda::block);
    test_group_view(config, cuda::cluster);
    test_group_view(config, cuda::grid);
  }
};

C2H_TEST("Group View", "[group_view]")
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

// Map a four-block grid into two groups containing one and three blocks.
// The flags let the view tests check that conversion preserves the mapping guarantees.
template <bool Exhaustive, bool Contiguous>
struct UnequalStaticCounts
{
  template <class Unit, class Parent, class Prev>
  __device__ auto map(const Unit&, const Parent&, const Prev& prev) const
  {
    using Result    = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<1, 3>, Exhaustive, Contiguous>;
    const auto rank = prev.unit_rank();

    return Result{rank == 0 ? 0u : 1u, rank == 0 ? 0u : rank - 1, prev.lane_mask()};
  }
};

template <bool Static, bool Exhaustive, bool Contiguous, class Config>
__device__ void test_nonuniform_group_view(Config config)
{
  const cudax::coop::this_grid parent{config};
  const cudax::coop::virtual_group group{cuda::block, parent, UnequalStaticCounts<Exhaustive, Contiguous>{}};

  // A view with the same unit type must preserve the original mapping result type.
  const cudax::coop::group_view same_unit{group};
  using Original = typename decltype(group)::__mapping_result_type;

  static_assert(cuda::std::is_same_v<typename decltype(same_unit)::__mapping_result_type, Original>);

  // Expected group membership is independent of the unit used by the view.
  const auto block_rank = cuda::block.rank_as<unsigned>(cuda::grid);
  const auto group_rank = block_rank == 0 ? 0u : 1u;
  const auto blocks     = block_rank == 0 ? 1u : 3u;
  const auto rank       = block_rank == 0 ? 0u : block_rank - 1;

  // All conversions retain the group metadata and scale unit counts and ranks.
  const auto check = [&](const auto& view, auto unit, unsigned units_per_block) {
    const auto result = view.__mapping_result();
    using Result      = decltype(result);

    static_assert(Result::static_group_count() == 2);
    if constexpr (Result::static_group_rank() != cuda::std::dynamic_extent)
    {
      CHECK(Result::static_group_rank() == group_rank);
    }
    static_assert(Result::is_always_exhaustive() == Exhaustive);
    static_assert(Result::is_always_contiguous() == Contiguous);

    CHECK(result.is_valid());
    CHECK(result.group_count() == 2);
    CHECK(result.group_rank() == group_rank);
    CHECK(result.unit_count() == blocks * units_per_block);

    if constexpr (cuda::std::is_same_v<decltype(unit), cuda::block_level>)
    {
      CHECK(result.unit_rank() == rank);
    }
    else
    {
      CHECK(result.unit_rank() == rank * units_per_block + unit.template rank_as<unsigned>(cuda::block));
    }

    CHECK(result.lane_mask() == group.__mapping_result().lane_mask());
  };

  check(same_unit, cuda::block, 1);

  // Changing units preserves the selected group rank even when the new unit count is dynamic.
  group.with_static_unit_count([&](auto specialized) {
    constexpr auto selected_rank = specialized.static_rank(cudax::coop::parent_group);
    static_assert(selected_rank < 2);
    const cudax::coop::group_view threads{cuda::gpu_thread, specialized};
    static_assert(threads.static_rank(cudax::coop::parent_group) == selected_rank);
    static_assert(cuda::gpu_thread.static_count(threads)
                  == (Static ? Original::static_unit_count(selected_rank) * 128 : cuda::std::dynamic_extent));
    check(threads, cuda::gpu_thread, 128);
  });

  // With 128 threads per block, the groups contain four and twelve warps.
  // Static dimensions preserve those counts; runtime dimensions require dynamic counts.
  const cudax::coop::group_view warps{cuda::warp, group};
  using Warps = typename decltype(warps)::__mapping_result_type;

  static_assert(Warps::has_uniform_static_unit_count() == !Static);
  static_assert(Warps::static_unit_count() == cuda::std::dynamic_extent);
  static_assert(Warps::static_unit_count(0) == (Static ? 4 : cuda::std::dynamic_extent));
  static_assert(Warps::static_unit_count(1) == (Static ? 12 : cuda::std::dynamic_extent));

  check(warps, cuda::warp, 4);

  // Specializing a view must retain the original group, including on repeated dispatch.
  warps.with_static_unit_count([&](const auto& specialized) {
    using Specialized = cuda::std::remove_cvref_t<decltype(specialized)>;
    using Result      = typename Specialized::__mapping_result_type;
    using Group       = cuda::std::remove_cv_t<decltype(group)>;
    static_assert(cuda::std::is_same_v<Specialized, cudax::coop::group_view<cuda::warp_level, Group, Result>>);
    static_assert((Result::static_group_rank() != cuda::std::dynamic_extent) == Static);
    check(specialized, cuda::warp, 4);
    specialized.with_static_unit_count([&](const auto& again) {
      static_assert(cuda::std::is_same_v<cuda::std::remove_cvref_t<decltype(again)>, Specialized>);
      check(again, cuda::warp, 4);
    });
  });

  // Direct conversion to threads scales each original block count by 128.
  const cudax::coop::group_view threads{cuda::gpu_thread, group};
  using Threads = typename decltype(threads)::__mapping_result_type;

  static_assert(Threads::has_uniform_static_unit_count() == !Static);
  static_assert(Threads::static_unit_count(0) == (Static ? 128 : cuda::std::dynamic_extent));
  static_assert(Threads::static_unit_count(1) == (Static ? 384 : cuda::std::dynamic_extent));

  check(threads, cuda::gpu_thread, 128);

  // Converting blocks to warps to threads must match direct conversion, including
  // the result type and the fallback when block dimensions are dynamic.
  const cudax::coop::group_view converted_threads{cuda::gpu_thread, warps};

  static_assert(cuda::std::is_same_v<typename decltype(converted_threads)::__mapping_result_type, Threads>);

  check(converted_threads, cuda::gpu_thread, 128);

  // Exercise the conversion from a view while retaining the original group type.
  const cudax::coop::group_view<cuda::thread_level, cuda::std::remove_cv_t<decltype(group)>, Threads> converted{
    cuda::gpu_thread, warps};

  check(converted, cuda::gpu_thread, 128);
}

struct NonuniformViewKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    constexpr bool is_static = decltype(cuda::gpu_thread.extents(cuda::block, config))::rank_dynamic() == 0;

    // Check that conversion preserves both strong and conservative mapping guarantees.
    test_nonuniform_group_view<is_static, true, true>(config);
    test_nonuniform_group_view<is_static, false, false>(config);
  }
};

C2H_TEST("Group View preserves per-group static unit counts", "[group_view]")
{
  const cuda::stream stream{cuda::devices[0]};

  // Static block dimensions allow per-group counts to be recomputed at compile time.
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<4>(), cuda::block_dims<128>()), NonuniformViewKernel{});

  // The same launch shape with runtime dimensions exercises the dynamic fallback.
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<4>(), cuda::block_dims(128)), NonuniformViewKernel{});

  stream.sync();
}

template <class Parent, class View>
__device__ void test_viewed_parent(const Parent& parent, const View& view)
{
  const cudax::coop::group_by mapping{cuda::std::cw<32>};
  const cudax::coop::lane_synchronizer synchronizer{};

  const auto check = [&](const auto& child) {
    using Child  = cuda::std::remove_cvref_t<decltype(child)>;
    using Result = typename Child::__mapping_result_type;
    using Sync   = typename Child::__synchronizer_instance_type;

    static_assert(cuda::std::is_same_v<Child, cudax::coop::virtual_group<cuda::thread_level, Parent, Result>>
                  || cuda::std::is_same_v<Child, cudax::coop::generic_group<cuda::thread_level, Parent, Result, Sync>>);

    // Static counts must come from the view, including specialized views.
    const cudax::coop::group_view child_view{child};
    const cudax::coop::group_view parent_view{view};
    static_assert(cuda::gpu_thread.static_count(child_view) == 32);
    static_assert(Result::static_group_count()
                  == (cuda::gpu_thread.static_count(parent_view) == cuda::std::dynamic_extent
                        ? cuda::std::dynamic_extent
                        : cuda::gpu_thread.static_count(parent_view) / 32));
    CHECK(child.count(parent) == cuda::gpu_thread.count(view) / 32);
    CHECK(child.template count_as<unsigned>(parent) == cuda::gpu_thread.count(view) / 32);
    CHECK(child.rank(parent) == cuda::gpu_thread.rank(view) / 32);
    CHECK(child.template rank_as<unsigned>(parent) == cuda::gpu_thread.rank(view) / 32);
    check_parent_view_queries(child, parent);
    check_parent_view_queries(child_view, parent);
    CHECK(cuda::gpu_thread.count(child) == 32);
    CHECK(cuda::gpu_thread.rank(child) == cuda::gpu_thread.rank(view) % 32);
  };

  const cudax::coop::virtual_group virtual_child{cuda::gpu_thread, view, mapping};
  check(virtual_child);

  const cudax::coop::generic_group child{cuda::gpu_thread, view, mapping, synchronizer};
  check(child);
  child.sync();

  const auto made_virtual = cudax::coop::make_group<cudax::coop::virtual_group>(cuda::gpu_thread, view, mapping);
  check(made_virtual);

  const auto made = cudax::coop::make_group<cudax::coop::generic_group>(cuda::gpu_thread, view, mapping, synchronizer);
  check(made);
  made.sync();

  const auto optional = cudax::coop::make_group<cudax::coop::generic_group>(
    cudax::coop::non_exhaustive, cuda::gpu_thread, view, mapping, synchronizer);
  REQUIRE(optional.has_value());
  check(*optional);
  optional->sync();
}

struct ViewedParentKernel
{
  template <class Config>
  __device__ void operator()(Config config) const
  {
    const cudax::coop::this_block block{config};
    const cudax::coop::group_view block_view{block};
    test_viewed_parent(block, block_view);

    // The viewed group can have a coarser unit than its view and the child.
    const cudax::coop::virtual_group parent{cuda::warp, block, cudax::coop::identity_mapping{}};
    const cudax::coop::group_view threads{cuda::gpu_thread, parent};
    test_viewed_parent(parent, threads);

    const cudax::coop::group_view copied_threads{threads};
    static_assert(cuda::std::is_same_v<decltype(threads), decltype(copied_threads)>);
    test_viewed_parent(parent, copied_threads);

    // Specialized views must retain their static counts when the parent has unequal counts.
    const cudax::coop::virtual_group unequal{
      cuda::gpu_thread, block, cudax::coop::group_as{cuda::std::integer_sequence<int, 32, 96>{}}};
    unequal.with_static_unit_count([&](const auto& view) {
      test_viewed_parent(unequal, view);
      view.with_static_unit_count([&](const auto& specialized_view) {
        test_viewed_parent(unequal, specialized_view);
      });
    });
  }
};

// Same-unit views preserve the group's mapping result type without requiring device construction.
template <class Group>
using group_view_t = cudax::coop::group_view<typename Group::unit_level, Group, typename Group::__mapping_result_type>;

void test_viewed_parent_types()
{
  using Config = decltype(cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<128>()));
  using Parent = cudax::coop::this_block<Config::hierarchy_type>;
  using View   = group_view_t<Parent>;
  static_assert(cuda::std::is_same_v<cudax::coop::__viewed_group_t<View>, Parent>);
  static_assert(cudax::coop::__is_parent_group_query_v<Parent, Parent>);
  static_assert(cudax::coop::__is_parent_group_query_v<View, Parent>);

  // Return-type queries must also be well-formed outside device functions.
  static_assert(can_rank_parent<View, cuda::block_level>);
  using Result      = Parent::__mapping_result_type;
  using Sync        = Parent::__synchronizer_instance_type;
  using Virtual     = cudax::coop::virtual_group<cuda::thread_level, Parent, Result>;
  using Generic     = cudax::coop::generic_group<cuda::thread_level, Parent, Result, Sync>;
  using VirtualView = group_view_t<Virtual>;
  using GenericView = group_view_t<Generic>;
  static_assert(can_rank_parent<VirtualView, Parent>);
  static_assert(can_rank_parent<GenericView, Parent>);
}

C2H_TEST("Groups constructed from views use the viewed parent", "[group_view]")
{
  test_viewed_parent_types();

  const cuda::stream stream{cuda::devices[0]};

  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<128>()), ViewedParentKernel{});
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(128)), ViewedParentKernel{});

  stream.sync();
}
