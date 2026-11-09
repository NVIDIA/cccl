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

template <class Config, class Level>
__device__ void test_group_view(Config config, Level level)
{
  const auto g = cudax::coop::make_this_group(level, config);

  // Test the view is not default constructible.
  {
    using GroupView = decltype(cudax::coop::group_view{g});
    static_assert(!cuda::std::is_default_constructible_v<GroupView>);
  }

  // Test the view is copy constructible.
  {
    using GroupView = decltype(cudax::coop::group_view{g});
    static_assert(cuda::std::is_copy_constructible_v<GroupView>);
  }

  // Views support assignment and explicitly nested view types.
  {
    cudax::coop::group_view view{g};
    using View = decltype(view);
    static_assert(cuda::std::is_copy_assignable_v<View>);
    static_assert(cuda::std::is_move_assignable_v<View>);
    View other{g};
    other = view;
    view  = cuda::std::move(other);

    const cudax::coop::group_view<typename View::unit_type, View> nested{view};
    static_assert(nested.static_count(level) == g.static_count(level));
    REQUIRE(nested.count(level) == g.count(level));
    REQUIRE(nested.template count_as<unsigned>(level) == g.template count_as<unsigned>(level));
    REQUIRE(nested.rank(level) == g.rank(level));
    REQUIRE(nested.template rank_as<unsigned>(level) == g.template rank_as<unsigned>(level));
    nested.sync();
  }

  // Test view over the group.
  {
    const cudax::coop::group_view gv{g};

    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv.static_count(level) == 1);
    REQUIRE(gv.count(level) == 1);
    REQUIRE(gv.rank(level) == 0);

    static_assert(gv.is_always_contiguous() == g.is_always_contiguous());

    gv.sync();
    gv.wait(gv.arrive());
    gv.sync_aligned();
    gv.wait_aligned(gv.arrive_aligned());

    // Test that the view is copyable.
    const cudax::coop::group_view gv2{gv};

    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    gv2.sync();
    gv2.wait(gv2.arrive());
    gv2.sync_aligned();
    gv2.wait_aligned(gv2.arrive_aligned());
  }

  // Test view over the group with the unit changed.
  {
    const cudax::coop::group_view gv{cuda::gpu_thread, g};

    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv.static_count(level) == 1);
    REQUIRE(gv.count(level) == 1);
    REQUIRE(gv.rank(level) == 0);

    static_assert(gv.is_always_contiguous() == g.is_always_contiguous());

    gv.sync();
    gv.wait(gv.arrive());
    gv.sync_aligned();
    gv.wait_aligned(gv.arrive_aligned());

    // Test that the view is copyable.
    const cudax::coop::group_view gv2{gv};

    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    gv2.sync();
    gv2.wait(gv2.arrive());
    gv2.sync_aligned();
    gv2.wait_aligned(gv2.arrive_aligned());
  }

  // Test view of the group view of the group with the unit changed.
  {
    const cudax::coop::group_view gv{g};

    const cudax::coop::group_view gv2{cuda::gpu_thread, gv};

    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    static_assert(gv2.is_always_contiguous() == g.is_always_contiguous());

    gv2.sync();
    gv2.wait(gv2.arrive());
    gv2.sync_aligned();
    gv2.wait_aligned(gv2.arrive_aligned());
  }

  // Test view of a group view of a group with the unit changed twice.
  if constexpr (!cuda::std::is_same_v<Level, cuda::thread_level>)
  {
    const cudax::coop::group_view gv{cuda::warp, g};

    const cudax::coop::group_view gv2{cuda::gpu_thread, gv};

    if constexpr (cuda::std::is_same_v<Level, cuda::thread_level>)
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == 1);
      REQUIRE(cuda::gpu_thread.rank(gv2) == 0);
    }
    else
    {
      REQUIRE(cuda::gpu_thread.count(gv2) == cuda::gpu_thread.count(level));
      REQUIRE(cuda::gpu_thread.rank(gv2) == cuda::gpu_thread.rank(level));
    }

    static_assert(gv2.static_count(level) == 1);
    REQUIRE(gv2.count(level) == 1);
    REQUIRE(gv2.rank(level) == 0);

    static_assert(gv2.is_always_contiguous() == g.is_always_contiguous());

    gv2.sync();
    gv2.wait(gv2.arrive());
    gv2.sync_aligned();
    gv2.wait_aligned(gv2.arrive_aligned());
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
    REQUIRE(g2_view.count(g2) == cuda::warp.count(level) / n);
    static_assert(g2_view.is_always_contiguous());
    g2_view.sync();
    g2_view.wait(g2_view.arrive());
    g2_view.sync_aligned();
    g2_view.wait_aligned(g2_view.arrive_aligned());

    const cudax::coop::group_view g2_view_threads{cuda::gpu_thread, g2_view};
    REQUIRE(cuda::gpu_thread.count(g2_view_threads) == n * cuda::gpu_thread.count(cuda::warp));
    REQUIRE(g2_view_threads.count(g2) == n);
    static_assert(g2_view_threads.is_always_contiguous());
    g2_view_threads.sync();
    g2_view_threads.wait(g2_view_threads.arrive());
    g2_view_threads.sync_aligned();
    g2_view_threads.wait_aligned(g2_view_threads.arrive_aligned());
  }
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

  // With 128 threads per block, the groups contain four and twelve warps.
  // Static dimensions preserve those counts; runtime dimensions require dynamic counts.
  const cudax::coop::group_view warps{cuda::warp, group};
  using Warps = typename decltype(warps)::__mapping_result_type;

  static_assert(Warps::has_uniform_static_unit_count() == !Static);
  static_assert(Warps::static_unit_count() == cuda::std::dynamic_extent);
  static_assert(Warps::static_unit_count(0) == (Static ? 4 : cuda::std::dynamic_extent));
  static_assert(Warps::static_unit_count(1) == (Static ? 12 : cuda::std::dynamic_extent));

  check(warps, cuda::warp, 4);

  // Direct conversion to threads scales each original block count by 128.
  const cudax::coop::group_view threads{cuda::gpu_thread, group};
  using Threads = typename decltype(threads)::__mapping_result_type;

  static_assert(Threads::has_uniform_static_unit_count() == !Static);
  static_assert(Threads::static_unit_count(0) == (Static ? 128 : cuda::std::dynamic_extent));
  static_assert(Threads::static_unit_count(1) == (Static ? 384 : cuda::std::dynamic_extent));

  check(threads, cuda::gpu_thread, 128);

  // Converting blocks to warps to threads must match direct conversion, including
  // the result type and the fallback when block dimensions are dynamic.
  const cudax::coop::group_view nested_threads{cuda::gpu_thread, warps};

  static_assert(cuda::std::is_same_v<typename decltype(nested_threads)::__mapping_result_type, Threads>);

  check(nested_threads, cuda::gpu_thread, 128);

  // Exercise the conversion from a view while retaining the original group type.
  const cudax::coop::group_view<cuda::thread_level, cuda::std::remove_cv_t<decltype(group)>> converted{
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
