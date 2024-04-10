//===----------------------------------------------------------------------===//
//
// Part of CUDA Next in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <iostream>

#include "catch2_helpers/testing_common.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

TEST_CASE("Basic", "[hierarchy]")
{
  constexpr auto block_cnt = 256;
  constexpr auto grid_cnt  = 512;
  auto dimensions = cuda_next::make_hierarchy(cuda_next::block_dims<block_cnt>(), cuda_next::grid_dims<grid_cnt>());
  static_assert(dimensions.flatten().x == grid_cnt * block_cnt);
  static_assert(dimensions.flatten(cuda_next::thread).x == grid_cnt * block_cnt);
  static_assert(dimensions.flatten(cuda_next::thread, cuda_next::grid).x == grid_cnt * block_cnt);
  static_assert(dimensions.count() == grid_cnt * block_cnt);
  static_assert(dimensions.count(cuda_next::thread) == grid_cnt * block_cnt);
  static_assert(dimensions.count(cuda_next::thread, cuda_next::grid) == grid_cnt * block_cnt);
  static_assert(dimensions.static_count() == grid_cnt * block_cnt);
  static_assert(dimensions.static_count(cuda_next::thread) == grid_cnt * block_cnt);
  static_assert(dimensions.static_count(cuda_next::thread, cuda_next::grid) == grid_cnt * block_cnt);

  static_assert(dimensions.flatten(cuda_next::thread, cuda_next::block).x == block_cnt);
  static_assert(dimensions.flatten(cuda_next::block, cuda_next::grid).x == grid_cnt);
  static_assert(dimensions.count(cuda_next::thread, cuda_next::block) == block_cnt);
  static_assert(dimensions.count(cuda_next::block, cuda_next::grid) == grid_cnt);
  static_assert(dimensions.static_count(cuda_next::thread, cuda_next::block) == block_cnt);
  static_assert(dimensions.static_count(cuda_next::block, cuda_next::grid) == grid_cnt);

  auto dimensions_dyn = cuda_next::make_hierarchy(cuda_next::block_dims(block_cnt), cuda_next::grid_dims(grid_cnt));

  test_host_dev(dimensions_dyn, [=] __host__ __device__(const decltype(dimensions_dyn)& dims) {
    HOST_DEV_REQUIRE(dims.flatten().x == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread).x == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread, cuda_next::grid).x == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.count() == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread) == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread, cuda_next::grid) == grid_cnt * block_cnt);

    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread, cuda_next::block).x == block_cnt);
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::block, cuda_next::grid).x == grid_cnt);
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread, cuda_next::block) == block_cnt);
    HOST_DEV_REQUIRE(dims.count(cuda_next::block, cuda_next::grid) == grid_cnt);
  });
  static_assert(dimensions_dyn.static_count(cuda_next::thread, cuda_next::block) == cuda::std::dynamic_extent);
  static_assert(dimensions_dyn.static_count(cuda_next::thread, cuda_next::grid) == cuda::std::dynamic_extent);

  auto dims_multidim = cuda_next::block_dims<2, 3, 4>() & cuda_next::grid_dims<16, 4, 1>();

  static_assert(dims_multidim.flatten() == dim3(32, 12, 4));
  static_assert(dims_multidim.flatten(cuda_next::thread) == dim3(32, 12, 4));
  static_assert(dims_multidim.flatten(cuda_next::thread, cuda_next::grid) == dim3(32, 12, 4));
  static_assert(dims_multidim.flatten().extent(0) == 32);
  static_assert(dims_multidim.flatten().extent(1) == 12);
  static_assert(dims_multidim.flatten().extent(2) == 4);
  static_assert(dims_multidim.count() == 512 * 3);
  static_assert(dims_multidim.count(cuda_next::thread) == 512 * 3);
  static_assert(dims_multidim.count(cuda_next::thread, cuda_next::grid) == 512 * 3);
  static_assert(dims_multidim.static_count() == 512 * 3);
  static_assert(dims_multidim.static_count(cuda_next::thread) == 512 * 3);
  static_assert(dims_multidim.static_count(cuda_next::thread, cuda_next::grid) == 512 * 3);

  static_assert(dims_multidim.flatten(cuda_next::thread, cuda_next::block) == dim3(2, 3, 4));
  static_assert(dims_multidim.flatten(cuda_next::block, cuda_next::grid) == dim3(16, 4, 1));
  static_assert(dims_multidim.count(cuda_next::thread, cuda_next::block) == 24);
  static_assert(dims_multidim.count(cuda_next::block, cuda_next::grid) == 64);
  static_assert(dims_multidim.static_count(cuda_next::thread, cuda_next::block) == 24);
  static_assert(dims_multidim.static_count(cuda_next::block, cuda_next::grid) == 64);

  auto dims_multidim_dyn = cuda_next::block_dims(dim3(2, 3, 4)) & cuda_next::grid_dims(dim3(16, 4, 1));

  test_host_dev(dims_multidim_dyn, [] __host__ __device__(const decltype(dims_multidim_dyn)& dims) {
    HOST_DEV_REQUIRE(dims.flatten() == dim3(32, 12, 4));
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread) == dim3(32, 12, 4));
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread, cuda_next::grid) == dim3(32, 12, 4));
    HOST_DEV_REQUIRE(dims.flatten().extent(0) == 32);
    HOST_DEV_REQUIRE(dims.flatten().extent(1) == 12);
    HOST_DEV_REQUIRE(dims.flatten().extent(2) == 4);
    HOST_DEV_REQUIRE(dims.count() == 512 * 3);
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread) == 512 * 3);
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread, cuda_next::grid) == 512 * 3);

    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread, cuda_next::block) == dim3(2, 3, 4));
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::block, cuda_next::grid) == dim3(16, 4, 1));
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread, cuda_next::block) == 24);
    HOST_DEV_REQUIRE(dims.count(cuda_next::block, cuda_next::grid) == 64);
  });
  static_assert(dimensions_dyn.static_count(cuda_next::thread, cuda_next::block) == cuda::std::dynamic_extent);
  static_assert(dimensions_dyn.static_count(cuda_next::thread, cuda_next::grid) == cuda::std::dynamic_extent);

  auto dims_mixed = cuda_next::block_dims<block_cnt>() & cuda_next::grid_dims(dim3(8, 4, 2));

  test_host_dev(dims_mixed, [] __host__ __device__(const decltype(dims_mixed)& dims) {
    HOST_DEV_REQUIRE(dims.flatten() == dim3(2048, 4, 2));
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread) == dim3(2048, 4, 2));
    HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread, cuda_next::grid) == dim3(2048, 4, 2));
    HOST_DEV_REQUIRE(dims.flatten().extent(0) == 2048);
    HOST_DEV_REQUIRE(dims.flatten().extent(1) == 4);
    HOST_DEV_REQUIRE(dims.flatten().extent(2) == 2);
    HOST_DEV_REQUIRE(dims.count() == 16 * 1024);
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread) == 16 * 1024);
    HOST_DEV_REQUIRE(dims.count(cuda_next::thread, cuda_next::grid) == 16 * 1024);

    HOST_DEV_REQUIRE(dims.flatten(cuda_next::block, cuda_next::grid) == dim3(8, 4, 2));
    HOST_DEV_REQUIRE(dims.count(cuda_next::block, cuda_next::grid) == 64);
  });
  static_assert(dims_mixed.flatten(cuda_next::thread, cuda_next::block) == block_cnt);
  static_assert(dims_mixed.count(cuda_next::thread, cuda_next::block) == block_cnt);
  static_assert(dims_mixed.static_count(cuda_next::thread, cuda_next::block) == block_cnt);
  static_assert(dims_mixed.static_count(cuda_next::block, cuda_next::grid) == cuda::std::dynamic_extent);

  // TODO include mixed static and dynamic info on a single level
  // Currently bugged in std::extents
}

TEST_CASE("Cluster dims", "[hierarchy]")
{
  SECTION("Static cluster dims")
  {
    auto dimensions = cuda_next::make_hierarchy(
      cuda_next::block_dims<256>(), cuda_next::cluster_dims<8>(), cuda_next::grid_dims<512>());

    static_assert(dimensions.flatten().x == 1024 * 1024);
    static_assert(dimensions.count() == 1024 * 1024);
    static_assert(dimensions.static_count() == 1024 * 1024);

    static_assert(dimensions.flatten(cuda_next::thread, cuda_next::block).x == 256);
    static_assert(dimensions.flatten(cuda_next::block, cuda_next::grid).x == 4 * 1024);
    static_assert(dimensions.count(cuda_next::thread, cuda_next::cluster) == 2 * 1024);
    static_assert(dimensions.count(cuda_next::cluster) == 512);
    static_assert(dimensions.static_count(cuda_next::cluster) == 512);
    static_assert(dimensions.static_count(cuda_next::block, cuda_next::cluster) == 8);
  }
  SECTION("Mixed cluster dims")
  {
    auto dims_mixed = cuda_next::make_hierarchy(
      cuda_next::block_dims<256>(), cuda_next::cluster_dims(dim3(2, 2, 1)), cuda_next::grid_dims(dim3(1, 3, 9)));
    test_host_dev(
      dims_mixed,
      [] __host__ __device__(const decltype(dims_mixed)& dims) {
        HOST_DEV_REQUIRE(dims.flatten() == dim3(512, 6, 9));
        HOST_DEV_REQUIRE(dims.count() == 27 * 1024);

        HOST_DEV_REQUIRE(dims.flatten(cuda_next::block, cuda_next::grid) == dim3(2, 6, 9));
        HOST_DEV_REQUIRE(dims.count(cuda_next::block, cuda_next::grid) == 108);
        HOST_DEV_REQUIRE(dims.flatten(cuda_next::cluster, cuda_next::grid) == dim3(1, 3, 9));
        HOST_DEV_REQUIRE(dims.flatten(cuda_next::thread, cuda_next::cluster) == dim3(512, 2, 1));
      },
      arch_filter<std::less<int>, 90>);
    static_assert(dims_mixed.flatten(cuda_next::thread, cuda_next::block) == 256);
    static_assert(dims_mixed.count(cuda_next::thread, cuda_next::block) == 256);
    static_assert(dims_mixed.static_count(cuda_next::thread, cuda_next::block) == 256);
    static_assert(dims_mixed.static_count(cuda_next::block, cuda_next::cluster) == cuda::std::dynamic_extent);
    static_assert(dims_mixed.static_count(cuda_next::block) == cuda::std::dynamic_extent);
  }
}

TEST_CASE("Flatten static", "[hierarchy]")
{
  const auto block_cnt = 128;
  const auto grid_x    = 256;
  const auto grid_y    = 4;
  const auto grid_z    = 1;

  constexpr auto static_dims = cuda_next::block_dims<block_cnt>() & cuda_next::grid_dims<grid_x, grid_y, grid_z>();
  using dims_type            = decltype(static_dims);

  test_host_dev(static_dims, [=] __host__ __device__(const dims_type& dims) {
    static_assert(dims_type::static_count() == block_cnt * 1024);
    auto flattened = dims.flatten();
    static_assert(flattened.static_extent(0) == grid_x * block_cnt);
    static_assert(flattened.static_extent(1) == grid_y);
    static_assert(flattened.static_extent(2) == grid_z);
    static_assert(flattened.extent(0) == grid_x * block_cnt);
    static_assert(flattened.extent(1) == grid_y);
    static_assert(flattened.extent(2) == grid_z);
    HOST_DEV_REQUIRE(flattened.x == grid_x * block_cnt);
    HOST_DEV_REQUIRE(flattened.y == grid_y);
    HOST_DEV_REQUIRE(flattened.z == grid_z);

    dim3 dim3s = flattened;
    HOST_DEV_REQUIRE(dim3s.x == grid_x * block_cnt);
    HOST_DEV_REQUIRE(dim3s.y == grid_y);
    HOST_DEV_REQUIRE(dim3s.z == grid_z);
  });
}

TEST_CASE("Different constructions", "[hierarchy]")
{
  const auto block_cnt   = 512;
  const auto cluster_cnt = 8;
  const auto grid_cnt    = 256;
  const auto dimensions2 =
    cuda_next::block_dims<block_cnt>() & cuda_next::cluster_dims<cluster_cnt>() & cuda_next::grid_dims(grid_cnt);
  const auto dimensions3 =
    cuda_next::grid_dims(grid_cnt) & cuda_next::cluster_dims<cluster_cnt>() & cuda_next::block_dims<block_cnt>();

  const auto dimensions4 =
    cuda_next::cluster_dims<cluster_cnt>() & cuda_next::grid_dims(grid_cnt) & cuda_next::block_dims<block_cnt>();
  const auto dimensions5 =
    cuda_next::cluster_dims<cluster_cnt>() & cuda_next::block_dims<block_cnt>() & cuda_next::grid_dims(grid_cnt);

  const auto dimensions6 = cuda_next::make_hierarchy(
    cuda_next::block_dims<block_cnt>(), cuda_next::cluster_dims<cluster_cnt>(), cuda_next::grid_dims(grid_cnt));
  const auto dimensions7 = cuda_next::make_hierarchy(
    cuda_next::grid_dims(grid_cnt), cuda_next::cluster_dims<cluster_cnt>(), cuda_next::block_dims<block_cnt>());

  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions3)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions4)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions5)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions6)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions7)>);

  const auto dims_weird_order =
    cuda_next::grid_dims(grid_cnt) & (cuda_next::cluster_dims<cluster_cnt>() & cuda_next::block_dims<block_cnt>());
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dims_weird_order)>);

  static_assert(dimensions2.count(cuda_next::thread, cuda_next::block) == block_cnt);
  static_assert(dimensions2.count(cuda_next::thread, cuda_next::cluster) == cluster_cnt * block_cnt);
  static_assert(dimensions2.count(cuda_next::block, cuda_next::cluster) == cluster_cnt);
  HOST_DEV_REQUIRE(dimensions2.count() == grid_cnt * cluster_cnt * block_cnt);

  static_assert(cuda_next::has_level<cuda_next::block_level, decltype(dimensions2)>);
  static_assert(cuda_next::has_level<cuda_next::cluster_level, decltype(dimensions2)>);
  static_assert(cuda_next::has_level<cuda_next::grid_level, decltype(dimensions2)>);
  static_assert(!cuda_next::has_level<cuda_next::thread_level, decltype(dimensions2)>);
}

TEST_CASE("Replace level", "[hierarchy]")
{
  const auto dimensions = cuda_next::block_dims<512>() & cuda_next::cluster_dims<8>() & cuda_next::grid_dims(256);
  const auto fragment   = dimensions.fragment(cuda_next::block, cuda_next::grid);
  static_assert(!cuda_next::has_level<cuda_next::block_level, decltype(fragment)>);
  static_assert(!cuda_next::has_level_or_unit<cuda_next::thread_level, decltype(fragment)>);
  static_assert(cuda_next::has_level<cuda_next::cluster_level, decltype(fragment)>);
  static_assert(cuda_next::has_level<cuda_next::grid_level, decltype(fragment)>);
  static_assert(cuda_next::has_level_or_unit<cuda_next::block_level, decltype(fragment)>);

  // TODO we probably should introduce a way to do this without the operator
  const auto replaced = fragment & cuda_next::block_dims(256);
  static_assert(cuda_next::has_level<cuda_next::block_level, decltype(replaced)>);
  static_assert(cuda_next::has_level_or_unit<cuda_next::thread_level, decltype(replaced)>);
  REQUIRE(replaced.count(cuda_next::thread, cuda_next::block) == 256);
}

template <typename Dims>
__global__ void kernel(Dims d)
{
  auto grid  = cg::this_grid();
  auto block = cg::this_thread_block();

  assert(grid.thread_rank() == (cuda_next::hierarchy::rank(cuda_next::thread, cuda_next::grid)));
  assert(grid.block_rank() == (cuda_next::hierarchy::rank(cuda_next::block, cuda_next::grid)));
  assert(grid.thread_rank() == cuda_next::grid.rank(cuda_next::thread));
  assert(grid.block_rank() == cuda_next::grid.rank(cuda_next::block));

  assert(grid.block_index() == (cuda_next::hierarchy::index(cuda_next::block, cuda_next::grid)));
  assert(grid.block_index() == cuda_next::grid.index(cuda_next::block));

  assert(grid.num_threads() == (cuda_next::hierarchy::count(cuda_next::thread, cuda_next::grid)));
  assert(grid.num_blocks() == (cuda_next::hierarchy::count(cuda_next::block, cuda_next::grid)));

  assert(grid.num_threads() == cuda_next::grid.count(cuda_next::thread));
  assert(grid.num_blocks() == cuda_next::grid.count(cuda_next::block));

  assert(grid.dim_blocks() == (cuda_next::hierarchy::dims<cuda_next::block_level, cuda_next::grid_level>()));
  assert(grid.dim_blocks() == cuda_next::grid.dims(cuda_next::block));

  assert(block.thread_rank() == (cuda_next::hierarchy::rank<cuda_next::thread_level, cuda_next::block_level>()));
  assert(block.thread_index() == (cuda_next::hierarchy::index<cuda_next::thread_level, cuda_next::block_level>()));
  assert(block.num_threads() == (cuda_next::hierarchy::count<cuda_next::thread_level, cuda_next::block_level>()));
  assert(block.dim_threads() == (cuda_next::hierarchy::dims<cuda_next::thread_level, cuda_next::block_level>()));

  assert(block.thread_rank() == cuda_next::block.rank(cuda_next::thread));
  assert(block.thread_index() == cuda_next::block.index(cuda_next::thread));
  assert(block.num_threads() == cuda_next::block.count(cuda_next::thread));
  assert(block.dim_threads() == cuda_next::block.dims(cuda_next::thread));

  auto block_index = d.index(cuda_next::thread, cuda_next::block);
  assert(block_index == block.thread_index());
  auto grid_index = d.index();
  assert(grid_index.x == grid.block_index().x * block.dim_threads().x + block.thread_index().x);
  assert(grid_index.y == grid.block_index().y * block.dim_threads().y + block.thread_index().y);
  assert(grid_index.z == grid.block_index().z * block.dim_threads().z + block.thread_index().z);

  assert(d.rank(cuda_next::block) == grid.block_rank());
  assert(d.rank(cuda_next::thread, cuda_next::block) == block.thread_rank());
  assert(d.rank() == grid.thread_rank());
}

TEST_CASE("Dims queries indexing and ambient hierarchy", "[hierarchy]")
{
  const auto dims = cuda::std::make_tuple(
    cuda_next::block_dims(dim3(64, 4, 2)) & cuda_next::grid_dims(dim3(12, 6, 3)),
    cuda_next::block_dims(dim3(2, 4, 64)) & cuda_next::grid_dims(dim3(3, 6, 12)),
    cuda_next::block_dims<256>() & cuda_next::grid_dims<4>(),
    cuda_next::block_dims<16, 2, 4>() & cuda_next::grid_dims<2, 3, 4>(),
    cuda_next::block_dims(dim3(8, 4, 2)) & cuda_next::grid_dims<4, 5, 6>(),
    cuda_next::block_dims<8, 2, 4>() & cuda_next::grid_dims(dim3(5, 4, 3)));

  apply_each(
    [](const auto& launch_dims) {
      auto [grid, block] = cuda_next::get_launch_dimensions(launch_dims);

      kernel<<<grid, block>>>(launch_dims);
      CUDART(cudaDeviceSynchronize());
    },
    dims);
}

template <typename Dims>
__global__ void rank_kernel_optimized(Dims d, unsigned int* out)
{
  auto thread_id = d.rank(cuda_next::thread, cuda_next::block);
  out[thread_id] = thread_id;
}

template <typename Dims>
__global__ void rank_kernel(Dims d, unsigned int* out)
{
  auto thread_id = cuda_next::hierarchy::rank(cuda_next::thread, cuda_next::block);
  out[thread_id] = thread_id;
}

template <typename Dims>
__global__ void rank_kernel_cg(Dims d, unsigned int* out)
{
  auto thread_id = cg::thread_block::thread_rank();
  out[thread_id] = thread_id;
}

// Testcase mostly for generated code comparison
TEST_CASE("On device rank calculation", "[hierarchy]")
{
  unsigned int* ptr;
  CUDART(cudaMalloc((void**) &ptr, 2 * 1024 * sizeof(unsigned int)));

  const auto dims_static = cuda_next::block_dims<256>() & cuda_next::grid_dims(dim3(2, 2, 2));
  rank_kernel<<<256, dim3(2, 2, 2)>>>(dims_static, ptr);
  CUDART(cudaDeviceSynchronize());
  rank_kernel_cg<<<256, dim3(2, 2, 2)>>>(dims_static, ptr);
  CUDART(cudaDeviceSynchronize());
  rank_kernel_optimized<<<256, dim3(2, 2, 2)>>>(dims_static, ptr);
  CUDART(cudaDeviceSynchronize());
  CUDART(cudaFree(ptr));
}

TEST_CASE("Trivially constructable", "[hierarchy]")
{
  // static_assert(std::is_trivial_v<decltype(cuda_next::block_dims(256))>);
  // static_assert(std::is_trivial_v<decltype(cuda_next::block_dims<256>())>);

  // Hierarchy is not trivially copyable (yet), because tuple is not
  // static_assert(std::is_trivially_copyable_v<decltype(cuda_next::block_dims<256>()
  // & cuda_next::grid_dims<256>())>);
  // static_assert(std::is_trivially_copyable_v<decltype(cuda_next::std::make_tuple(cuda_next::block_dims<256>(),
  // cuda_next::grid_dims<256>()))>);
}
