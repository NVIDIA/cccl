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

void basic_test_implementation()
{
  constexpr auto block_cnt = 256;
  constexpr auto grid_cnt  = 512;
  auto dimensions          = cudax::make_hierarchy(cudax::block_dims<block_cnt>(), cudax::grid_dims<grid_cnt>());
  static_assert(dimensions.flatten().x == grid_cnt * block_cnt);
  static_assert(dimensions.flatten(cudax::thread).x == grid_cnt * block_cnt);
  static_assert(dimensions.flatten(cudax::thread, cudax::grid).x == grid_cnt * block_cnt);
  static_assert(dimensions.count() == grid_cnt * block_cnt);
  static_assert(dimensions.count(cudax::thread) == grid_cnt * block_cnt);
  static_assert(dimensions.count(cudax::thread, cudax::grid) == grid_cnt * block_cnt);
  static_assert(dimensions.static_count() == grid_cnt * block_cnt);
  static_assert(dimensions.static_count(cudax::thread) == grid_cnt * block_cnt);
  static_assert(dimensions.static_count(cudax::thread, cudax::grid) == grid_cnt * block_cnt);

  static_assert(dimensions.flatten(cudax::thread, cudax::block).x == block_cnt);
  static_assert(dimensions.flatten(cudax::block, cudax::grid).x == grid_cnt);
  static_assert(dimensions.count(cudax::thread, cudax::block) == block_cnt);
  static_assert(dimensions.count(cudax::block, cudax::grid) == grid_cnt);
  static_assert(dimensions.static_count(cudax::thread, cudax::block) == block_cnt);
  static_assert(dimensions.static_count(cudax::block, cudax::grid) == grid_cnt);

  auto dimensions_dyn = cudax::make_hierarchy(cudax::block_dims(block_cnt), cudax::grid_dims(grid_cnt));

  test_host_dev(dimensions_dyn, [=] __host__ __device__(const decltype(dimensions_dyn)& dims) {
    HOST_DEV_REQUIRE(dims.flatten().x == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.flatten(cudax::thread).x == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.flatten(cudax::thread, cudax::grid).x == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.count() == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.count(cudax::thread) == grid_cnt * block_cnt);
    HOST_DEV_REQUIRE(dims.count(cudax::thread, cudax::grid) == grid_cnt * block_cnt);

    HOST_DEV_REQUIRE(dims.flatten(cudax::thread, cudax::block).x == block_cnt);
    HOST_DEV_REQUIRE(dims.flatten(cudax::block, cudax::grid).x == grid_cnt);
    HOST_DEV_REQUIRE(dims.count(cudax::thread, cudax::block) == block_cnt);
    HOST_DEV_REQUIRE(dims.count(cudax::block, cudax::grid) == grid_cnt);
  });
  static_assert(dimensions_dyn.static_count(cudax::thread, cudax::block) == cuda::std::dynamic_extent);
  static_assert(dimensions_dyn.static_count(cudax::thread, cudax::grid) == cuda::std::dynamic_extent);

  auto dims_multidim = cudax::block_dims<2, 3, 4>() & cudax::grid_dims<16, 4, 1>();

  static_assert(dims_multidim.flatten() == dim3(32, 12, 4));
  static_assert(dims_multidim.flatten(cudax::thread) == dim3(32, 12, 4));
  static_assert(dims_multidim.flatten(cudax::thread, cudax::grid) == dim3(32, 12, 4));
  static_assert(dims_multidim.flatten().extent(0) == 32);
  static_assert(dims_multidim.flatten().extent(1) == 12);
  static_assert(dims_multidim.flatten().extent(2) == 4);
  static_assert(dims_multidim.count() == 512 * 3);
  static_assert(dims_multidim.count(cudax::thread) == 512 * 3);
  static_assert(dims_multidim.count(cudax::thread, cudax::grid) == 512 * 3);
  static_assert(dims_multidim.static_count() == 512 * 3);
  static_assert(dims_multidim.static_count(cudax::thread) == 512 * 3);
  static_assert(dims_multidim.static_count(cudax::thread, cudax::grid) == 512 * 3);

  static_assert(dims_multidim.flatten(cudax::thread, cudax::block) == dim3(2, 3, 4));
  static_assert(dims_multidim.flatten(cudax::block, cudax::grid) == dim3(16, 4, 1));
  static_assert(dims_multidim.count(cudax::thread, cudax::block) == 24);
  static_assert(dims_multidim.count(cudax::block, cudax::grid) == 64);
  static_assert(dims_multidim.static_count(cudax::thread, cudax::block) == 24);
  static_assert(dims_multidim.static_count(cudax::block, cudax::grid) == 64);

  auto dims_multidim_dyn = cudax::block_dims(dim3(2, 3, 4)) & cudax::grid_dims(dim3(16, 4, 1));

  test_host_dev(dims_multidim_dyn, [] __host__ __device__(const decltype(dims_multidim_dyn)& dims) {
    HOST_DEV_REQUIRE(dims.flatten() == dim3(32, 12, 4));
    HOST_DEV_REQUIRE(dims.flatten(cudax::thread) == dim3(32, 12, 4));
    HOST_DEV_REQUIRE(dims.flatten(cudax::thread, cudax::grid) == dim3(32, 12, 4));
    HOST_DEV_REQUIRE(dims.flatten().extent(0) == 32);
    HOST_DEV_REQUIRE(dims.flatten().extent(1) == 12);
    HOST_DEV_REQUIRE(dims.flatten().extent(2) == 4);
    HOST_DEV_REQUIRE(dims.count() == 512 * 3);
    HOST_DEV_REQUIRE(dims.count(cudax::thread) == 512 * 3);
    HOST_DEV_REQUIRE(dims.count(cudax::thread, cudax::grid) == 512 * 3);

    HOST_DEV_REQUIRE(dims.flatten(cudax::thread, cudax::block) == dim3(2, 3, 4));
    HOST_DEV_REQUIRE(dims.flatten(cudax::block, cudax::grid) == dim3(16, 4, 1));
    HOST_DEV_REQUIRE(dims.count(cudax::thread, cudax::block) == 24);
    HOST_DEV_REQUIRE(dims.count(cudax::block, cudax::grid) == 64);
  });
  static_assert(dimensions_dyn.static_count(cudax::thread, cudax::block) == cuda::std::dynamic_extent);
  static_assert(dimensions_dyn.static_count(cudax::thread, cudax::grid) == cuda::std::dynamic_extent);

  auto dims_mixed = cudax::block_dims<block_cnt>() & cudax::grid_dims(dim3(8, 4, 2));

  test_host_dev(dims_mixed, [] __host__ __device__(const decltype(dims_mixed)& dims) {
    HOST_DEV_REQUIRE(dims.flatten() == dim3(2048, 4, 2));
    HOST_DEV_REQUIRE(dims.flatten(cudax::thread) == dim3(2048, 4, 2));
    HOST_DEV_REQUIRE(dims.flatten(cudax::thread, cudax::grid) == dim3(2048, 4, 2));
    HOST_DEV_REQUIRE(dims.flatten().extent(0) == 2048);
    HOST_DEV_REQUIRE(dims.flatten().extent(1) == 4);
    HOST_DEV_REQUIRE(dims.flatten().extent(2) == 2);
    HOST_DEV_REQUIRE(dims.count() == 16 * 1024);
    HOST_DEV_REQUIRE(dims.count(cudax::thread) == 16 * 1024);
    HOST_DEV_REQUIRE(dims.count(cudax::thread, cudax::grid) == 16 * 1024);

    HOST_DEV_REQUIRE(dims.flatten(cudax::block, cudax::grid) == dim3(8, 4, 2));
    HOST_DEV_REQUIRE(dims.count(cudax::block, cudax::grid) == 64);
  });
  static_assert(dims_mixed.flatten(cudax::thread, cudax::block) == block_cnt);
  static_assert(dims_mixed.count(cudax::thread, cudax::block) == block_cnt);
  static_assert(dims_mixed.static_count(cudax::thread, cudax::block) == block_cnt);
  static_assert(dims_mixed.static_count(cudax::block, cudax::grid) == cuda::std::dynamic_extent);

  // TODO include mixed static and dynamic info on a single level
  // Currently bugged in std::extents
}

TEST_CASE("Basic", "[hierarchy]")
{
  basic_test_implementation();
}

void cluster_dims_test()
{
  SECTION("Static cluster dims")
  {
    auto dimensions =
      cudax::make_hierarchy(cudax::block_dims<256>(), cudax::cluster_dims<8>(), cudax::grid_dims<512>());

    static_assert(dimensions.flatten().x == 1024 * 1024);
    static_assert(dimensions.count() == 1024 * 1024);
    static_assert(dimensions.static_count() == 1024 * 1024);

    static_assert(dimensions.flatten(cudax::thread, cudax::block).x == 256);
    static_assert(dimensions.flatten(cudax::block, cudax::grid).x == 4 * 1024);
    static_assert(dimensions.count(cudax::thread, cudax::cluster) == 2 * 1024);
    static_assert(dimensions.count(cudax::cluster) == 512);
    static_assert(dimensions.static_count(cudax::cluster) == 512);
    static_assert(dimensions.static_count(cudax::block, cudax::cluster) == 8);
  }
  SECTION("Mixed cluster dims")
  {
    auto dims_mixed = cudax::make_hierarchy(
      cudax::block_dims<256>(), cudax::cluster_dims(dim3(2, 2, 1)), cudax::grid_dims(dim3(1, 3, 9)));
    test_host_dev(
      dims_mixed,
      [] __host__ __device__(const decltype(dims_mixed)& dims) {
        HOST_DEV_REQUIRE(dims.flatten() == dim3(512, 6, 9));
        HOST_DEV_REQUIRE(dims.count() == 27 * 1024);

        HOST_DEV_REQUIRE(dims.flatten(cudax::block, cudax::grid) == dim3(2, 6, 9));
        HOST_DEV_REQUIRE(dims.count(cudax::block, cudax::grid) == 108);
        HOST_DEV_REQUIRE(dims.flatten(cudax::cluster, cudax::grid) == dim3(1, 3, 9));
        HOST_DEV_REQUIRE(dims.flatten(cudax::thread, cudax::cluster) == dim3(512, 2, 1));
      },
      arch_filter<std::less<int>, 90>);
    static_assert(dims_mixed.flatten(cudax::thread, cudax::block) == 256);
    static_assert(dims_mixed.count(cudax::thread, cudax::block) == 256);
    static_assert(dims_mixed.static_count(cudax::thread, cudax::block) == 256);
    static_assert(dims_mixed.static_count(cudax::block, cudax::cluster) == cuda::std::dynamic_extent);
    static_assert(dims_mixed.static_count(cudax::block) == cuda::std::dynamic_extent);
  }
}

TEST_CASE("Cluster dims", "[hierarchy]")
{
  cluster_dims_test();
}

void flatten_static_test()
{
  constexpr auto block_cnt = 128;
  constexpr auto grid_x    = 256;
  constexpr auto grid_y    = 4;
  constexpr auto grid_z    = 1;

  auto static_dims = cudax::block_dims<block_cnt>() & cudax::grid_dims<grid_x, grid_y, grid_z>();
  using dims_type  = decltype(static_dims);

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

TEST_CASE("Flatten static", "[hierarchy]")
{
  flatten_static_test();
}

TEST_CASE("Different constructions", "[hierarchy]")
{
  const auto block_cnt   = 512;
  const auto cluster_cnt = 8;
  const auto grid_cnt    = 256;

  [[maybe_unused]] const auto dimensions2 =
    cudax::block_dims<block_cnt>() & cudax::cluster_dims<cluster_cnt>() & cudax::grid_dims(grid_cnt);
  [[maybe_unused]] const auto dimensions3 =
    cudax::grid_dims(grid_cnt) & cudax::cluster_dims<cluster_cnt>() & cudax::block_dims<block_cnt>();

  [[maybe_unused]] const auto dimensions4 =
    cudax::cluster_dims<cluster_cnt>() & cudax::grid_dims(grid_cnt) & cudax::block_dims<block_cnt>();
  [[maybe_unused]] const auto dimensions5 =
    cudax::cluster_dims<cluster_cnt>() & cudax::block_dims<block_cnt>() & cudax::grid_dims(grid_cnt);

  [[maybe_unused]] const auto dimensions6 = cudax::make_hierarchy(
    cudax::block_dims<block_cnt>(), cudax::cluster_dims<cluster_cnt>(), cudax::grid_dims(grid_cnt));
  [[maybe_unused]] const auto dimensions7 = cudax::make_hierarchy(
    cudax::grid_dims(grid_cnt), cudax::cluster_dims<cluster_cnt>(), cudax::block_dims<block_cnt>());

  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions3)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions4)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions5)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions6)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions7)>);

  [[maybe_unused]] const auto dims_weird_order =
    cudax::grid_dims(grid_cnt) & (cudax::cluster_dims<cluster_cnt>() & cudax::block_dims<block_cnt>());
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dims_weird_order)>);

  static_assert(dimensions2.count(cudax::thread, cudax::block) == block_cnt);
  static_assert(dimensions2.count(cudax::thread, cudax::cluster) == cluster_cnt * block_cnt);
  static_assert(dimensions2.count(cudax::block, cudax::cluster) == cluster_cnt);
  HOST_DEV_REQUIRE(dimensions2.count() == grid_cnt * cluster_cnt * block_cnt);

  static_assert(cudax::has_level<cudax::block_level, decltype(dimensions2)>);
  static_assert(cudax::has_level<cudax::cluster_level, decltype(dimensions2)>);
  static_assert(cudax::has_level<cudax::grid_level, decltype(dimensions2)>);
  static_assert(!cudax::has_level<cudax::thread_level, decltype(dimensions2)>);
}

TEST_CASE("Replace level", "[hierarchy]")
{
  const auto dimensions = cudax::block_dims<512>() & cudax::cluster_dims<8>() & cudax::grid_dims(256);
  const auto fragment   = dimensions.fragment(cudax::block, cudax::grid);
  static_assert(!cudax::has_level<cudax::block_level, decltype(fragment)>);
  static_assert(!cudax::has_level_or_unit<cudax::thread_level, decltype(fragment)>);
  static_assert(cudax::has_level<cudax::cluster_level, decltype(fragment)>);
  static_assert(cudax::has_level<cudax::grid_level, decltype(fragment)>);
  static_assert(cudax::has_level_or_unit<cudax::block_level, decltype(fragment)>);

  // TODO we probably should introduce a way to do this without the operator
  const auto replaced = fragment & cudax::block_dims(256);
  static_assert(cudax::has_level<cudax::block_level, decltype(replaced)>);
  static_assert(cudax::has_level_or_unit<cudax::thread_level, decltype(replaced)>);
  REQUIRE(replaced.count(cudax::thread, cudax::block) == 256);
}

template <typename Dims>
__global__ void kernel(Dims d)
{
  auto grid  = cg::this_grid();
  auto block = cg::this_thread_block();

  assert(grid.thread_rank() == (cudax::hierarchy::rank(cudax::thread, cudax::grid)));
  assert(grid.block_rank() == (cudax::hierarchy::rank(cudax::block, cudax::grid)));
  assert(grid.thread_rank() == cudax::grid.rank(cudax::thread));
  assert(grid.block_rank() == cudax::grid.rank(cudax::block));

  assert(grid.block_index() == (cudax::hierarchy::index(cudax::block, cudax::grid)));
  assert(grid.block_index() == cudax::grid.index(cudax::block));

  assert(grid.num_threads() == (cudax::hierarchy::count(cudax::thread, cudax::grid)));
  assert(grid.num_blocks() == (cudax::hierarchy::count(cudax::block, cudax::grid)));

  assert(grid.num_threads() == cudax::grid.count(cudax::thread));
  assert(grid.num_blocks() == cudax::grid.count(cudax::block));

  assert(grid.dim_blocks() == (cudax::hierarchy::dims<cudax::block_level, cudax::grid_level>()));
  assert(grid.dim_blocks() == cudax::grid.dims(cudax::block));

  assert(block.thread_rank() == (cudax::hierarchy::rank<cudax::thread_level, cudax::block_level>()));
  assert(block.thread_index() == (cudax::hierarchy::index<cudax::thread_level, cudax::block_level>()));
  assert(block.num_threads() == (cudax::hierarchy::count<cudax::thread_level, cudax::block_level>()));
  assert(block.dim_threads() == (cudax::hierarchy::dims<cudax::thread_level, cudax::block_level>()));

  assert(block.thread_rank() == cudax::block.rank(cudax::thread));
  assert(block.thread_index() == cudax::block.index(cudax::thread));
  assert(block.num_threads() == cudax::block.count(cudax::thread));
  assert(block.dim_threads() == cudax::block.dims(cudax::thread));

  auto block_index = d.index(cudax::thread, cudax::block);
  assert(block_index == block.thread_index());
  auto grid_index = d.index();
  assert(grid_index.x == grid.block_index().x * block.dim_threads().x + block.thread_index().x);
  assert(grid_index.y == grid.block_index().y * block.dim_threads().y + block.thread_index().y);
  assert(grid_index.z == grid.block_index().z * block.dim_threads().z + block.thread_index().z);

  assert(d.rank(cudax::block) == grid.block_rank());
  assert(d.rank(cudax::thread, cudax::block) == block.thread_rank());
  assert(d.rank() == grid.thread_rank());
}

TEST_CASE("Dims queries indexing and ambient hierarchy", "[hierarchy]")
{
  const auto dims = cuda::std::make_tuple(
    cudax::block_dims(dim3(64, 4, 2)) & cudax::grid_dims(dim3(12, 6, 3)),
    cudax::block_dims(dim3(2, 4, 64)) & cudax::grid_dims(dim3(3, 6, 12)),
    cudax::block_dims<256>() & cudax::grid_dims<4>(),
    cudax::block_dims<16, 2, 4>() & cudax::grid_dims<2, 3, 4>(),
    cudax::block_dims(dim3(8, 4, 2)) & cudax::grid_dims<4, 5, 6>(),
    cudax::block_dims<8, 2, 4>() & cudax::grid_dims(dim3(5, 4, 3)));

  apply_each(
    [](const auto& launch_dims) {
      auto [grid, block] = cudax::get_launch_dimensions(launch_dims);

      kernel<<<grid, block>>>(launch_dims);
      CUDART(cudaDeviceSynchronize());
    },
    dims);
}

template <typename Dims>
__global__ void rank_kernel_optimized(Dims d, unsigned int* out)
{
  auto thread_id = d.rank(cudax::thread, cudax::block);
  out[thread_id] = thread_id;
}

template <typename Dims>
__global__ void rank_kernel(Dims d, unsigned int* out)
{
  auto thread_id = cudax::hierarchy::rank(cudax::thread, cudax::block);
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

  const auto dims_static = cudax::block_dims<256>() & cudax::grid_dims(dim3(2, 2, 2));
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
  // static_assert(std::is_trivial_v<decltype(cudax::block_dims(256))>);
  // static_assert(std::is_trivial_v<decltype(cudax::block_dims<256>())>);

  // Hierarchy is not trivially copyable (yet), because tuple is not
  // static_assert(std::is_trivially_copyable_v<decltype(cudax::block_dims<256>()
  // & cudax::grid_dims<256>())>);
  // static_assert(std::is_trivially_copyable_v<decltype(cudax::std::make_tuple(cudax::block_dims<256>(),
  // cudax::grid_dims<256>()))>);
}
