//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <cooperative_groups.h>
#include <host_device.cuh>

namespace cg = cooperative_groups;

struct basic_test_single_dim
{
  static constexpr int block_size = 256;
  static constexpr int grid_size  = 512;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    CUDAX_REQUIRE(dims.extents().x == grid_size * block_size);
    CUDAX_REQUIRE(dims.extents(cudax::thread).x == grid_size * block_size);
    CUDAX_REQUIRE(dims.extents(cudax::thread, cudax::grid).x == grid_size * block_size);
    CUDAX_REQUIRE(dims.count() == grid_size * block_size);
    CUDAX_REQUIRE(dims.count(cudax::thread) == grid_size * block_size);
    CUDAX_REQUIRE(dims.count(cudax::thread, cudax::grid) == grid_size * block_size);

    CUDAX_REQUIRE(dims.extents(cudax::thread, cudax::block).x == block_size);
    CUDAX_REQUIRE(dims.extents(cudax::block, cudax::grid).x == grid_size);
    CUDAX_REQUIRE(dims.count(cudax::thread, cudax::block) == block_size);
    CUDAX_REQUIRE(dims.count(cudax::block, cudax::grid) == grid_size);
  }

  void run()
  {
    auto dimensions = cudax::make_hierarchy(cudax::block_dims<block_size>(), cudax::grid_dims<grid_size>());
    static_assert(dimensions.extents().x == grid_size * block_size);
    static_assert(dimensions.extents(cudax::thread).x == grid_size * block_size);
    static_assert(dimensions.extents(cudax::thread, cudax::grid).x == grid_size * block_size);
    static_assert(dimensions.count() == grid_size * block_size);
    static_assert(dimensions.count(cudax::thread) == grid_size * block_size);
    static_assert(dimensions.count(cudax::thread, cudax::grid) == grid_size * block_size);
    static_assert(dimensions.static_count() == grid_size * block_size);
    static_assert(dimensions.static_count(cudax::thread) == grid_size * block_size);
    static_assert(dimensions.static_count(cudax::thread, cudax::grid) == grid_size * block_size);

    static_assert(dimensions.extents(cudax::thread, cudax::block).x == block_size);
    static_assert(dimensions.extents(cudax::block, cudax::grid).x == grid_size);
    static_assert(dimensions.count(cudax::thread, cudax::block) == block_size);
    static_assert(dimensions.count(cudax::block, cudax::grid) == grid_size);
    static_assert(dimensions.static_count(cudax::thread, cudax::block) == block_size);
    static_assert(dimensions.static_count(cudax::block, cudax::grid) == grid_size);

    auto dimensions_dyn = cudax::make_hierarchy(cudax::block_dims(block_size), cudax::grid_dims(grid_size));

    test_host_dev(dimensions_dyn, *this);

    static_assert(dimensions_dyn.static_count(cudax::thread, cudax::block) == cuda::std::dynamic_extent);
    static_assert(dimensions_dyn.static_count(cudax::thread, cudax::grid) == cuda::std::dynamic_extent);
  }
};

struct basic_test_multi_dim
{
  static constexpr int block_size = 256;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    CUDAX_REQUIRE(dims.extents() == dim3(32, 12, 4));
    CUDAX_REQUIRE(dims.extents(cudax::thread) == dim3(32, 12, 4));
    CUDAX_REQUIRE(dims.extents(cudax::thread, cudax::grid) == dim3(32, 12, 4));
    CUDAX_REQUIRE(dims.extents().extent(0) == 32);
    CUDAX_REQUIRE(dims.extents().extent(1) == 12);
    CUDAX_REQUIRE(dims.extents().extent(2) == 4);
    CUDAX_REQUIRE(dims.count() == 512 * 3);
    CUDAX_REQUIRE(dims.count(cudax::thread) == 512 * 3);
    CUDAX_REQUIRE(dims.count(cudax::thread, cudax::grid) == 512 * 3);

    CUDAX_REQUIRE(dims.extents(cudax::thread, cudax::block) == dim3(2, 3, 4));
    CUDAX_REQUIRE(dims.extents(cudax::block, cudax::grid) == dim3(16, 4, 1));
    CUDAX_REQUIRE(dims.count(cudax::thread, cudax::block) == 24);
    CUDAX_REQUIRE(dims.count(cudax::block, cudax::grid) == 64);
  }

  void run()
  {
    auto dims_multidim = cudax::block_dims<2, 3, 4>() & cudax::grid_dims<16, 4, 1>();

    static_assert(dims_multidim.extents() == dim3(32, 12, 4));
    static_assert(dims_multidim.extents(cudax::thread) == dim3(32, 12, 4));
    static_assert(dims_multidim.extents(cudax::thread, cudax::grid) == dim3(32, 12, 4));
    static_assert(dims_multidim.extents().extent(0) == 32);
    static_assert(dims_multidim.extents().extent(1) == 12);
    static_assert(dims_multidim.extents().extent(2) == 4);
    static_assert(dims_multidim.count() == 512 * 3);
    static_assert(dims_multidim.count(cudax::thread) == 512 * 3);
    static_assert(dims_multidim.count(cudax::thread, cudax::grid) == 512 * 3);
    static_assert(dims_multidim.static_count() == 512 * 3);
    static_assert(dims_multidim.static_count(cudax::thread) == 512 * 3);
    static_assert(dims_multidim.static_count(cudax::thread, cudax::grid) == 512 * 3);

    static_assert(dims_multidim.extents(cudax::thread, cudax::block) == dim3(2, 3, 4));
    static_assert(dims_multidim.extents(cudax::block, cudax::grid) == dim3(16, 4, 1));
    static_assert(dims_multidim.count(cudax::thread, cudax::block) == 24);
    static_assert(dims_multidim.count(cudax::block, cudax::grid) == 64);
    static_assert(dims_multidim.static_count(cudax::thread, cudax::block) == 24);
    static_assert(dims_multidim.static_count(cudax::block, cudax::grid) == 64);

    auto dims_multidim_dyn = cudax::block_dims(dim3(2, 3, 4)) & cudax::grid_dims(dim3(16, 4, 1));

    test_host_dev(dims_multidim_dyn, *this);

    static_assert(dims_multidim_dyn.static_count(cudax::thread, cudax::block) == cuda::std::dynamic_extent);
    static_assert(dims_multidim_dyn.static_count(cudax::thread, cudax::grid) == cuda::std::dynamic_extent);
  }
};

struct basic_test_mixed
{
  static constexpr int block_size = 256;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    CUDAX_REQUIRE(dims.extents() == dim3(2048, 4, 2));
    CUDAX_REQUIRE(dims.extents(cudax::thread) == dim3(2048, 4, 2));
    CUDAX_REQUIRE(dims.extents(cudax::thread, cudax::grid) == dim3(2048, 4, 2));
    CUDAX_REQUIRE(dims.extents().extent(0) == 2048);
    CUDAX_REQUIRE(dims.extents().extent(1) == 4);
    CUDAX_REQUIRE(dims.extents().extent(2) == 2);
    CUDAX_REQUIRE(dims.count() == 16 * 1024);
    CUDAX_REQUIRE(dims.count(cudax::thread) == 16 * 1024);
    CUDAX_REQUIRE(dims.count(cudax::thread, cudax::grid) == 16 * 1024);

    CUDAX_REQUIRE(dims.extents(cudax::block, cudax::grid) == dim3(8, 4, 2));
    CUDAX_REQUIRE(dims.count(cudax::block, cudax::grid) == 64);
  }

  void run()
  {
    auto dims_mixed = cudax::block_dims<block_size>() & cudax::grid_dims(dim3(8, 4, 2));

    test_host_dev(dims_mixed, *this);
    static_assert(dims_mixed.extents(cudax::thread, cudax::block) == block_size);
    static_assert(dims_mixed.count(cudax::thread, cudax::block) == block_size);
    static_assert(dims_mixed.static_count(cudax::thread, cudax::block) == block_size);
    static_assert(dims_mixed.static_count(cudax::block, cudax::grid) == cuda::std::dynamic_extent);

    // TODO include mixed static and dynamic info on a single level
    // Currently bugged in std::extents
  }
};

TEST_CASE("Basic", "[hierarchy]")
{
  basic_test_single_dim().run();
  basic_test_multi_dim().run();
  basic_test_mixed().run();
}

struct basic_test_cluster
{
  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    CUDAX_REQUIRE(dims.extents() == dim3(512, 6, 9));
    CUDAX_REQUIRE(dims.count() == 27 * 1024);

    CUDAX_REQUIRE(dims.extents(cudax::block, cudax::grid) == dim3(2, 6, 9));
    CUDAX_REQUIRE(dims.count(cudax::block, cudax::grid) == 108);
    CUDAX_REQUIRE(dims.extents(cudax::cluster, cudax::grid) == dim3(1, 3, 9));
    CUDAX_REQUIRE(dims.extents(cudax::thread, cudax::cluster) == dim3(512, 2, 1));
  }

  void run()
  {
    SECTION("Static cluster dims")
    {
      auto dimensions =
        cudax::make_hierarchy(cudax::block_dims<256>(), cudax::cluster_dims<8>(), cudax::grid_dims<512>());

      static_assert(dimensions.extents().x == 1024 * 1024);
      static_assert(dimensions.count() == 1024 * 1024);
      static_assert(dimensions.static_count() == 1024 * 1024);

      static_assert(dimensions.extents(cudax::thread, cudax::block).x == 256);
      static_assert(dimensions.extents(cudax::block, cudax::grid).x == 4 * 1024);
      static_assert(dimensions.count(cudax::thread, cudax::cluster) == 2 * 1024);
      static_assert(dimensions.count(cudax::cluster) == 512);
      static_assert(dimensions.static_count(cudax::cluster) == 512);
      static_assert(dimensions.static_count(cudax::block, cudax::cluster) == 8);
    }
    SECTION("Mixed cluster dims")
    {
      auto dims_mixed = cudax::make_hierarchy(
        cudax::block_dims<256>(), cudax::cluster_dims(dim3(2, 2, 1)), cudax::grid_dims(dim3(1, 3, 9)));
      test_host_dev(dims_mixed, *this, arch_filter<std::less<int>, 90>);
      static_assert(dims_mixed.extents(cudax::thread, cudax::block) == 256);
      static_assert(dims_mixed.count(cudax::thread, cudax::block) == 256);
      static_assert(dims_mixed.static_count(cudax::thread, cudax::block) == 256);
      static_assert(dims_mixed.static_count(cudax::block, cudax::cluster) == cuda::std::dynamic_extent);
      static_assert(dims_mixed.static_count(cudax::block) == cuda::std::dynamic_extent);
    }
  }
};

TEST_CASE("Cluster dims", "[hierarchy]")
{
  basic_test_cluster().run();
}

TEST_CASE("Different constructions", "[hierarchy]")
{
  const auto block_size  = 512;
  const auto cluster_cnt = 8;
  const auto grid_size   = 256;

  [[maybe_unused]] const auto dimensions2 =
    cudax::block_dims<block_size>() & cudax::cluster_dims<cluster_cnt>() & cudax::grid_dims(grid_size);
  [[maybe_unused]] const auto dimensions3 =
    cudax::grid_dims(grid_size) & cudax::cluster_dims<cluster_cnt>() & cudax::block_dims<block_size>();

  [[maybe_unused]] const auto dimensions4 =
    cudax::cluster_dims<cluster_cnt>() & cudax::grid_dims(grid_size) & cudax::block_dims<block_size>();
  [[maybe_unused]] const auto dimensions5 =
    cudax::cluster_dims<cluster_cnt>() & cudax::block_dims<block_size>() & cudax::grid_dims(grid_size);

  [[maybe_unused]] const auto dimensions6 = cudax::make_hierarchy(
    cudax::block_dims<block_size>(), cudax::cluster_dims<cluster_cnt>(), cudax::grid_dims(grid_size));
  [[maybe_unused]] const auto dimensions7 = cudax::make_hierarchy(
    cudax::grid_dims(grid_size), cudax::cluster_dims<cluster_cnt>(), cudax::block_dims<block_size>());

  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions3)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions4)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions5)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions6)>);
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dimensions7)>);

  [[maybe_unused]] const auto dims_weird_order =
    cudax::grid_dims(grid_size) & (cudax::cluster_dims<cluster_cnt>() & cudax::block_dims<block_size>());
  static_assert(std::is_same_v<decltype(dimensions2), decltype(dims_weird_order)>);

  static_assert(dimensions2.count(cudax::thread, cudax::block) == block_size);
  static_assert(dimensions2.count(cudax::thread, cudax::cluster) == cluster_cnt * block_size);
  static_assert(dimensions2.count(cudax::block, cudax::cluster) == cluster_cnt);
  CUDAX_REQUIRE(dimensions2.count() == grid_size * cluster_cnt * block_size);

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

  const auto replaced = cudax::hierarchy_add_level(fragment, cudax::block_dims(256));
  static_assert(cudax::has_level<cudax::block_level, decltype(replaced)>);
  static_assert(cudax::has_level_or_unit<cudax::thread_level, decltype(replaced)>);
  CUDAX_REQUIRE(replaced.count(cudax::thread, cudax::block) == 256);
}

template <typename Dims>
__global__ void kernel(Dims d)
{
  auto grid  = cg::this_grid();
  auto block = cg::this_thread_block();

  CUDAX_REQUIRE(grid.thread_rank() == (cudax::hierarchy::rank(cudax::thread, cudax::grid)));
  CUDAX_REQUIRE(grid.block_rank() == (cudax::hierarchy::rank(cudax::block, cudax::grid)));
  CUDAX_REQUIRE(grid.thread_rank() == cudax::grid.rank(cudax::thread));
  CUDAX_REQUIRE(grid.block_rank() == cudax::grid.rank(cudax::block));

  CUDAX_REQUIRE(grid.block_index() == (cudax::hierarchy::index(cudax::block, cudax::grid)));
  CUDAX_REQUIRE(grid.block_index() == cudax::grid.index(cudax::block));

  CUDAX_REQUIRE(grid.num_threads() == (cudax::hierarchy::count(cudax::thread, cudax::grid)));
  CUDAX_REQUIRE(grid.num_blocks() == (cudax::hierarchy::count(cudax::block, cudax::grid)));

  CUDAX_REQUIRE(grid.num_threads() == (cudax::grid.count(cudax::thread)));
  CUDAX_REQUIRE(grid.num_blocks() == cudax::grid.count(cudax::block));

  CUDAX_REQUIRE(grid.dim_blocks() == (cudax::hierarchy::extents<cudax::block_level, cudax::grid_level>()));
  CUDAX_REQUIRE(grid.dim_blocks() == cudax::grid.extents(cudax::block));

  CUDAX_REQUIRE(block.thread_rank() == (cudax::hierarchy::rank<cudax::thread_level, cudax::block_level>()));
  CUDAX_REQUIRE(block.thread_index() == (cudax::hierarchy::index<cudax::thread_level, cudax::block_level>()));
  CUDAX_REQUIRE(block.num_threads() == (cudax::hierarchy::count<cudax::thread_level, cudax::block_level>()));
  CUDAX_REQUIRE(block.dim_threads() == (cudax::hierarchy::extents<cudax::thread_level, cudax::block_level>()));

  CUDAX_REQUIRE(block.thread_rank() == cudax::block.rank(cudax::thread));
  CUDAX_REQUIRE(block.thread_index() == cudax::block.index(cudax::thread));
  CUDAX_REQUIRE(block.num_threads() == cudax::block.count(cudax::thread));
  CUDAX_REQUIRE(block.dim_threads() == cudax::block.extents(cudax::thread));

  auto block_index = d.index(cudax::thread, cudax::block);
  CUDAX_REQUIRE(block_index == block.thread_index());
  auto grid_index = d.index();
  CUDAX_REQUIRE(
    grid_index.x
    == static_cast<unsigned long long>(grid.block_index().x) * block.dim_threads().x + block.thread_index().x);
  CUDAX_REQUIRE(
    grid_index.y
    == static_cast<unsigned long long>(grid.block_index().y) * block.dim_threads().y + block.thread_index().y);
  CUDAX_REQUIRE(
    grid_index.z
    == static_cast<unsigned long long>(grid.block_index().z) * block.dim_threads().z + block.thread_index().z);

  CUDAX_REQUIRE(d.rank(cudax::block) == grid.block_rank());
  CUDAX_REQUIRE(d.rank(cudax::thread, cudax::block) == block.thread_rank());
  CUDAX_REQUIRE(d.rank() == grid.thread_rank());
}

TEST_CASE("Dims queries indexing and ambient hierarchy", "[hierarchy]")
{
  const auto dims = cuda::std::make_tuple(
    cudax::block_dims(dim3(64, 4, 2)) & cudax::grid_dims(dim3(12, 6, 3)),
    cudax::block_dims(dim3(2, 4, 64)) & cudax::grid_dims(dim3(3, 6, 12)),
    cudax::block_dims<256>() & cudax::grid_dims<4>(),
    cudax::block_dims<16, 2, 4>() & cudax::grid_dims<2, 3, 4>(),
    cudax::block_dims(dim3(8, 4, 2)) & cudax::grid_dims<4, 5, 6>(),
#if defined(NDEBUG)
    cudax::block_dims<32>() & cudax::grid_dims<(1 << 30) - 2>(),
#endif
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

template <typename Dimensions>
__global__ void examples_kernel(Dimensions dims)
{
  using namespace cuda::experimental;

  {
    auto thread_index_in_block = dims.index(thread, block);
    CUDAX_REQUIRE(thread_index_in_block == threadIdx);
    auto block_index_in_grid = dims.index(block);
    CUDAX_REQUIRE(block_index_in_grid == blockIdx);
  }
  {
    int thread_rank_in_block = dims.rank(thread, block);
    int block_rank_in_grid   = dims.rank(block);
  }
  {
    // Can be called with the instances of level types
    int num_threads_in_block = hierarchy::count(thread, block);
    int num_blocks_in_grid   = grid.count(block);

    // Or using the level types as template arguments
    int num_threads_in_grid = hierarchy::count<thread_level, grid_level>();
  }
  {
    // Can be called with the instances of level types
    int thread_rank_in_block = hierarchy::rank(thread, block);
    int block_rank_in_grid   = grid.rank(block);

    // Or using the level types as template arguments
    int thread_rank_in_grid = hierarchy::rank<thread_level, grid_level>();
  }
  {
    // Can be called with the instances of level types
    auto block_dims = hierarchy::extents(thread, block);
    CUDAX_REQUIRE(block_dims == blockDim);
    auto grid_dims = grid.extents(block);
    CUDAX_REQUIRE(grid_dims == gridDim);

    // Or using the level types as template arguments
    auto grid_dims_in_threads = hierarchy::extents<thread_level, grid_level>();
  }
  {
    // Can be called with the instances of level types
    auto thread_index_in_block = hierarchy::index(thread, block);
    CUDAX_REQUIRE(thread_index_in_block == threadIdx);
    auto block_index_in_grid = grid.index(block);
    CUDAX_REQUIRE(block_index_in_grid == blockIdx);

    // Or using the level types as template arguments
    auto thread_index_in_grid = hierarchy::index<thread_level, grid_level>();
  }
}

// Test examples from the inline rst documentation
TEST_CASE("Examples", "[hierarchy]")
{
  using namespace cuda::experimental;

  {
    auto hierarchy     = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
    auto fragment      = hierarchy.fragment(block, grid);
    auto new_hierarchy = hierarchy_add_level(fragment, block_dims<128>());
    static_assert(new_hierarchy.count(thread, block) == 128);
  }
  {
    auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
    static_assert(hierarchy.count(thread, cluster) == 4 * 8 * 8 * 8);
    CUDAX_REQUIRE(hierarchy.count() == 256 * 4 * 8 * 8 * 8);
    CUDAX_REQUIRE(hierarchy.count(cluster) == 256);
  }
  {
    [[maybe_unused]] auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
    static_assert(hierarchy.static_count(thread, cluster) == 4 * 8 * 8 * 8);
    CUDAX_REQUIRE(hierarchy.static_count() == cuda::std::dynamic_extent);
  }
  {
    auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
    static_assert(hierarchy.extents(thread, cluster).extent(0) == 4 * 8);
    static_assert(hierarchy.extents(thread, cluster).extent(1) == 8);
    static_assert(hierarchy.extents(thread, cluster).extent(2) == 8);
    CUDAX_REQUIRE(hierarchy.extents().extent(0) == 256 * 4 * 8);
    CUDAX_REQUIRE(hierarchy.extents(cluster).extent(0) == 256);
  }
  {
    [[maybe_unused]] auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
    static_assert(decltype(hierarchy.level(cluster).dims)::static_extent(0) == 4);
  }
  {
    auto partial1                    = make_hierarchy_fragment<block_level>(grid_dims(256), cluster_dims<4>());
    [[maybe_unused]] auto hierarchy1 = hierarchy_add_level(partial1, block_dims<8, 8, 8>());
    auto partial2                    = make_hierarchy_fragment<thread_level>(block_dims<8, 8, 8>(), cluster_dims<4>());
    [[maybe_unused]] auto hierarchy2 = hierarchy_add_level(partial2, grid_dims(256));
    static_assert(cuda::std::is_same_v<decltype(hierarchy1), decltype(hierarchy2)>);
  }
  {
    [[maybe_unused]] auto hierarchy1 = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
    [[maybe_unused]] auto hierarchy2 = make_hierarchy(block_dims<8, 8, 8>(), cluster_dims<4>(), grid_dims(256));
    static_assert(cuda::std::is_same_v<decltype(hierarchy1), decltype(hierarchy2)>);
  }
  {
    auto hierarchy = make_hierarchy(grid_dims(256), cluster_dims<4>(), block_dims<8, 8, 8>());
    auto [grid_dimensions, cluster_dimensions, block_dimensions] = get_launch_dimensions(hierarchy);
    CUDAX_REQUIRE(grid_dimensions.x == 256 * 4);
    CUDAX_REQUIRE(cluster_dimensions.x == 4);
    CUDAX_REQUIRE(block_dimensions.x == 8);
    CUDAX_REQUIRE(block_dimensions.y == 8);
    CUDAX_REQUIRE(block_dimensions.z == 8);
  }
  {
    auto hierarchy                           = make_hierarchy(grid_dims(16), block_dims<8, 8, 8>());
    auto [grid_dimensions, block_dimensions] = get_launch_dimensions(hierarchy);
    examples_kernel<<<grid_dimensions, block_dimensions>>>(hierarchy);
    CUDART(cudaGetLastError());
    CUDART(cudaDeviceSynchronize());
  }
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

TEST_CASE("cudax::distribute", "[hierarchy]")
{
  int numElements               = 50000;
  constexpr int threadsPerBlock = 256;
  auto dims                     = cudax::distribute<threadsPerBlock>(numElements);

  CUDAX_REQUIRE(dims.count(cudax::thread, cudax::block) == 256);
  CUDAX_REQUIRE(dims.count(cudax::block, cudax::grid) == (numElements + threadsPerBlock - 1) / threadsPerBlock);
}
