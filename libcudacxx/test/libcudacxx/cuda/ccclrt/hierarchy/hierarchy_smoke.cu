//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <iostream>

#include <cooperative_groups.h>
#include <host_device.cuh>

#include "testing.cuh"

namespace cg = cooperative_groups;

struct basic_test_single_dim
{
  static constexpr int block_size = 256;
  static constexpr int grid_size  = 512;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    // device-side require doesn't work with clang-cuda for now
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(dims.extents().x == grid_size * block_size);
    CCCLRT_REQUIRE(dims.extents(cuda::thread).x == grid_size * block_size);
    CCCLRT_REQUIRE(dims.extents(cuda::thread, cuda::grid).x == grid_size * block_size);
    CCCLRT_REQUIRE(dims.count() == grid_size * block_size);
    CCCLRT_REQUIRE(dims.count(cuda::thread) == grid_size * block_size);
    CCCLRT_REQUIRE(dims.count(cuda::thread, cuda::grid) == grid_size * block_size);

    CCCLRT_REQUIRE(dims.extents(cuda::thread, cuda::block).x == block_size);
    CCCLRT_REQUIRE(dims.extents(cuda::block, cuda::grid).x == grid_size);
    CCCLRT_REQUIRE(dims.count(cuda::thread, cuda::block) == block_size);
    CCCLRT_REQUIRE(dims.count(cuda::block, cuda::grid) == grid_size);
#endif
  }

  void run()
  {
    auto dims = cuda::make_hierarchy(cuda::block_dims<block_size>(), cuda::grid_dims<grid_size>());
    static_assert(dims.extents().x == grid_size * block_size);
    static_assert(dims.extents(cuda::thread).x == grid_size * block_size);
    static_assert(dims.extents(cuda::thread, cuda::grid).x == grid_size * block_size);
    static_assert(dims.count() == grid_size * block_size);
    static_assert(dims.count(cuda::thread) == grid_size * block_size);
    static_assert(dims.count(cuda::thread, cuda::grid) == grid_size * block_size);
    static_assert(dims.static_count() == grid_size * block_size);
    static_assert(dims.static_count(cuda::thread) == grid_size * block_size);
    static_assert(dims.static_count(cuda::thread, cuda::grid) == grid_size * block_size);
    static_assert(dims.static_extents()[0] == grid_size * block_size);
    static_assert(dims.static_extents(cuda::thread)[0] == grid_size * block_size);
    static_assert(dims.static_extents(cuda::thread, cuda::grid)[0] == grid_size * block_size);

    static_assert(dims.extents(cuda::thread, cuda::block).x == block_size);
    static_assert(dims.extents(cuda::block, cuda::grid).x == grid_size);
    static_assert(dims.count(cuda::thread, cuda::block) == block_size);
    static_assert(dims.count(cuda::block, cuda::grid) == grid_size);
    static_assert(dims.static_count(cuda::thread, cuda::block) == block_size);
    static_assert(dims.static_count(cuda::block, cuda::grid) == grid_size);
    static_assert(dims.static_extents(cuda::thread, cuda::block)[0] == block_size);

    auto dims_dyn = cuda::make_hierarchy(cuda::block_dims(block_size), cuda::grid_dims(grid_size));

    test_host_dev(dims_dyn, *this);

    static_assert(dims_dyn.static_count(cuda::thread, cuda::block) == cuda::std::dynamic_extent);
    static_assert(dims_dyn.static_count(cuda::thread, cuda::grid) == cuda::std::dynamic_extent);
    static_assert(dims_dyn.static_extents(cuda::thread, cuda::block)[0] == cuda::std::dynamic_extent);
    static_assert(dims_dyn.static_extents(cuda::thread, cuda::grid)[0] == cuda::std::dynamic_extent);

    // Test that we can also drop the empty parens in the level constructors:
    auto config = cuda::make_hierarchy(cuda::block_dims<block_size>, cuda::grid_dims<grid_size>);
    CCCLRT_REQUIRE(dims == config);
  }
};

struct basic_test_multi_dim
{
  static constexpr int block_size = 256;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    // device-side require doesn't work with clang-cuda for now
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(dims.extents() == dim3(32, 12, 4));
    CCCLRT_REQUIRE(dims.extents(cuda::thread) == dim3(32, 12, 4));
    CCCLRT_REQUIRE(dims.extents(cuda::thread, cuda::grid) == dim3(32, 12, 4));
    CCCLRT_REQUIRE(dims.extents().extent(0) == 32);
    CCCLRT_REQUIRE(dims.extents().extent(1) == 12);
    CCCLRT_REQUIRE(dims.extents().extent(2) == 4);
    CCCLRT_REQUIRE(dims.count() == 512 * 3);
    CCCLRT_REQUIRE(dims.count(cuda::thread) == 512 * 3);
    CCCLRT_REQUIRE(dims.count(cuda::thread, cuda::grid) == 512 * 3);

    CCCLRT_REQUIRE(dims.extents(cuda::thread, cuda::block) == dim3(2, 3, 4));
    CCCLRT_REQUIRE(dims.extents(cuda::block, cuda::grid) == dim3(16, 4, 1));
    CCCLRT_REQUIRE(dims.count(cuda::thread, cuda::block) == 24);
    CCCLRT_REQUIRE(dims.count(cuda::block, cuda::grid) == 64);
#endif
  }

  void run()
  {
    auto dims_multidim = cuda::make_hierarchy(cuda::block_dims<2, 3, 4>(), cuda::grid_dims<16, 4, 1>());

    static_assert(dims_multidim.extents() == dim3(32, 12, 4));
    static_assert(dims_multidim.extents(cuda::thread) == dim3(32, 12, 4));
    static_assert(dims_multidim.extents(cuda::thread, cuda::grid) == dim3(32, 12, 4));
    static_assert(dims_multidim.extents().extent(0) == 32);
    static_assert(dims_multidim.extents().extent(1) == 12);
    static_assert(dims_multidim.extents().extent(2) == 4);
    static_assert(dims_multidim.count() == 512 * 3);
    static_assert(dims_multidim.count(cuda::thread) == 512 * 3);
    static_assert(dims_multidim.count(cuda::thread, cuda::grid) == 512 * 3);
    static_assert(dims_multidim.static_count() == 512 * 3);
    static_assert(dims_multidim.static_count(cuda::thread) == 512 * 3);
    static_assert(dims_multidim.static_count(cuda::thread, cuda::grid) == 512 * 3);
    static_assert(dims_multidim.static_extents() == cuda::std::array<cuda::std::size_t, 3>{32, 12, 4});
    static_assert(dims_multidim.static_extents(cuda::thread) == cuda::std::array<cuda::std::size_t, 3>{32, 12, 4});
    static_assert(
      dims_multidim.static_extents(cuda::thread, cuda::grid) == cuda::std::array<cuda::std::size_t, 3>{32, 12, 4});

    static_assert(dims_multidim.extents(cuda::thread, cuda::block) == dim3(2, 3, 4));
    static_assert(dims_multidim.extents(cuda::block, cuda::grid) == dim3(16, 4, 1));
    static_assert(dims_multidim.count(cuda::thread, cuda::block) == 24);
    static_assert(dims_multidim.count(cuda::block, cuda::grid) == 64);
    static_assert(dims_multidim.static_count(cuda::thread, cuda::block) == 24);
    static_assert(dims_multidim.static_count(cuda::block, cuda::grid) == 64);
    static_assert(
      dims_multidim.static_extents(cuda::thread, cuda::block) == cuda::std::array<cuda::std::size_t, 3>{2, 3, 4});
    static_assert(
      dims_multidim.static_extents(cuda::block, cuda::grid) == cuda::std::array<cuda::std::size_t, 3>{16, 4, 1});

    auto dims_multidim_dyn = cuda::make_hierarchy(cuda::block_dims(dim3(2, 3, 4)), cuda::grid_dims(dim3(16, 4, 1)));

    test_host_dev(dims_multidim_dyn, *this);

    static_assert(dims_multidim_dyn.static_count(cuda::thread, cuda::block) == cuda::std::dynamic_extent);
    static_assert(dims_multidim_dyn.static_count(cuda::thread, cuda::grid) == cuda::std::dynamic_extent);
  }
};

struct basic_test_mixed
{
  static constexpr int block_size = 256;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    // device-side require doesn't work with clang-cuda for now
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(dims.extents() == dim3(2048, 4, 2));
    CCCLRT_REQUIRE(dims.extents(cuda::thread) == dim3(2048, 4, 2));
    CCCLRT_REQUIRE(dims.extents(cuda::thread, cuda::grid) == dim3(2048, 4, 2));
    CCCLRT_REQUIRE(dims.extents().extent(0) == 2048);
    CCCLRT_REQUIRE(dims.extents().extent(1) == 4);
    CCCLRT_REQUIRE(dims.extents().extent(2) == 2);
    CCCLRT_REQUIRE(dims.count() == 16 * 1024);
    CCCLRT_REQUIRE(dims.count(cuda::thread) == 16 * 1024);
    CCCLRT_REQUIRE(dims.count(cuda::thread, cuda::grid) == 16 * 1024);

    CCCLRT_REQUIRE(dims.extents(cuda::block, cuda::grid) == dim3(8, 4, 2));
    CCCLRT_REQUIRE(dims.count(cuda::block, cuda::grid) == 64);
#endif
  }

  void run()
  {
    auto dims_mixed = cuda::make_hierarchy(cuda::block_dims<block_size>(), cuda::grid_dims(dim3(8, 4, 2)));

    test_host_dev(dims_mixed, *this);
    static_assert(dims_mixed.extents(cuda::thread, cuda::block) == block_size);
    static_assert(dims_mixed.count(cuda::thread, cuda::block) == block_size);
    static_assert(dims_mixed.static_count(cuda::thread, cuda::block) == block_size);
    static_assert(dims_mixed.static_count(cuda::block, cuda::grid) == cuda::std::dynamic_extent);
    static_assert(dims_mixed.static_extents(cuda::thread, cuda::block)[0] == block_size);

    // TODO include mixed static and dynamic info on a single level
    // Currently bugged in std::extents
  }
};

C2H_TEST("Basic", "[hierarchy]")
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
    // device-side require doesn't work with clang-cuda for now
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(dims.extents() == dim3(512, 6, 9));
    CCCLRT_REQUIRE(dims.count() == 27 * 1024);

    CCCLRT_REQUIRE(dims.extents(cuda::block, cuda::grid) == dim3(2, 6, 9));
    CCCLRT_REQUIRE(dims.count(cuda::block, cuda::grid) == 108);
    CCCLRT_REQUIRE(dims.extents(cuda::cluster, cuda::grid) == dim3(1, 3, 9));
    CCCLRT_REQUIRE(dims.extents(cuda::thread, cuda::cluster) == dim3(512, 2, 1));
#endif
  }

  void run()
  {
    SECTION("Static cluster dims")
    {
      auto dims = cuda::make_hierarchy(cuda::block_dims<256>(), cuda::cluster_dims<8>(), cuda::grid_dims<512>());

      static_assert(dims.extents().x == 1024 * 1024);
      static_assert(dims.count() == 1024 * 1024);
      static_assert(dims.static_count() == 1024 * 1024);
      static_assert(dims.static_extents()[0] == 1024 * 1024);

      static_assert(dims.extents(cuda::thread, cuda::block).x == 256);
      static_assert(dims.extents(cuda::block, cuda::grid).x == 4 * 1024);
      static_assert(dims.count(cuda::thread, cuda::cluster) == 2 * 1024);
      static_assert(dims.count(cuda::cluster) == 512);
      static_assert(dims.static_count(cuda::cluster) == 512);
      static_assert(dims.static_count(cuda::block, cuda::cluster) == 8);
      static_assert(dims.static_extents(cuda::thread, cuda::block)[0] == 256);
      static_assert(dims.static_extents(cuda::block, cuda::grid)[0] == 4 * 1024);
    }
    SECTION("Mixed cluster dims")
    {
      auto dims_mixed = cuda::make_hierarchy(
        cuda::block_dims<256>(), cuda::cluster_dims(dim3(2, 2, 1)), cuda::grid_dims(dim3(1, 3, 9)));
      test_host_dev(dims_mixed, *this, arch_filter<std::less<int>, 90>);
      static_assert(dims_mixed.extents(cuda::thread, cuda::block) == 256);
      static_assert(dims_mixed.count(cuda::thread, cuda::block) == 256);
      static_assert(dims_mixed.static_count(cuda::thread, cuda::block) == 256);
      static_assert(dims_mixed.static_count(cuda::block, cuda::cluster) == cuda::std::dynamic_extent);
      static_assert(dims_mixed.static_count(cuda::block) == cuda::std::dynamic_extent);
      static_assert(dims_mixed.static_extents(cuda::thread, cuda::block)[0] == 256);
      static_assert(dims_mixed.static_extents(cuda::block, cuda::cluster)[0] == cuda::std::dynamic_extent);
      static_assert(dims_mixed.static_extents(cuda::block)[0] == cuda::std::dynamic_extent);
    }
  }
};

C2H_TEST("Cluster dims", "[hierarchy]")
{
  basic_test_cluster().run();
}

C2H_TEST("Different constructions", "[hierarchy]")
{
  /*
  const auto block_size  = 512;
  const auto cluster_cnt = 8;
  const auto grid_size   = 256;

  [[maybe_unused]] const auto config =
    cuda::block_dims<block_size>() & cuda::cluster_dims<cluster_cnt>() &
  cuda::grid_dims(grid_size);
  [[maybe_unused]] const auto config2 =
    cuda::grid_dims(grid_size) & cuda::cluster_dims<cluster_cnt>() &
  cuda::block_dims<block_size>();

  [[maybe_unused]] const auto config3 =
    cuda::cluster_dims<cluster_cnt>() & cuda::grid_dims(grid_size) &
  cuda::block_dims<block_size>();
  [[maybe_unused]] const auto config4 =
    cuda::cluster_dims<cluster_cnt>() & cuda::block_dims<block_size>() &
  cuda::grid_dims(grid_size);

  [[maybe_unused]] const auto config5 =
    cuda::make_config(cuda::block_dims<block_size>(),
  cuda::cluster_dims<cluster_cnt>(), cuda::grid_dims(grid_size));
  [[maybe_unused]] const auto config6 =
    cuda::make_config(cuda::grid_dims(grid_size),
  cuda::cluster_dims<cluster_cnt>(), cuda::block_dims<block_size>());

  static_assert(std::is_same_v<decltype(config), decltype(config2)>);
  static_assert(std::is_same_v<decltype(config), decltype(config3)>);
  static_assert(std::is_same_v<decltype(config), decltype(config4)>);
  static_assert(std::is_same_v<decltype(config), decltype(config5)>);
  static_assert(std::is_same_v<decltype(config), decltype(config6)>);

  [[maybe_unused]] const auto conf_weird_order =
    cuda::grid_dims(grid_size) & (cuda::cluster_dims<cluster_cnt>() &
  cuda::block_dims<block_size>());
  static_assert(std::is_same_v<decltype(config), decltype(conf_weird_order)>);

  static_assert(config.dims.count(cuda::thread, cuda::block) == block_size);
  static_assert(config.dims.count(cuda::thread, cuda::cluster) == cluster_cnt *
  block_size); static_assert(config.dims.count(cuda::block, cuda::cluster) ==
  cluster_cnt); CCCLRT_REQUIRE(config.dims.count() == grid_size * cluster_cnt *
  block_size);

  static_assert(cuda::has_level<cuda::block_level, decltype(config.dims)>);
  static_assert(cuda::has_level<cuda::cluster_level, decltype(config.dims)>);
  static_assert(cuda::has_level<cuda::grid_level, decltype(config.dims)>);
  static_assert(!cuda::has_level<cuda::thread_level, decltype(config.dims)>);
  */
}

C2H_TEST("Replace level", "[hierarchy]")
{
  const auto dimensions = cuda::make_hierarchy(cuda::block_dims<512>(), cuda::cluster_dims<8>(), cuda::grid_dims(256));
  const auto fragment   = dimensions.fragment(cuda::block, cuda::grid);
  static_assert(!cuda::has_level<cuda::block_level, decltype(fragment)>);
  static_assert(!cuda::has_level_or_unit<cuda::thread_level, decltype(fragment)>);
  static_assert(cuda::has_level<cuda::cluster_level, decltype(fragment)>);
  static_assert(cuda::has_level<cuda::grid_level, decltype(fragment)>);
  static_assert(cuda::has_level_or_unit<cuda::block_level, decltype(fragment)>);

  const auto replaced = cuda::hierarchy_add_level(fragment, cuda::block_dims(256));
  static_assert(cuda::has_level<cuda::block_level, decltype(replaced)>);
  static_assert(cuda::has_level_or_unit<cuda::thread_level, decltype(replaced)>);
  CCCLRT_REQUIRE(replaced.count(cuda::thread, cuda::block) == 256);
}

template <typename Hierarchy>
__global__ void kernel(Hierarchy hierarchy)
{
  auto grid  = cg::this_grid();
  auto block = cg::this_thread_block();

  CCCLRT_REQUIRE_DEVICE(grid.thread_rank() == (cuda::hierarchy::rank(cuda::thread, cuda::grid)));
  CCCLRT_REQUIRE_DEVICE(grid.block_rank() == (cuda::hierarchy::rank(cuda::block, cuda::grid)));
  CCCLRT_REQUIRE_DEVICE(grid.thread_rank() == cuda::grid.rank(cuda::thread));
  CCCLRT_REQUIRE_DEVICE(grid.block_rank() == cuda::grid.rank(cuda::block));

  CCCLRT_REQUIRE_DEVICE(grid.block_index() == (cuda::hierarchy::index(cuda::block, cuda::grid)));
  CCCLRT_REQUIRE_DEVICE(grid.block_index() == cuda::grid.index(cuda::block));

  CCCLRT_REQUIRE_DEVICE(grid.num_threads() == (cuda::hierarchy::count(cuda::thread, cuda::grid)));
  CCCLRT_REQUIRE_DEVICE(grid.num_blocks() == (cuda::hierarchy::count(cuda::block, cuda::grid)));

  CCCLRT_REQUIRE_DEVICE(grid.num_threads() == (cuda::grid.count(cuda::thread)));
  CCCLRT_REQUIRE_DEVICE(grid.num_blocks() == cuda::grid.count(cuda::block));

  CCCLRT_REQUIRE_DEVICE(grid.dim_blocks() == (cuda::hierarchy::extents<cuda::block_level, cuda::grid_level>()));
  CCCLRT_REQUIRE_DEVICE(grid.dim_blocks() == cuda::grid.extents(cuda::block));

  CCCLRT_REQUIRE_DEVICE(block.thread_rank() == (cuda::hierarchy::rank<cuda::thread_level, cuda::block_level>()));
  CCCLRT_REQUIRE_DEVICE(block.thread_index() == (cuda::hierarchy::index<cuda::thread_level, cuda::block_level>()));
  CCCLRT_REQUIRE_DEVICE(block.num_threads() == (cuda::hierarchy::count<cuda::thread_level, cuda::block_level>()));
  CCCLRT_REQUIRE_DEVICE(block.dim_threads() == (cuda::hierarchy::extents<cuda::thread_level, cuda::block_level>()));

  CCCLRT_REQUIRE_DEVICE(block.thread_rank() == cuda::block.rank(cuda::thread));
  CCCLRT_REQUIRE_DEVICE(block.thread_index() == cuda::block.index(cuda::thread));
  CCCLRT_REQUIRE_DEVICE(block.num_threads() == cuda::block.count(cuda::thread));
  CCCLRT_REQUIRE_DEVICE(block.dim_threads() == cuda::block.extents(cuda::thread));

  auto block_index = hierarchy.index(cuda::thread, cuda::block);
  CCCLRT_REQUIRE_DEVICE(block_index == block.thread_index());
  auto grid_index = hierarchy.index();
  CCCLRT_REQUIRE_DEVICE(
    grid_index.x
    == static_cast<unsigned long long>(grid.block_index().x) * block.dim_threads().x + block.thread_index().x);
  CCCLRT_REQUIRE_DEVICE(
    grid_index.y
    == static_cast<unsigned long long>(grid.block_index().y) * block.dim_threads().y + block.thread_index().y);
  CCCLRT_REQUIRE_DEVICE(
    grid_index.z
    == static_cast<unsigned long long>(grid.block_index().z) * block.dim_threads().z + block.thread_index().z);

  CCCLRT_REQUIRE_DEVICE(hierarchy.rank(cuda::block) == grid.block_rank());
  CCCLRT_REQUIRE_DEVICE(hierarchy.rank(cuda::thread, cuda::block) == block.thread_rank());
  CCCLRT_REQUIRE_DEVICE(hierarchy.rank() == grid.thread_rank());
}

C2H_TEST("Dims queries indexing and ambient hierarchy", "[hierarchy]")
{
  const auto hierarchies = cuda::std::make_tuple(
    cuda::make_hierarchy(cuda::block_dims(dim3(64, 4, 2)), cuda::grid_dims(dim3(12, 6, 3))),
    cuda::make_hierarchy(cuda::block_dims(dim3(2, 4, 64)), cuda::grid_dims(dim3(3, 6, 12))),
    cuda::make_hierarchy(cuda::block_dims<256>(), cuda::grid_dims<4>()),
    cuda::make_hierarchy(cuda::block_dims<16, 2, 4>(), cuda::grid_dims<2, 3, 4>()),
    cuda::make_hierarchy(cuda::block_dims(dim3(8, 4, 2)), cuda::grid_dims<4, 5, 6>()),
#if defined(NDEBUG)
    cuda::make_hierarchy(cuda::block_dims<32>(), cuda::grid_dims<(1 << 30) - 2>()),
#endif
    cuda::make_hierarchy(cuda::block_dims<8, 2, 4>(), cuda::grid_dims(dim3(5, 4, 3))));

  apply_each(
    [](const auto& hierarchy) {
      auto [grid, block] = cuda::get_launch_dimensions(hierarchy);

      kernel<<<grid, block>>>(hierarchy);
      CUDART(cudaDeviceSynchronize());
    },
    hierarchies);
}

template <typename Hierarchy>
__global__ void rank_kernel_optimized(Hierarchy hierarchy, unsigned int* out)
{
  auto thread_id = hierarchy.rank(cuda::thread, cuda::block);
  out[thread_id] = thread_id;
}

template <typename Hierarchy>
__global__ void rank_kernel(Hierarchy hierarchy, unsigned int* out)
{
  auto thread_id = cuda::hierarchy::rank(cuda::thread, cuda::block);
  out[thread_id] = thread_id;
}

template <typename Hierarchy>
__global__ void rank_kernel_cg(Hierarchy hierarchy, unsigned int* out)
{
  auto thread_id = cg::thread_block::thread_rank();
  out[thread_id] = thread_id;
}

// Testcase mostly for generated code comparison
C2H_TEST("On device rank calculation", "[hierarchy]")
{
  unsigned int* ptr;
  CUDART(cudaMalloc((void**) &ptr, 2 * 1024 * sizeof(unsigned int)));

  const auto hierarchy_static = cuda::make_hierarchy(cuda::block_dims<256>(), cuda::grid_dims(dim3(2, 2, 2)));
  rank_kernel<<<dim3(2, 2, 2), 256>>>(hierarchy_static, ptr);
  CUDART(cudaDeviceSynchronize());
  rank_kernel_cg<<<dim3(2, 2, 2), 256>>>(hierarchy_static, ptr);
  CUDART(cudaDeviceSynchronize());
  rank_kernel_optimized<<<dim3(2, 2, 2), 256>>>(hierarchy_static, ptr);
  CUDART(cudaDeviceSynchronize());
  CUDART(cudaFree(ptr));
}

template <typename Hierarchy>
__global__ void examples_kernel(Hierarchy hierarchy)
{
  {
    auto thread_index_in_block = hierarchy.index(cuda::thread, cuda::block);
    CCCLRT_REQUIRE_DEVICE(thread_index_in_block == threadIdx);
    auto block_index_in_grid = hierarchy.index(cuda::block);
    CCCLRT_REQUIRE_DEVICE(block_index_in_grid == blockIdx);
  }
  {
    int thread_rank_in_block = hierarchy.rank(cuda::thread, cuda::block);
    int block_rank_in_grid   = hierarchy.rank(cuda::block);
  }
  {
    // Can be called with the instances of level types
    int num_threads_in_block = cuda::hierarchy::count(cuda::thread, cuda::block);
    int num_blocks_in_grid   = cuda::grid.count(cuda::block);

    // Or using the level types as template arguments
    int num_threads_in_grid = cuda::hierarchy::count<cuda::thread_level, cuda::grid_level>();
  }
  {
    // Can be called with the instances of level types
    int thread_rank_in_block = cuda::hierarchy::rank(cuda::thread, cuda::block);
    int block_rank_in_grid   = cuda::grid.rank(cuda::block);

    // Or using the level types as template arguments
    int thread_rank_in_grid = cuda::hierarchy::rank<cuda::thread_level, cuda::grid_level>();
  }
  {
    // Can be called with the instances of level types
    auto block_dims = cuda::hierarchy::extents(cuda::thread, cuda::block);
    CCCLRT_REQUIRE_DEVICE(block_dims == blockDim);
    auto grid_dims = cuda::grid.extents(cuda::block);
    CCCLRT_REQUIRE_DEVICE(grid_dims == gridDim);

    // Or using the level types as template arguments
    auto grid_dims_in_threads = cuda::hierarchy::extents<cuda::thread_level, cuda::grid_level>();
  }
  {
    // Can be called with the instances of level types
    auto thread_index_in_block = cuda::hierarchy::index(cuda::thread, cuda::block);
    CCCLRT_REQUIRE_DEVICE(thread_index_in_block == threadIdx);
    auto block_index_in_grid = cuda::grid.index(cuda::block);
    CCCLRT_REQUIRE_DEVICE(block_index_in_grid == blockIdx);

    // Or using the level types as template arguments
    auto thread_index_in_grid = cuda::hierarchy::index<cuda::thread_level, cuda::grid_level>();
  }
}

// Test examples from the inline rst documentation
C2H_TEST("Examples", "[hierarchy]")
{
  // GCC 7 and 8 complains here that the hierarchy was not declared constexpr
#if !_CCCL_COMPILER(GCC) || _CCCL_COMPILER(GCC, >, 8)
  {
    auto hierarchy = cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    auto fragment  = hierarchy.fragment(cuda::block, cuda::grid);
    auto new_hierarchy = cuda::hierarchy_add_level(fragment, cuda::block_dims<128>());
    static_assert(new_hierarchy.count(cuda::thread, cuda::block) == 128);
  }
  {
    auto hierarchy = cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(hierarchy.count(cuda::thread, cuda::cluster) == 4 * 8 * 8 * 8);
    CCCLRT_REQUIRE(hierarchy.count() == 256 * 4 * 8 * 8 * 8);
    CCCLRT_REQUIRE(hierarchy.count(cuda::cluster) == 256);
  }
  {
    [[maybe_unused]] auto hierarchy =
      cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(hierarchy.static_count(cuda::thread, cuda::cluster) == 4 * 8 * 8 * 8);
    CCCLRT_REQUIRE(hierarchy.static_count() == cuda::std::dynamic_extent);
  }
  {
    auto hierarchy = cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(hierarchy.extents(cuda::thread, cuda::cluster).extent(0) == 4 * 8);
    static_assert(hierarchy.extents(cuda::thread, cuda::cluster).extent(1) == 8);
    static_assert(hierarchy.extents(cuda::thread, cuda::cluster).extent(2) == 8);
    CCCLRT_REQUIRE(hierarchy.extents().extent(0) == 256 * 4 * 8);
    CCCLRT_REQUIRE(hierarchy.extents(cuda::cluster).extent(0) == 256);
  }
#endif
  {
    [[maybe_unused]] auto hierarchy =
      cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(decltype(hierarchy.level(cuda::cluster).dims)::static_extent(0) == 4);
  }
  {
    auto partial1 = cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims(256), cuda::cluster_dims<4>());
    [[maybe_unused]] auto hierarchy1 = cuda::hierarchy_add_level(partial1, cuda::block_dims<8, 8, 8>());
    auto partial2 = cuda::make_hierarchy<cuda::thread_level>(cuda::block_dims<8, 8, 8>(), cuda::cluster_dims<4>());
    [[maybe_unused]] auto hierarchy2 = cuda::hierarchy_add_level(partial2, cuda::grid_dims(256));
    static_assert(cuda::std::is_same_v<decltype(hierarchy1), decltype(hierarchy2)>);
  }
  {
    [[maybe_unused]] auto hierarchy1 =
      cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    [[maybe_unused]] auto hierarchy2 =
      cuda::make_hierarchy(cuda::block_dims<8, 8, 8>(), cuda::cluster_dims<4>(), cuda::grid_dims(256));
    static_assert(cuda::std::is_same_v<decltype(hierarchy1), decltype(hierarchy2)>);
  }
  {
    auto hierarchy = cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    auto [grid_dimensions, cluster_dimensions, block_dimensions] = cuda::get_launch_dimensions(hierarchy);
    CCCLRT_REQUIRE(grid_dimensions.x == 256 * 4);
    CCCLRT_REQUIRE(cluster_dimensions.x == 4);
    CCCLRT_REQUIRE(block_dimensions.x == 8);
    CCCLRT_REQUIRE(block_dimensions.y == 8);
    CCCLRT_REQUIRE(block_dimensions.z == 8);
  }
  {
    auto hierarchy                           = cuda::make_hierarchy(cuda::grid_dims(16), cuda::block_dims<8, 8, 8>());
    auto [grid_dimensions, block_dimensions] = cuda::get_launch_dimensions(hierarchy);
    examples_kernel<<<grid_dimensions, block_dimensions>>>(hierarchy);
    CUDART(cudaGetLastError());
    CUDART(cudaDeviceSynchronize());
  }
}

C2H_TEST("Trivially constructable", "[hierarchy]")
{
  // static_assert(std::is_trivial_v<decltype(cuda::block_dims(256))>);
  // static_assert(std::is_trivial_v<decltype(cuda::block_dims<256>())>);

  // Hierarchy is not trivially copyable (yet), because tuple is not
  // static_assert(std::is_trivially_copyable_v<decltype(cuda::block_dims<256>()
  // & cuda::grid_dims<256>())>);
  // static_assert(std::is_trivially_copyable_v<decltype(cuda::std::make_tuple(cuda::block_dims<256>(),
  // cuda::grid_dims<256>()))>);
}

C2H_TEST("cuda::distribute", "[hierarchy]")
{
  /*
  int numElements               = 50000;
  constexpr int threadsPerBlock = 256;
  auto config                   =
  cuda::distribute<threadsPerBlock>(numElements);

  CCCLRT_REQUIRE(config.dims.count(cuda::thread, cuda::block) == 256);
  CCCLRT_REQUIRE(config.dims.count(cuda::block, cuda::grid) == (numElements +
  threadsPerBlock - 1) / threadsPerBlock);
  */
}

C2H_TEST("hierarchy merge", "[hierarchy]")
{
  SECTION("Non overlapping")
  {
    auto h1       = cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<2>());
    auto h2       = cuda::make_hierarchy<cuda::thread_level>(cuda::block_dims<3>());
    auto combined = h1.combine(h2);
    static_assert(combined.count(cuda::thread) == 6);
    static_assert(combined.count(cuda::thread, cuda::block) == 3);
    static_assert(combined.count(cuda::block) == 2);
    auto combined_the_other_way = h2.combine(h1);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_the_other_way)>);
    static_assert(combined_the_other_way.count(cuda::thread) == 6);

    auto dynamic_values   = cuda::make_hierarchy(cuda::cluster_dims(4), cuda::block_dims(5));
    auto combined_dynamic = dynamic_values.combine(h1);
    CCCLRT_REQUIRE(combined_dynamic.count(cuda::thread) == 40);
  }
  SECTION("Overlapping")
  {
    auto h1       = cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<2>(), cuda::cluster_dims<3>());
    auto h2       = cuda::make_hierarchy<cuda::thread_level>(cuda::block_dims<4>(), cuda::cluster_dims<5>());
    auto combined = h1.combine(h2);
    static_assert(combined.count(cuda::thread) == 24);
    static_assert(combined.count(cuda::thread, cuda::block) == 4);
    static_assert(combined.count(cuda::block) == 6);

    auto combined_the_other_way = h2.combine(h1);
    static_assert(!cuda::std::is_same_v<decltype(combined), decltype(combined_the_other_way)>);
    static_assert(combined_the_other_way.count(cuda::thread) == 40);
    static_assert(combined_the_other_way.count(cuda::thread, cuda::block) == 4);
    static_assert(combined_the_other_way.count(cuda::block) == 10);

    auto ultimate_combination = combined.combine(combined_the_other_way);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(ultimate_combination)>);
    static_assert(ultimate_combination.count(cuda::thread) == 24);

    auto block_level_replacement = cuda::make_hierarchy<cuda::thread_level>(cuda::block_dims<6>());
    auto with_block_replaced     = block_level_replacement.combine(combined);
    static_assert(with_block_replaced.count(cuda::thread) == 36);
    static_assert(with_block_replaced.count(cuda::thread, cuda::block) == 6);

    auto grid_cluster_level_replacement =
      cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<7>(), cuda::cluster_dims<8>());
    auto with_grid_cluster_replaced = grid_cluster_level_replacement.combine(combined);
    static_assert(with_grid_cluster_replaced.count(cuda::thread) == 7 * 8 * 4);
    static_assert(with_grid_cluster_replaced.count(cuda::block, cuda::cluster) == 8);
    static_assert(with_grid_cluster_replaced.count(cuda::cluster) == 7);
  }
}
