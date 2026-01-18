//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__type_traits/vector_type.h>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>

#include <iostream>

#include <cooperative_groups.h>
#include <host_device.cuh>

#include "testing.cuh"

namespace cg = cooperative_groups;

using size_t3 = cuda::__vector_type_t<cuda::std::size_t, 3>;

struct basic_test_single_dim
{
  static constexpr int block_size = 256;
  static constexpr int grid_size  = 512;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    // todo: allow this after fixing CCCLRT_REQUIRE with clang-cuda
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::grid, dims).x == grid_size * block_size);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, dims) == grid_size * block_size);

    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::block, dims).x == block_size);
    CCCLRT_REQUIRE(cuda::block.dims(cuda::grid, dims).x == grid_size);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::block, dims) == block_size);
    CCCLRT_REQUIRE(cuda::block.count(cuda::grid, dims) == grid_size);
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  }

  void run()
  {
    auto dims = cuda::make_hierarchy(cuda::block_dims<block_size>(), cuda::grid_dims<grid_size>());
    static_assert(cuda::gpu_thread.dims(cuda::grid, dims).x == grid_size * block_size);
    static_assert(cuda::gpu_thread.count(cuda::grid, dims) == grid_size * block_size);
    static_assert(cuda::gpu_thread.static_dims(cuda::grid, dims)[0] == grid_size * block_size);

    static_assert(cuda::gpu_thread.dims(cuda::block, dims).x == block_size);
    static_assert(cuda::block.dims(cuda::grid, dims).x == grid_size);
    static_assert(cuda::gpu_thread.count(cuda::block, dims) == block_size);
    static_assert(cuda::block.count(cuda::grid, dims) == grid_size);
    static_assert(cuda::gpu_thread.static_dims(cuda::block, dims)[0] == block_size);

    auto dims_dyn = cuda::make_hierarchy(cuda::block_dims(block_size), cuda::grid_dims(grid_size));

    test_host_dev(dims_dyn, *this);

    static_assert(cuda::gpu_thread.static_dims(cuda::block, dims_dyn)[0] == cuda::std::dynamic_extent);
    static_assert(cuda::gpu_thread.static_dims(cuda::grid, dims_dyn)[0] == cuda::std::dynamic_extent);

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
    // todo: allow this after fixing CCCLRT_REQUIRE with clang-cuda
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::grid, dims) == dim3(32, 12, 4));
    CCCLRT_REQUIRE(cuda::gpu_thread.extents(cuda::grid, dims).extent(0) == 32);
    CCCLRT_REQUIRE(cuda::gpu_thread.extents(cuda::grid, dims).extent(1) == 12);
    CCCLRT_REQUIRE(cuda::gpu_thread.extents(cuda::grid, dims).extent(2) == 4);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, dims) == 512 * 3);

    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::block, dims) == dim3(2, 3, 4));
    CCCLRT_REQUIRE(cuda::block.dims(cuda::grid, dims) == dim3(16, 4, 1));
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::block, dims) == 24);
    CCCLRT_REQUIRE(cuda::block.count(cuda::grid, dims) == 64);
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  }

  void run()
  {
    auto dims_multidim = cuda::make_hierarchy(cuda::block_dims<2, 3, 4>(), cuda::grid_dims<16, 4, 1>());

    static_assert(cuda::gpu_thread.dims(cuda::grid, dims_multidim) == dim3(32, 12, 4));
    static_assert(cuda::gpu_thread.extents(cuda::grid, dims_multidim).extent(0) == 32);
    static_assert(cuda::gpu_thread.extents(cuda::grid, dims_multidim).extent(1) == 12);
    static_assert(cuda::gpu_thread.extents(cuda::grid, dims_multidim).extent(2) == 4);
    static_assert(cuda::gpu_thread.count(cuda::grid, dims_multidim) == 512 * 3);
    static_assert(cuda::gpu_thread.static_dims(cuda::grid, dims_multidim) == size_t3{32, 12, 4});

    static_assert(cuda::gpu_thread.dims(cuda::block, dims_multidim) == dim3(2, 3, 4));
    static_assert(cuda::block.dims(cuda::grid, dims_multidim) == dim3(16, 4, 1));
    static_assert(cuda::gpu_thread.count(cuda::block, dims_multidim) == 24);
    static_assert(cuda::block.count(cuda::grid, dims_multidim) == 64);
    static_assert(cuda::gpu_thread.static_dims(cuda::block, dims_multidim) == size_t3{2, 3, 4});
    static_assert(cuda::block.static_dims(cuda::grid, dims_multidim) == size_t3{16, 4, 1});

    auto dims_multidim_dyn = cuda::make_hierarchy(cuda::block_dims(dim3(2, 3, 4)), cuda::grid_dims(dim3(16, 4, 1)));

    test_host_dev(dims_multidim_dyn, *this);
  }
};

struct basic_test_mixed
{
  static constexpr int block_size = 256;

  template <typename DynDims>
  __host__ __device__ void operator()(const DynDims& dims) const
  {
    // todo: allow this after fixing CCCLRT_REQUIRE with clang-cuda
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::grid, dims) == dim3(2048, 4, 2));
    CCCLRT_REQUIRE(cuda::gpu_thread.extents(cuda::grid, dims).extent(0) == 2048);
    CCCLRT_REQUIRE(cuda::gpu_thread.extents(cuda::grid, dims).extent(1) == 4);
    CCCLRT_REQUIRE(cuda::gpu_thread.extents(cuda::grid, dims).extent(2) == 2);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, dims) == 16 * 1024);

    CCCLRT_REQUIRE(cuda::block.dims(cuda::grid, dims) == dim3(8, 4, 2));
    CCCLRT_REQUIRE(cuda::block.count(cuda::grid, dims) == 64);
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  }

  void run()
  {
    auto dims_mixed = cuda::make_hierarchy(cuda::block_dims<block_size>(), cuda::grid_dims(dim3(8, 4, 2)));

    test_host_dev(dims_mixed, *this);
    static_assert(cuda::gpu_thread.dims(cuda::block, dims_mixed).x == block_size);
    static_assert(cuda::gpu_thread.count(cuda::block, dims_mixed) == block_size);
    static_assert(cuda::gpu_thread.static_dims(cuda::block, dims_mixed)[0] == block_size);

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
    // todo: allow this after fixing CCCLRT_REQUIRE with clang-cuda
#if !_CCCL_CUDA_COMPILER(CLANG)
    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::grid, dims) == dim3(512, 6, 9));
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, dims) == 27 * 1024);

    CCCLRT_REQUIRE(cuda::block.dims(cuda::grid, dims) == dim3(2, 6, 9));
    CCCLRT_REQUIRE(cuda::block.count(cuda::grid, dims) == 108);
    CCCLRT_REQUIRE(cuda::cluster.dims(cuda::grid, dims) == dim3(1, 3, 9));
    CCCLRT_REQUIRE(cuda::gpu_thread.dims(cuda::cluster, dims) == dim3(512, 2, 1));
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  }

  void run()
  {
    SECTION("Static cluster dims")
    {
      auto dims = cuda::make_hierarchy(cuda::block_dims<256>(), cuda::cluster_dims<8>(), cuda::grid_dims<512>());

      static_assert(cuda::gpu_thread.dims(cuda::grid, dims).x == 1024 * 1024);
      static_assert(cuda::gpu_thread.count(cuda::grid, dims) == 1024 * 1024);
      static_assert(cuda::gpu_thread.static_dims(cuda::grid, dims)[0] == 1024 * 1024);

      static_assert(cuda::gpu_thread.dims(cuda::block, dims).x == 256);
      static_assert(cuda::block.dims(cuda::grid, dims).x == 4 * 1024);
      static_assert(cuda::gpu_thread.count(cuda::cluster, dims) == 2 * 1024);
      static_assert(cuda::cluster.count(cuda::grid, dims) == 512);
      static_assert(cuda::gpu_thread.static_dims(cuda::block, dims)[0] == 256);
      static_assert(cuda::block.static_dims(cuda::grid, dims)[0] == 4 * 1024);
    }
    SECTION("Mixed cluster dims")
    {
      auto dims_mixed = cuda::make_hierarchy(
        cuda::block_dims<256>(), cuda::cluster_dims(dim3(2, 2, 1)), cuda::grid_dims(dim3(1, 3, 9)));
      test_host_dev(dims_mixed, *this, arch_filter<std::less<int>, 90>);
      static_assert(cuda::gpu_thread.dims(cuda::block, dims_mixed).x == 256);
      static_assert(cuda::gpu_thread.count(cuda::block, dims_mixed) == 256);
      static_assert(cuda::gpu_thread.static_dims(cuda::block, dims_mixed)[0] == 256);
      static_assert(cuda::block.static_dims(cuda::cluster, dims_mixed)[0] == cuda::std::dynamic_extent);
      static_assert(cuda::block.static_dims(cuda::grid, dims_mixed)[0] == cuda::std::dynamic_extent);
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

  static_assert(config.hierarchy().count(cuda::gpu_thread, cuda::block) == block_size);
  static_assert(config.hierarchy().count(cuda::gpu_thread, cuda::cluster) == cluster_cnt *
  block_size); static_assert(config.hierarchy().count(cuda::block, cuda::cluster) ==
  cluster_cnt); CCCLRT_REQUIRE(config.hierarchy().count() == grid_size * cluster_cnt *
  block_size);

  static_assert(config.hierarchy().has_level(cuda::block));
  static_assert(config.hierarchy().has_level(cuda::cluster));
  static_assert(config.hierarchy().has_level(cuda::grid));
  static_assert(!config.hierarchy().has_level(cuda::thread));
  */
}

C2H_TEST("Replace level", "[hierarchy]")
{
// GCC 7 and 8 complains here that the hierarchy was not declared constexpr
#if !_CCCL_COMPILER(GCC, <, 9)
  const auto dimensions = cuda::make_hierarchy(cuda::block_dims<512>(), cuda::cluster_dims<8>(), cuda::grid_dims(256));
  const auto fragment   = dimensions.fragment(cuda::block, cuda::grid);
  static_assert(!fragment.has_level(cuda::block));
  static_assert(!cuda::__has_bottom_unit_or_level_v<cuda::thread_level, decltype(fragment)>);
  static_assert(fragment.has_level(cuda::cluster));
  static_assert(fragment.has_level(cuda::grid));
  static_assert(cuda::__has_bottom_unit_or_level_v<cuda::block_level, decltype(fragment)>);

  const auto replaced = cuda::hierarchy_add_level(fragment, cuda::block_dims(256));
  static_assert(replaced.has_level(cuda::block));
  static_assert(cuda::__has_bottom_unit_or_level_v<cuda::thread_level, decltype(replaced)>);
  CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::block, replaced) == 256);
#endif // !_CCCL_COMPILER(GCC, <, 9)
}

template <typename Hierarchy>
__global__ void kernel(Hierarchy hierarchy)
{
  auto grid  = cg::this_grid();
  auto block = cg::this_thread_block();

  CCCLRT_REQUIRE_DEVICE(grid.thread_rank() == cuda::gpu_thread.rank(cuda::grid));
  CCCLRT_REQUIRE_DEVICE(grid.block_rank() == cuda::block.rank(cuda::grid));
  CCCLRT_REQUIRE_DEVICE(grid.block_index() == cuda::block.index(cuda::grid));
  CCCLRT_REQUIRE_DEVICE(grid.num_threads() == cuda::gpu_thread.count(cuda::grid));
  CCCLRT_REQUIRE_DEVICE(grid.num_blocks() == cuda::block.count(cuda::grid));
  CCCLRT_REQUIRE_DEVICE(grid.dim_blocks() == cuda::block.dims(cuda::grid));

  CCCLRT_REQUIRE_DEVICE(block.thread_rank() == cuda::gpu_thread.rank(cuda::block));
  CCCLRT_REQUIRE_DEVICE(block.thread_index() == cuda::gpu_thread.index(cuda::block));
  CCCLRT_REQUIRE_DEVICE(block.num_threads() == cuda::gpu_thread.count(cuda::block));
  CCCLRT_REQUIRE_DEVICE(block.dim_threads() == cuda::gpu_thread.dims(cuda::block));

  CCCLRT_REQUIRE_DEVICE(block.thread_index() == cuda::gpu_thread.index(cuda::block, hierarchy));

  const auto grid_index = cuda::gpu_thread.index_as<unsigned long long>(cuda::grid, hierarchy);
  CCCLRT_REQUIRE_DEVICE(
    grid_index.x
    == static_cast<unsigned long long>(grid.block_index().x) * block.dim_threads().x + block.thread_index().x);
  CCCLRT_REQUIRE_DEVICE(
    grid_index.y
    == static_cast<unsigned long long>(grid.block_index().y) * block.dim_threads().y + block.thread_index().y);
  CCCLRT_REQUIRE_DEVICE(
    grid_index.z
    == static_cast<unsigned long long>(grid.block_index().z) * block.dim_threads().z + block.thread_index().z);

  CCCLRT_REQUIRE_DEVICE(grid.block_rank() == cuda::block.rank(cuda::grid, hierarchy));
  CCCLRT_REQUIRE_DEVICE(block.thread_rank() == cuda::gpu_thread.rank(cuda::block, hierarchy));
  CCCLRT_REQUIRE_DEVICE(grid.thread_rank() == cuda::gpu_thread.rank(cuda::grid, hierarchy));
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
  auto thread_id = cuda::gpu_thread.rank(cuda::block, hierarchy);
  out[thread_id] = thread_id;
}

template <typename Hierarchy>
__global__ void rank_kernel(Hierarchy hierarchy, unsigned int* out)
{
  auto thread_id = cuda::gpu_thread.rank(cuda::block);
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
    auto thread_index_in_block = cuda::gpu_thread.index(cuda::block, hierarchy);
    CCCLRT_REQUIRE_DEVICE(thread_index_in_block == threadIdx);
    auto block_index_in_grid = cuda::block.index(cuda::grid, hierarchy);
    CCCLRT_REQUIRE_DEVICE(block_index_in_grid == blockIdx);
  }
  {
    int thread_rank_in_block = cuda::gpu_thread.rank(cuda::block, hierarchy);
    int block_rank_in_grid   = cuda::block.rank(cuda::grid, hierarchy);
  }
  {
    // Can be called with the instances of level types
    int num_threads_in_block = cuda::gpu_thread.count(cuda::block);
    int num_blocks_in_grid   = cuda::block.count(cuda::grid);

    // Or using the level types as template arguments
    int num_threads_in_grid = cuda::gpu_thread.count(cuda::grid);
  }
  {
    // Can be called with the instances of level types
    int thread_rank_in_block = cuda::gpu_thread.rank(cuda::block);
    int block_rank_in_grid   = cuda::block.rank(cuda::grid);

    // Or using the level types as template arguments
    int thread_rank_in_grid = cuda::gpu_thread.rank(cuda::grid);
  }
  {
    // Can be called with the instances of level types
    CCCLRT_REQUIRE_DEVICE(cuda::gpu_thread.dims(cuda::block) == blockDim);
    CCCLRT_REQUIRE_DEVICE(cuda::block.dims(cuda::grid) == gridDim);

    // Or using the level types as template arguments
    auto grid_dims_in_threads = cuda::gpu_thread.dims(cuda::grid);
  }
  {
    // Can be called with the instances of level types
    CCCLRT_REQUIRE_DEVICE(cuda::gpu_thread.index(cuda::block) == threadIdx);
    CCCLRT_REQUIRE_DEVICE(cuda::block.index(cuda::grid) == blockIdx);

    // Or using the level types as template arguments
    auto thread_index_in_grid = cuda::gpu_thread.index(cuda::grid);
  }
}

// Test examples from the inline rst documentation
C2H_TEST("Examples", "[hierarchy]")
{
  // GCC 7 and 8 complains here that the hierarchy was not declared constexpr
#if !_CCCL_COMPILER(GCC, <, 9)
  {
    auto hierarchy = cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    auto fragment  = hierarchy.fragment(cuda::block, cuda::grid);
    auto new_hierarchy = cuda::hierarchy_add_level(fragment, cuda::block_dims<128>());
    static_assert(cuda::gpu_thread.count(cuda::block, new_hierarchy) == 128);
  }
  {
    auto hierarchy = cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(cuda::gpu_thread.count(cuda::cluster, hierarchy) == 4 * 8 * 8 * 8);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, hierarchy) == 256 * 4 * 8 * 8 * 8);
    CCCLRT_REQUIRE(cuda::cluster.count(cuda::grid, hierarchy) == 256);
  }
  {
    [[maybe_unused]] auto hierarchy =
      cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(cuda::gpu_thread.count(cuda::cluster, hierarchy) == 4 * 8 * 8 * 8);
  }
  {
    auto hierarchy = cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(cuda::gpu_thread.extents(cuda::cluster, hierarchy).extent(0) == 4 * 8);
    static_assert(cuda::gpu_thread.extents(cuda::cluster, hierarchy).extent(1) == 8);
    static_assert(cuda::gpu_thread.extents(cuda::cluster, hierarchy).extent(2) == 8);
    CCCLRT_REQUIRE(cuda::gpu_thread.extents(cuda::grid, hierarchy).extent(0) == 256 * 4 * 8);
    CCCLRT_REQUIRE(cuda::cluster.extents(cuda::grid, hierarchy).extent(0) == 256);
  }
#endif // !_CCCL_COMPILER(GCC, <, 9)
  {
    [[maybe_unused]] auto hierarchy =
      cuda::make_hierarchy(cuda::grid_dims(256), cuda::cluster_dims<4>(), cuda::block_dims<8, 8, 8>());
    static_assert(decltype(hierarchy.level(cuda::cluster).extents())::static_extent(0) == 4);
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
  unsigned numElements          = 50000;
  constexpr int threadsPerBlock = 256;
  auto config                   = cuda::distribute<threadsPerBlock>(numElements);

  CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::block, config) == 256);
  CCCLRT_REQUIRE(cuda::block.count(cuda::grid, config) == (numElements + threadsPerBlock - 1) / threadsPerBlock);
}

C2H_TEST("hierarchy merge", "[hierarchy]")
{
  SECTION("Non overlapping")
  {
    auto h1       = cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<2>());
    auto h2       = cuda::make_hierarchy<cuda::thread_level>(cuda::block_dims<3>());
    auto combined = h1.combine(h2);
    static_assert(cuda::gpu_thread.count(cuda::grid, combined) == 6);
    static_assert(cuda::gpu_thread.count(cuda::block, combined) == 3);
    static_assert(cuda::block.count(cuda::grid, combined) == 2);
    auto combined_the_other_way = h2.combine(h1);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(combined_the_other_way)>);
    static_assert(cuda::gpu_thread.count(cuda::grid, combined_the_other_way) == 6);

    auto dynamic_values   = cuda::make_hierarchy(cuda::cluster_dims(4), cuda::block_dims(5));
    auto combined_dynamic = dynamic_values.combine(h1);
    CCCLRT_REQUIRE(cuda::gpu_thread.count(cuda::grid, combined_dynamic) == 40);
  }
  SECTION("Overlapping")
  {
    auto h1       = cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<2>(), cuda::cluster_dims<3>());
    auto h2       = cuda::make_hierarchy<cuda::thread_level>(cuda::block_dims<4>(), cuda::cluster_dims<5>());
    auto combined = h1.combine(h2);
    static_assert(cuda::gpu_thread.count(cuda::grid, combined) == 24);
    static_assert(cuda::gpu_thread.count(cuda::block, combined) == 4);
    static_assert(cuda::block.count(cuda::grid, combined) == 6);

    auto combined_the_other_way = h2.combine(h1);
    static_assert(!cuda::std::is_same_v<decltype(combined), decltype(combined_the_other_way)>);
    static_assert(cuda::gpu_thread.count(cuda::grid, combined_the_other_way) == 40);
    static_assert(cuda::gpu_thread.count(cuda::block, combined_the_other_way) == 4);
    static_assert(cuda::block.count(cuda::grid, combined_the_other_way) == 10);

    auto ultimate_combination = combined.combine(combined_the_other_way);
    static_assert(cuda::std::is_same_v<decltype(combined), decltype(ultimate_combination)>);
    static_assert(cuda::gpu_thread.count(cuda::grid, ultimate_combination) == 24);

    auto block_level_replacement = cuda::make_hierarchy<cuda::thread_level>(cuda::block_dims<6>());
    auto with_block_replaced     = block_level_replacement.combine(combined);
    static_assert(cuda::gpu_thread.count(cuda::grid, with_block_replaced) == 36);
    static_assert(cuda::gpu_thread.count(cuda::block, with_block_replaced) == 6);

    auto grid_cluster_level_replacement =
      cuda::make_hierarchy<cuda::block_level>(cuda::grid_dims<7>(), cuda::cluster_dims<8>());
    auto with_grid_cluster_replaced = grid_cluster_level_replacement.combine(combined);
    static_assert(cuda::gpu_thread.count(cuda::grid, with_grid_cluster_replaced) == 7 * 8 * 4);
    static_assert(cuda::block.count(cuda::cluster, with_grid_cluster_replaced) == 8);
    static_assert(cuda::cluster.count(cuda::grid, with_grid_cluster_replaced) == 7);
  }
}
