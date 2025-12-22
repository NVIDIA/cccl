//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo: enable with nvrtc
// UNSUPPORTED: nvrtc

#include "hierarchy_queries.h"

#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

template <class Hierarchy, class GridExts, class ClusterExts, class BlockExts>
__device__ void test_thread(
  const Hierarchy& hier, const GridExts& grid_exts, const ClusterExts& cluster_exts, const BlockExts& block_exts)
{
  // 1. Test cuda::gpu_thread.dims(x, hier)
  test_dims(blockDim, cuda::gpu_thread, cuda::block, hier);
  if constexpr (Hierarchy::has_level(cuda::cluster))
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterDim().x;
                   exp.y *= __clusterDim().y;
                   exp.z *= __clusterDim().z;
                 }))
    test_dims(exp, cuda::gpu_thread, cuda::cluster, hier);
  }
  {
    const uint3 exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    test_dims(exp, cuda::gpu_thread, cuda::grid, hier);
  }

  // 2. Test cuda::gpu_thread.static_dims(x, hier)
  test_static_dims(ulonglong3{BlockExts::static_extent(0), BlockExts::static_extent(1), BlockExts::static_extent(2)},
                   cuda::gpu_thread,
                   cuda::block,
                   hier);
  if constexpr (Hierarchy::has_level(cuda::cluster))
  {
    const ulonglong3 exp{
      mul_static_extents(ClusterExts::static_extent(0), BlockExts::static_extent(0)),
      mul_static_extents(ClusterExts::static_extent(1), BlockExts::static_extent(1)),
      mul_static_extents(ClusterExts::static_extent(2), BlockExts::static_extent(2)),
    };
    test_static_dims(exp, cuda::gpu_thread, cuda::cluster, hier);
  }
  {
    const ulonglong3 exp{
      mul_static_extents(GridExts::static_extent(0), ClusterExts::static_extent(0), BlockExts::static_extent(0)),
      mul_static_extents(GridExts::static_extent(1), ClusterExts::static_extent(1), BlockExts::static_extent(1)),
      mul_static_extents(GridExts::static_extent(2), ClusterExts::static_extent(2), BlockExts::static_extent(2)),
    };
    test_static_dims(exp, cuda::gpu_thread, cuda::grid, hier);
  }

  // 3. Test cuda::gpu_thread.extents(x)
  test_extents(block_exts, cuda::gpu_thread, cuda::block, hier);
  if constexpr (Hierarchy::has_level(cuda::cluster))
  {
    uint3 dims = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   dims.x *= __clusterDim().x;
                   dims.y *= __clusterDim().y;
                   dims.z *= __clusterDim().z;
                 }))

    const cuda::std::extents<unsigned,
                             mul_static_extents(ClusterExts::static_extent(0), BlockExts::static_extent(0)),
                             mul_static_extents(ClusterExts::static_extent(1), BlockExts::static_extent(1)),
                             mul_static_extents(ClusterExts::static_extent(2), BlockExts::static_extent(2))>
      exp{dims.x, dims.y, dims.z};

    test_extents(exp, cuda::gpu_thread, cuda::cluster, hier);
  }
  {
    const cuda::std::extents<
      unsigned,
      mul_static_extents(GridExts::static_extent(0), ClusterExts::static_extent(0), BlockExts::static_extent(0)),
      mul_static_extents(GridExts::static_extent(1), ClusterExts::static_extent(1), BlockExts::static_extent(1)),
      mul_static_extents(GridExts::static_extent(2), ClusterExts::static_extent(2), BlockExts::static_extent(2))>
      exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    test_extents(exp, cuda::gpu_thread, cuda::grid, hier);
  }

  // 4. Test cuda::gpu_thread.count(x, hier)
  test_count(cuda::std::size_t{blockDim.z} * blockDim.y * blockDim.x, cuda::gpu_thread, cuda::block, hier);
  if constexpr (Hierarchy::has_level(cuda::cluster))
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterDim().x;
                   exp.y *= __clusterDim().y;
                   exp.z *= __clusterDim().z;
                 }))
    test_count(cuda::std::size_t{exp.z} * exp.y * exp.x, cuda::gpu_thread, cuda::cluster, hier);
  }
  {
    const uint3 exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    test_count(cuda::std::size_t{exp.z} * exp.y * exp.x, cuda::gpu_thread, cuda::grid, hier);
  }

  // 5. test cuda::gpu_thread.index(x, hier)
  test_index(threadIdx, cuda::gpu_thread, cuda::block, hier);
  if constexpr (Hierarchy::has_level(cuda::cluster))
  {
    uint3 exp = threadIdx;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x += blockDim.x * __clusterRelativeBlockIdx().x;
                   exp.y += blockDim.y * __clusterRelativeBlockIdx().y;
                   exp.z += blockDim.z * __clusterRelativeBlockIdx().z;
                 }))
    test_index(exp, cuda::gpu_thread, cuda::cluster, hier);
  }
  {
    const uint3 exp{
      threadIdx.x + blockDim.x * blockIdx.x,
      threadIdx.y + blockDim.y * blockIdx.y,
      threadIdx.z + blockDim.z * blockIdx.z,
    };
    test_index(exp, cuda::gpu_thread, cuda::grid, hier);
  }

  // 6. Test cuda::gpu_thread.rank(x, hier)
  test_rank((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x, cuda::gpu_thread, cuda::block, hier);
  if constexpr (Hierarchy::has_level(cuda::cluster))
  {
    cuda::std::size_t exp = 0;
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      ({
                        exp = (((__clusterRelativeBlockIdx().z * __clusterDim().y * __clusterDim().x)
                                + __clusterRelativeBlockIdx().y * __clusterDim().x)
                               + __clusterRelativeBlockIdx().x)
                              * (blockDim.x * blockDim.y * blockDim.z)
                            + ((threadIdx.z * blockDim.y * blockDim.x) + threadIdx.y * blockDim.x) + threadIdx.x;
                      }),
                      ({ exp = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x; }))
    test_rank(exp, cuda::gpu_thread, cuda::cluster, hier);
  }
  {
    const cuda::std::size_t exp =
      (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x)
        * (blockDim.x * blockDim.y * blockDim.z)
      + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    test_rank(exp, cuda::gpu_thread, cuda::grid, hier);
  }
}

__device__ void test_device()
{
  // todo: make hierarchy constructible on device
  // test_thread(cuda::make_hierarchy(cuda::grid_dims(gridDim), cuda::block_dims(blockDim)));
}

#if !_CCCL_COMPILER(NVRTC)
template <class Hierarchy, class GridExts, class BlockExts>
__global__ void test_kernel(Hierarchy hier, GridExts grid_exts, BlockExts block_exts)
{
  test_thread(hier, grid_exts, cuda::std::extents<unsigned, 1, 1, 1>{}, block_exts);
}

template <class Hierarchy, class GridExts, class ClusterExts, class BlockExts>
__global__ void test_kernel(Hierarchy hier, GridExts grid_exts, ClusterExts cluster_exts, BlockExts block_exts)
{
  test_thread(hier, grid_exts, cluster_exts, block_exts);
}

template <class GridExts, class BlockExts>
void test_launch(GridExts grid_exts, BlockExts block_exts)
{
  const dim3 grid_dims{grid_exts.extent(0), grid_exts.extent(1), grid_exts.extent(2)};
  const dim3 block_dims{block_exts.extent(0), block_exts.extent(1), block_exts.extent(2)};

  const cuda::std::dims<3, unsigned> grid_exts_dyn{grid_exts.extent(0), grid_exts.extent(1), grid_exts.extent(2)};
  const cuda::std::dims<3, unsigned> block_exts_dyn{block_exts.extent(0), block_exts.extent(1), block_exts.extent(2)};

  // 1. Launch hierarchy with all static extents.
  test_kernel<<<grid_dims, block_dims>>>(
    cuda::make_hierarchy(
      cuda::grid_dims<GridExts::static_extent(0), GridExts::static_extent(1), GridExts::static_extent(2)>(),
      cuda::block_dims<BlockExts::static_extent(0), BlockExts::static_extent(1), BlockExts::static_extent(2)>()),
    grid_exts,
    block_exts);

  // 2. Launch hierarchy with static grid extents and dynamic block extents.
  test_kernel<<<grid_dims, block_dims>>>(
    cuda::make_hierarchy(
      cuda::grid_dims<GridExts::static_extent(0), GridExts::static_extent(1), GridExts::static_extent(2)>(),
      cuda::block_dims(block_dims)),
    grid_exts,
    block_exts_dyn);

  // 3. Launch hierarchy with dynamic grid extents and static block extents.
  test_kernel<<<grid_dims, block_dims>>>(
    cuda::make_hierarchy(
      cuda::grid_dims(grid_dims),
      cuda::block_dims<BlockExts::static_extent(0), BlockExts::static_extent(1), BlockExts::static_extent(2)>()),
    grid_exts_dyn,
    block_exts);

  // 4. Launch hierarchy with dynamic grid extents and dynamic block extents.
  test_kernel<<<grid_dims, block_dims>>>(
    cuda::make_hierarchy(cuda::grid_dims(grid_dims), cuda::block_dims(block_dims)), grid_exts_dyn, block_exts_dyn);
}

template <class GridExts, class ClusterExts, class BlockExts>
void test_launch(GridExts grid_exts, ClusterExts cluster_exts, BlockExts block_exts)
{
  const dim3 grid_dims{grid_exts.extent(0), grid_exts.extent(1), grid_exts.extent(2)};
  const dim3 cluster_dims{cluster_exts.extent(0), cluster_exts.extent(1), cluster_exts.extent(2)};
  const dim3 block_dims{block_exts.extent(0), block_exts.extent(1), block_exts.extent(2)};

  cuda::std::dims<3, unsigned> grid_exts_dyn{grid_exts.extent(0), grid_exts.extent(1), grid_exts.extent(2)};
  cuda::std::dims<3, unsigned> cluster_exts_dyn{cluster_exts.extent(0), cluster_exts.extent(1), cluster_exts.extent(2)};
  cuda::std::dims<3, unsigned> block_exts_dyn{block_exts.extent(0), block_exts.extent(1), block_exts.extent(2)};

  cudaLaunchAttribute attribute[1]{};
  attribute[0].id               = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster_dims.x;
  attribute[0].val.clusterDim.y = cluster_dims.y;
  attribute[0].val.clusterDim.z = cluster_dims.z;

  cudaLaunchConfig_t config{};
  config.gridDim  = dim3{grid_dims.x * cluster_dims.x, grid_dims.y * cluster_dims.y, grid_dims.z * cluster_dims.z};
  config.blockDim = block_dims;
  config.attrs    = attribute;
  config.numAttrs = 1;

  // 1. Launch hierarchy with all static extents.
  {
    auto hier = cuda::make_hierarchy(
      cuda::grid_dims<GridExts::static_extent(0), GridExts::static_extent(1), GridExts::static_extent(2)>(),
      cuda::cluster_dims<ClusterExts::static_extent(0), ClusterExts::static_extent(1), ClusterExts::static_extent(2)>(),
      cuda::block_dims<BlockExts::static_extent(0), BlockExts::static_extent(1), BlockExts::static_extent(2)>());
    auto kernel = test_kernel<decltype(hier), GridExts, ClusterExts, BlockExts>;
    assert(cudaLaunchKernelEx(&config, kernel, hier, grid_exts, cluster_exts, block_exts) == cudaSuccess);
  }

  // 2. Launch hierarchy with all dynamic extents.
  {
    auto hier =
      cuda::make_hierarchy(cuda::grid_dims(grid_dims), cuda::cluster_dims(cluster_dims), cuda::block_dims(block_dims));
    auto kernel =
      test_kernel<decltype(hier), decltype(grid_exts_dyn), decltype(cluster_exts_dyn), decltype(block_exts_dyn)>;
    assert(cudaLaunchKernelEx(&config, kernel, hier, grid_exts_dyn, cluster_exts_dyn, block_exts_dyn) == cudaSuccess);
  }
}

void test()
{
  int cc_major{};
  assert(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);

  // thread block clusters require compute capability at least 9.0
  const bool enable_clusters = cc_major >= 9;

  test_launch(cuda::std::extents<unsigned, 1, 1, 1>{}, cuda::std::extents<unsigned, 128, 1, 1>{});
  test_launch(cuda::std::extents<unsigned, 128, 1, 1>{}, cuda::std::extents<unsigned, 1, 1, 1>{});
  test_launch(cuda::std::extents<unsigned, 2, 3, 1>{}, cuda::std::extents<unsigned, 4, 2, 1>{});
  test_launch(cuda::std::extents<unsigned, 2, 3, 4>{}, cuda::std::extents<unsigned, 4, 2, 8>{});

  if (enable_clusters)
  {
    test_launch(cuda::std::extents<unsigned, 3, 5, 3>{},
                cuda::std::extents<unsigned, 4, 2, 1>{},
                cuda::std::extents<unsigned, 2, 8, 4>{});
  }

  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test();), (test_device();))
  return 0;
}
