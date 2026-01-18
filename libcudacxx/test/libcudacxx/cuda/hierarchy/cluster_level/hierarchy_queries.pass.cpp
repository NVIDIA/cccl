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
__device__ void test_cluster(const Hierarchy& hier, const GridExts& grid_exts, const ClusterExts&, const BlockExts&)
{
  uint3 dims = gridDim;
  NV_IF_TARGET(NV_PROVIDES_SM_90, (dims = __clusterGridDimInClusters();))

  uint3 index = blockIdx;
  NV_IF_TARGET(NV_PROVIDES_SM_90, (index = __clusterIdx();))

  // 1. Test cuda::cluster.dims(x, hier)
  test_dims(dims, cuda::cluster, cuda::grid, hier);

  // 2. Test cuda::cluster.static_dims(x, hier)
  {
    const ulonglong3 exp{
      GridExts::static_extent(0),
      GridExts::static_extent(1),
      GridExts::static_extent(2),
    };
    test_static_dims(exp, cuda::cluster, cuda::grid, hier);
  }

  // 3. Test cuda::cluster.extents(x)
  {
    const cuda::std::extents<unsigned, GridExts::static_extent(0), GridExts::static_extent(1), GridExts::static_extent(2)>
      exp{dims.x, dims.y, dims.z};
    test_extents(exp, cuda::cluster, cuda::grid, hier);
  }

  // 4. Test cuda::cluster.count(x, hier)
  test_count(cuda::std::size_t{dims.z} * dims.y * dims.x, cuda::cluster, cuda::grid, hier);

  // 5. test cuda::cluster.index(x, hier)
  test_index(index, cuda::cluster, cuda::grid, hier);

  // 6. Test cuda::cluster.rank(x, hier)
  {
    const cuda::std::size_t exp = (index.z * dims.y + index.y) * dims.x + index.x;
    test_rank(exp, cuda::cluster, cuda::grid, hier);
  }
}

__device__ void test_device()
{
  // todo: make hierarchy constructible on device
  // test_thread(cuda::make_hierarchy(cuda::grid_dims(gridDim), cuda::cluster_dims(clusterDim)));
}

#if !_CCCL_COMPILER(NVRTC)
template <class Hierarchy, class GridExts, class BlockExts>
__global__ void test_kernel(Hierarchy hier, GridExts grid_exts, BlockExts block_exts)
{
  test_cluster(hier, grid_exts, cuda::std::extents<unsigned, 1, 1, 1>{}, block_exts);
}

template <class Hierarchy, class GridExts, class ClusterExts, class BlockExts>
__global__ void test_kernel(Hierarchy hier, GridExts grid_exts, ClusterExts cluster_exts, BlockExts block_exts)
{
  test_cluster(hier, grid_exts, cluster_exts, block_exts);
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
