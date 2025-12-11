//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "hierarchy_queries.h"

__device__ void test_thread()
{
  constexpr cuda::std::size_t dext = cuda::std::dynamic_extent;

  // 1. Test cuda::gpu_thread.dims(x)
  test_dims(uint3{static_cast<unsigned>(warpSize), 1u, 1u}, cuda::gpu_thread, cuda::warp);
  test_dims(blockDim, cuda::gpu_thread, cuda::block);
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterGridDimInClusters().x;
                   exp.y *= __clusterGridDimInClusters().y;
                   exp.z *= __clusterGridDimInClusters().z;
                 }))
    test_dims(exp, cuda::gpu_thread, cuda::cluster);
  }
  test_dims({blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z}, cuda::gpu_thread, cuda::grid);

  // 2. Test cuda::gpu_thread.static_dims(x)
  test_static_dims(ulonglong3{cuda::std::size_t{32}, 1, 1}, cuda::gpu_thread, cuda::warp);
  test_static_dims(ulonglong3{dext, dext, dext}, cuda::gpu_thread, cuda::block);
  test_static_dims(ulonglong3{dext, dext, dext}, cuda::gpu_thread, cuda::cluster);
  test_static_dims(ulonglong3{dext, dext, dext}, cuda::gpu_thread, cuda::grid);

  // 3. Test cuda::gpu_thread.extents(x)
  test_extents(cuda::std::extents<unsigned, 32>{}, cuda::gpu_thread, cuda::warp);
  test_extents(cuda::std::dims<3, unsigned>{blockDim.x, blockDim.y, blockDim.z}, cuda::gpu_thread, cuda::block);
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterGridDimInClusters().x;
                   exp.y *= __clusterGridDimInClusters().y;
                   exp.z *= __clusterGridDimInClusters().z;
                 }))
    test_extents(cuda::std::dims<3, unsigned>{exp.x, exp.y, exp.z}, cuda::gpu_thread, cuda::cluster);
  }
  {
    const uint3 exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    test_extents(cuda::std::dims<3, unsigned>{exp.x, exp.y, exp.z}, cuda::gpu_thread, cuda::grid);
  }

  // 4. Test cuda::thread.count(x)
  test_count(32, cuda::gpu_thread, cuda::warp);
  test_count(cuda::std::size_t{blockDim.z} * blockDim.y * blockDim.x, cuda::gpu_thread, cuda::block);
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterDim().x;
                   exp.y *= __clusterDim().y;
                   exp.z *= __clusterDim().z;
                 }))
    test_count(cuda::std::size_t{exp.z} * exp.y * exp.x, cuda::gpu_thread, cuda::cluster);
  }
  {
    const uint3 exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    test_count(cuda::std::size_t{exp.z} * exp.y * exp.x, cuda::gpu_thread, cuda::grid);
  }

  // 5. test cuda::gpu_thread.index(x)
  test_index(uint3{cuda::ptx::get_sreg_laneid(), 0, 0}, cuda::gpu_thread, cuda::warp);
  test_index(threadIdx, cuda::gpu_thread, cuda::block);
  {
    uint3 exp = threadIdx;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x += blockDim.x * __clusterRelativeBlockIdx().x;
                   exp.y += blockDim.y * __clusterRelativeBlockIdx().y;
                   exp.z += blockDim.z * __clusterRelativeBlockIdx().z;
                 }))
    test_index(exp, cuda::gpu_thread, cuda::cluster);
  }
  {
    const uint3 exp{
      threadIdx.x + blockDim.x * blockIdx.x,
      threadIdx.y + blockDim.y * blockIdx.y,
      threadIdx.z + blockDim.z * blockIdx.z,
    };
    test_index(exp, cuda::gpu_thread, cuda::grid);
  }

  // 6. Test cuda::thread.rank(x)
  test_rank(cuda::ptx::get_sreg_laneid(), cuda::gpu_thread, cuda::warp);
  test_rank(((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x, cuda::gpu_thread, cuda::block);
  {
    cuda::std::size_t exp = 0;
    NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                      ({
                        exp = ((((__clusterRelativeBlockIdx().z * blockDim.z + threadIdx.z) * gridDim.y * blockDim.y)
                                + __clusterRelativeBlockIdx().y * blockDim.y + threadIdx.y)
                               * gridDim.x * blockDim.x)
                            + __clusterRelativeBlockIdx().x * blockDim.x + threadIdx.x;
                      }),
                      ({ exp = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x; }))
    test_rank(exp, cuda::gpu_thread, cuda::cluster);
  }
  {
    const cuda::std::size_t exp =
      ((((blockIdx.z * blockDim.z + threadIdx.z) * gridDim.y * blockDim.y) + blockIdx.y * blockDim.y + threadIdx.y)
       * gridDim.x * blockDim.x)
      + blockIdx.x * blockDim.x + threadIdx.x;
    test_rank(exp, cuda::gpu_thread, cuda::grid);
  }
}

#if !_CCCL_COMPILER(NVRTC)
__global__ void test_kernel()
{
  test_thread();
}

void test()
{
  int cc_major{};
  assert(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);

  // thread block clusters require compute capability at least 9.0
  const bool enable_clusters = cc_major >= 9;

  test_kernel<<<1, 128>>>();
  test_kernel<<<128, 1>>>();
  test_kernel<<<{2, 3}, {4, 2}>>>();
  test_kernel<<<{2, 3, 4}, {4, 2, 8}>>>();
  if (enable_clusters)
  {
    cudaLaunchAttribute attribute[1]{};
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 8;
    attribute[0].val.clusterDim.z = 4;

    cudaLaunchConfig_t config{};
    config.gridDim  = {4, 16, 8};
    config.blockDim = {2, 8, 4};
    config.attrs    = attribute;
    config.numAttrs = 1;

    assert(cudaLaunchKernelEx(&config, test_kernel) == cudaSuccess);
  }

  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test();), (test_thread();))
  return 0;
}
