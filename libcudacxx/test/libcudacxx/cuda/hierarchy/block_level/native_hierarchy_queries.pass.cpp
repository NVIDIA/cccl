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

#include <cuda/hierarchy>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

#include "hierarchy_queries.h"

__device__ void test_block()
{
  constexpr cuda::std::size_t dext = cuda::std::dynamic_extent;

  // 1. Test cuda::block.dims(x)
  {
    uint3 exp{1, 1, 1};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (exp = __clusterDim();))
    test_dims(exp, cuda::block, cuda::cluster);
  }
  test_dims(gridDim, cuda::block, cuda::grid);

  // 2. Test cuda::block.static_dims(x)
  test_static_dims(ulonglong3{dext, dext, dext}, cuda::block, cuda::cluster);
  test_static_dims(ulonglong3{dext, dext, dext}, cuda::block, cuda::grid);

  // 3. Test cuda::block.extents(x)
  {
    uint3 exp{1, 1, 1};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (exp = __clusterDim();))
    test_extents(cuda::std::dims<3, unsigned>{exp.x, exp.y, exp.z}, cuda::block, cuda::cluster);
  }
  test_extents(cuda::std::dims<3, unsigned>{gridDim.x, gridDim.y, gridDim.z}, cuda::block, cuda::grid);

  // 4. Test cuda::block.count(x)
  {
    cuda::std::size_t exp = 1;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp *= __clusterDim().x;
                   exp *= __clusterDim().y;
                   exp *= __clusterDim().z;
                 }))
    test_count(exp, cuda::block, cuda::cluster);
  }
  test_count(cuda::std::size_t{gridDim.z} * gridDim.y * gridDim.x, cuda::block, cuda::grid);

  // 5. test cuda::block.index(x)
  {
    uint3 exp{0, 0, 0};
    NV_IF_TARGET(NV_PROVIDES_SM_90, (exp = __clusterRelativeBlockIdx();))
    test_index(exp, cuda::block, cuda::cluster);
  }
  test_index(blockIdx, cuda::block, cuda::grid);

  // 6. Test cuda::block.rank(x)
  {
    cuda::std::size_t exp = 0;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp = ((__clusterRelativeBlockIdx().z * __clusterDim().y) + __clusterRelativeBlockIdx().y)
                         * __clusterDim().x
                       + __clusterRelativeBlockIdx().x;
                 }))
    test_rank(exp, cuda::block, cuda::cluster);
  }
  {
    const cuda::std::size_t exp = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    test_rank(exp, cuda::block, cuda::grid);
  }
}

#if !_CCCL_COMPILER(NVRTC)
__global__ void test_kernel()
{
  test_block();
}

void test()
{
  int cc_major{};
  assert(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);

  // thread block clusters require compute capability at least 9.0
  const bool enable_clusters = cc_major >= 9;

  test_kernel<<<1, 128>>>();
  test_kernel<<<128, 1>>>();
  test_kernel<<<dim3{2, 3}, dim3{4, 2}>>>();
  test_kernel<<<dim3{2, 3, 4}, dim3{4, 2, 8}>>>();
  if (enable_clusters)
  {
    cudaLaunchAttribute attribute[1]{};
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 4;
    attribute[0].val.clusterDim.y = 2;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config{};
    config.gridDim  = {12, 10, 3};
    config.blockDim = {2, 8, 4};
    config.attrs    = attribute;
    config.numAttrs = 1;

    void* pargs[1]{};
    assert(cudaLaunchKernelExC(&config, (const void*) test_kernel, pargs) == cudaSuccess);
  }

  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test();), (test_block();))
  return 0;
}
