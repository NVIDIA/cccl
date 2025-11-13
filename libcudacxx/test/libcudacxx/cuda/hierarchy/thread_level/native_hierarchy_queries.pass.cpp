//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__hierarchy_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/mdspan>
#include <cuda/std/type_traits>

template <class T>
__device__ void test_result(cuda::hierarchy_query_result<T, 1> res, unsigned exp)
{
  assert(res.x == exp);
}

__device__ void test_result(cuda::hierarchy_query_result<unsigned, 3> res, uint3 exp)
{
  assert(res.x == exp.x);
  assert(res.y == exp.y);
  assert(res.z == exp.z);
}

__device__ void test_result(cuda::hierarchy_query_result<cuda::std::size_t, 1> res, cuda::std::size_t exp)
{
  assert(res.x == exp);
}

__device__ void test_result(cuda::hierarchy_query_result<cuda::std::size_t, 3> res, ulonglong3 exp)
{
  assert(res.x == exp.x);
  assert(res.y == exp.y);
  assert(res.z == exp.z);
}

template <class Index, cuda::std::size_t... Exts>
__device__ void test_result(cuda::std::extents<Index, Exts...> res, cuda::std::extents<Index, Exts...> exp)
{
  for (cuda::std::size_t i = 0; i < sizeof...(Exts); ++i)
  {
    assert(res.extent(i) == exp.extent(i));
  }
}

__device__ void test_thread()
{
  constexpr ulonglong3 dynamic_extent3 = {
    cuda::std::dynamic_extent, cuda::std::dynamic_extent, cuda::std::dynamic_extent};

  // 1. Test cuda::device::thread.dims(x)
  test_result(cuda::device::thread.dims(cuda::device::warp), warpSize);
  test_result(cuda::device::thread.dims(cuda::device::block), blockDim);
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterGridDimInClusters().x;
                   exp.y *= __clusterGridDimInClusters().y;
                   exp.z *= __clusterGridDimInClusters().z;
                 }))
    test_result(cuda::device::thread.dims(cuda::device::cluster), exp);
  }
  {
    const uint3 exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    test_result(cuda::device::thread.dims(cuda::device::grid), exp);
  }

  // 2. Test cuda::device::thread.static_dims(x)
  test_result(cuda::device::thread.static_dims(cuda::device::warp), cuda::std::size_t{32});
  test_result(cuda::device::thread.static_dims(cuda::device::block), dynamic_extent3);
  test_result(cuda::device::thread.static_dims(cuda::device::cluster), dynamic_extent3);
  test_result(cuda::device::thread.static_dims(cuda::device::grid), dynamic_extent3);

  // 3. Test cuda::device::thread.extents(x)
  test_result(cuda::device::thread.extents(cuda::device::warp), cuda::std::extents<unsigned, 32>{});
  test_result(cuda::device::thread.extents(cuda::device::block),
              cuda::std::dims<3, unsigned>{blockDim.z, blockDim.y, blockDim.x});
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterGridDimInClusters().x;
                   exp.y *= __clusterGridDimInClusters().y;
                   exp.z *= __clusterGridDimInClusters().z;
                 }))
    test_result(cuda::device::thread.extents(cuda::device::cluster), cuda::std::dims<3, unsigned>{exp.z, exp.y, exp.x});
  }
  {
    const uint3 exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    test_result(cuda::device::thread.extents(cuda::device::grid), cuda::std::dims<3, unsigned>{exp.z, exp.y, exp.x});
  }

  // 4. Test cuda::device::thread.count(x)
  assert(cuda::device::thread.count(cuda::device::warp) == 32);
  assert(cuda::device::thread.count(cuda::device::block) == cuda::std::size_t{blockDim.z} * blockDim.y * blockDim.x);
  {
    uint3 exp = blockDim;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x *= __clusterDim().x;
                   exp.y *= __clusterDim().y;
                   exp.z *= __clusterDim().z;
                 }))
    assert(cuda::device::thread.count(cuda::device::cluster) == cuda::std::size_t{exp.z} * exp.y * exp.x);
  }
  {
    const uint3 exp{blockDim.x * gridDim.x, blockDim.y * gridDim.y, blockDim.z * gridDim.z};
    assert(cuda::device::thread.count(cuda::device::grid) == cuda::std::size_t{exp.z} * exp.y * exp.x);
  }

  // 5. test cuda::device::thread.index(x)
  test_result(cuda::device::thread.index(cuda::device::warp), cuda::ptx::get_sreg_laneid());
  test_result(cuda::device::thread.index(cuda::device::block), threadIdx);
  {
    uint3 exp = threadIdx;
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   exp.x += blockDim.x * __clusterRelativeBlockIdx().x;
                   exp.y += blockDim.y * __clusterRelativeBlockIdx().y;
                   exp.z += blockDim.z * __clusterRelativeBlockIdx().z;
                 }))
    test_result(cuda::device::thread.index(cuda::device::cluster), exp);
  }
  {
    const uint3 exp{
      threadIdx.x + blockDim.x * blockIdx.x,
      threadIdx.y + blockDim.y * blockIdx.y,
      threadIdx.z + blockDim.z * blockIdx.z,
    };
    test_result(cuda::device::thread.index(cuda::device::grid), exp);
  }

  // 6. Test cuda::device::thread.rank(x)
  assert(cuda::device::thread.rank(cuda::device::warp) == cuda::ptx::get_sreg_laneid());
  assert(cuda::device::thread.rank(cuda::device::block)
         == ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x);
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
    assert(cuda::device::thread.rank(cuda::device::cluster) == exp);
  }
  {
    const cuda::std::size_t exp =
      ((((blockIdx.z * blockDim.z + threadIdx.z) * gridDim.y * blockDim.y) + blockIdx.y * blockDim.y + threadIdx.y)
       * gridDim.x * blockDim.x)
      + blockIdx.x * blockDim.x + threadIdx.x;
    assert(cuda::device::thread.rank(cuda::device::grid) == exp);
  }
}

#if !_CCCL_COMPILER(NVRTC)
__global__ void test_kernel()
{
  test_thread();
}

void test()
{
  test_kernel<<<1, 128>>>();
  test_kernel<<<128, 1>>>();
  test_kernel<<<{2, 3}, {4, 2}>>>();
  test_kernel<<<{2, 3, 4}, {4, 2, 8}>>>();

  // thread block clusters require compute capability at least 9.0
  int cc_major{};
  assert(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);
  if (cc_major < 9)
  {
    return;
  }

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
  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test();), (test_thread();))
  return 0;
}
