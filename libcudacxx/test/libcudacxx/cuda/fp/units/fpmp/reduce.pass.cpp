// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// error: accessing gridDim/blockDim/blockIdx/threadIdx/warpSize is unsupported in tile code
// UNSUPPORTED: nvrtc
// note: the host half of this test launches the kernel through the CUDA runtime API,
// which is not available in NVRTC's device-only translation unit

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 compatibility with cooperative_groups::reduce.
//
//  Device-only test. Each thread contributes (seed + thread_id) and a tiled
//  sub-warp reduces the values with operator+; the reduced result of each
//  sub-warp is asserted on-device against the closed-form arithmetic-series sum.
//  A host-only build compiles the device work out.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_CUDA_COMPILATION()
#  include <cooperative_groups.h>

#  include <cooperative_groups/reduce.h>

template <int subwarp_size, typename mp_type>
__global__ void test_reduce_kernel(unsigned int seed)
{
  namespace cg            = cooperative_groups;
  auto this_block         = cg::this_thread_block();
  auto subwarp            = cg::tiled_partition<subwarp_size>(this_block);
  const auto thread_id    = this_block.thread_rank();
  const mp_type to_reduce = seed + thread_id;

  mp_type result = cg::reduce(subwarp, to_reduce, [](const mp_type& a, const mp_type& b) -> mp_type {
    return a + b;
  });

  if (subwarp.thread_rank() == 0)
  {
    const int i            = static_cast<int>(thread_id) / subwarp_size;
    const double to_double = static_cast<double>(result);
    const ::cuda::std::int64_t expected =
      // sum of the lowest subwarp ...
      (2 * ::cuda::std::int64_t{seed} + subwarp_size - 1) * subwarp_size / 2
      // ... plus the offset for higher subwarps.
      + i * subwarp_size * subwarp_size;
    assert(::cuda::std::fabs(to_double - static_cast<double>(expected)) <= 1e-4);
  }
}

// The launches must live outside the NV_IF_TARGET(NV_IS_HOST) block in main(): nvcc's device
// pass discards that block, so the kernel template would never be instantiated for the device
// and every launch would fail with cudaErrorInvalidDeviceFunction.
void run_reduce()
{
  const unsigned int seed = 10;
  test_reduce_kernel<4, cudax::fp32mp2><<<1, 4>>>(seed);
  test_reduce_kernel<32, cudax::fp32mp2><<<1, 64>>>(seed);
  test_reduce_kernel<4, cudax::fp64mp2><<<1, 4>>>(seed);
  test_reduce_kernel<32, cudax::fp64mp2><<<1, 64>>>(seed);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // _CCCL_CUDA_COMPILATION()

int main(int, char**)
{
#if _CCCL_CUDA_COMPILATION()
  // force_include.h makes this main __host__ __device__ and runs it twice: on the host,
  // then inside a kernel. Only the host run can launch, so NV_IS_HOST selects the driver
  // of the test, not the code under test -- the reductions themselves run on the GPU.
  NV_IF_TARGET(NV_IS_HOST, (run_reduce();))
#endif // _CCCL_CUDA_COMPILATION()
  return 0;
}
