// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// error: asm statement is unsupported in tile code

//===----------------------------------------------------------------------===//
//
//  Unit test: fpmp2 warp-shuffle overloads.
//
//  Device-only test. For __shfl_sync / __shfl_xor_sync / __shfl_down_sync /
//  __shfl_up_sync, the fpmp2 overload output is compared against the scalar CUDA
//  intrinsics applied independently to the hi/lo lanes. The kernel runs on a
//  single warp (32 lanes) and asserts equality on-device; a host-only build
//  compiles it out.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_CUDA_COMPILATION()
template <typename MP2>
TEST_DEVICE_FUNC MP2 make_lane_value(int lane)
{
  using FpType = decltype(MP2().hi());
  FpType hi    = static_cast<FpType>(lane) + static_cast<FpType>(0.25);
  FpType lo    = static_cast<FpType>(lane - 8) * static_cast<FpType>(0.03125);
  return MP2(hi, lo);
}

template <typename MP2>
TEST_DEVICE_FUNC void test_shfl()
{
  const int lane      = threadIdx.x & 31;
  const unsigned mask = 0xFFFFFFFFu;
  MP2 x               = make_lane_value<MP2>(lane);

  {
    MP2 y       = __shfl_sync(mask, x, 3, 16);
    auto ref_hi = ::__shfl_sync(mask, x.hi(), 3, 16);
    auto ref_lo = ::__shfl_sync(mask, x.lo(), 3, 16);
    assert(y.hi() == ref_hi && y.lo() == ref_lo);
  }
  {
    MP2 y       = __shfl_xor_sync(mask, x, 5, 16);
    auto ref_hi = ::__shfl_xor_sync(mask, x.hi(), 5, 16);
    auto ref_lo = ::__shfl_xor_sync(mask, x.lo(), 5, 16);
    assert(y.hi() == ref_hi && y.lo() == ref_lo);
  }
  {
    MP2 y       = __shfl_down_sync(mask, x, 2u, 16);
    auto ref_hi = ::__shfl_down_sync(mask, x.hi(), 2u, 16);
    auto ref_lo = ::__shfl_down_sync(mask, x.lo(), 2u, 16);
    assert(y.hi() == ref_hi && y.lo() == ref_lo);
  }
  {
    MP2 y       = __shfl_up_sync(mask, x, 2u, 16);
    auto ref_hi = ::__shfl_up_sync(mask, x.hi(), 2u, 16);
    auto ref_lo = ::__shfl_up_sync(mask, x.lo(), 2u, 16);
    assert(y.hi() == ref_hi && y.lo() == ref_lo);
  }
}

__global__ void test_kernel()
{
  test_shfl<cudax::fp32mp2>();
  test_shfl<cudax::fp64mp2>();
}
#endif // _CCCL_CUDA_COMPILATION()

int main(int, char**)
{
#if _CCCL_CUDA_COMPILATION()
  // force_include.h makes this main __host__ __device__ and runs it twice: on the host,
  // then inside a kernel. Only the host run can launch, so NV_IS_HOST selects the driver
  // of the test, not the code under test -- the shuffles themselves run on the GPU.
  //
  // test_kernel is not a template, so the launch can stay inside NV_IF_TARGET: it is
  // instantiated for the device regardless of the host-only block being discarded.
  NV_IF_TARGET(NV_IS_HOST,
               (test_kernel<<<1, 32>>>(); assert(cudaGetLastError() == cudaSuccess);
                assert(cudaDeviceSynchronize() == cudaSuccess);))
#endif // _CCCL_CUDA_COMPILATION()
  return 0;
}
