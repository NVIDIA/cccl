// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// UNSUPPORTED: enable-tile
// error: atomic operations are unsupported in tile code
// UNSUPPORTED: nvrtc
// note: the host half of this test launches the kernels through the CUDA runtime API,
// which is not available in NVRTC's device-only translation unit
// UNSUPPORTED: pre-sm-90
// note: the 128-bit compare-exchange (needed by the double-double atomics) requires
// compute capability >= 9.0 (Hopper) and PTX ISA 8.4+; the architecture half is checked
// here, the PTX ISA half below, since lit does not model the PTX ISA version

//===----------------------------------------------------------------------===//
//
//  Unit test: atomicAdd / atomicSub on fp64mp2 (double-double).
//
//  Device-only test (multi-block 128-bit atomics; requires compute capability
//  >= 9.0 / Hopper and PTX ISA 8.4+ for the 128-bit compare-exchange). Two checks:
//    - Atomicity: every thread does atomicAdd(1.0) then atomicSub(1.0); with
//      correct atomics the shared accumulator cancels back to ~0.
//    - Accuracy: many threads accumulate/subtract a small value and the result is
//      compared against the analytic sum within a relative tolerance.
//
//  The grid-wide accumulation is verified on the host after the kernels finish;
//  a host-only build (no CUDA) compiles the device work out.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a __host__ __device__ function in tile is not allowed

#include <cuda/fpmp>
#include <cuda/std/cassert>
#include <cuda/std/cmath>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Type alias for the double-double multi-precision floating-point type.
using dfloat = cudax::fp64mp2;

// Skip the test if the PTX ISA is insufficient for the 128-bit compare-exchange
#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 840
// Each thread adds then subtracts 1.0; the accumulator must cancel to ~0.
__global__ void test_atomicity_kernel_dd(unsigned int* idx, dfloat* res)
{
  dfloat val(1.0);
  atomicAdd(idx, 1u);
  atomicAdd(res, val);
  atomicSub(res, val);
}

__global__ void test_atomicAdd_accuracy_kernel_dd(dfloat* res, double value_to_add)
{
  dfloat val(value_to_add);
  atomicAdd(res, val);
}

__global__ void test_atomicSub_accuracy_kernel_dd(dfloat* res, double value_to_sub)
{
  dfloat val(value_to_sub);
  atomicSub(res, val);
}

void run_atomicity()
{
  const int num_threads = 512;
  const int num_blocks  = 4;

  unsigned int* d_idx = nullptr;
  dfloat* d_res       = nullptr;
  assert(cudaMalloc(&d_idx, sizeof(unsigned int)) == cudaSuccess);
  assert(cudaMalloc(&d_res, sizeof(dfloat)) == cudaSuccess);

  unsigned int h_idx = 0;
  dfloat h_res(0.0);
  assert(cudaMemcpy(d_idx, &h_idx, sizeof(unsigned int), cudaMemcpyHostToDevice) == cudaSuccess);
  assert(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice) == cudaSuccess);

  test_atomicity_kernel_dd<<<num_blocks, num_threads>>>(d_idx, d_res);
  assert(cudaGetLastError() == cudaSuccess);
  assert(cudaDeviceSynchronize() == cudaSuccess);

  assert(cudaMemcpy(&h_idx, d_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost) == cudaSuccess);
  assert(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost) == cudaSuccess);

  const double result = static_cast<double>(h_res);

  assert(h_idx == static_cast<unsigned int>(num_threads * num_blocks));
  assert(::cuda::std::fabs(result) < 1e-14);

  assert(cudaFree(d_idx) == cudaSuccess);
  assert(cudaFree(d_res) == cudaSuccess);
}

void run_accuracy()
{
  const int num_threads   = 512;
  const int num_blocks    = 4;
  const int total_threads = num_threads * num_blocks;

  // Test 1: add a small value from all threads.
  {
    dfloat* d_res = nullptr;
    assert(cudaMalloc(&d_res, sizeof(dfloat)) == cudaSuccess);
    dfloat h_res(0.0);
    assert(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice) == cudaSuccess);

    const double value_to_add = 0.1;
    test_atomicAdd_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value_to_add);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost) == cudaSuccess);

    const double result    = static_cast<double>(h_res);
    const double expected  = value_to_add * total_threads;
    const double rel_error = ::cuda::std::fabs(result - expected) / expected;
    assert(rel_error <= 1e-14);

    assert(cudaFree(d_res) == cudaSuccess);
  }

  // Test 2: add then subtract the same value (start at 100.0).
  {
    dfloat* d_res = nullptr;
    assert(cudaMalloc(&d_res, sizeof(dfloat)) == cudaSuccess);
    dfloat h_res(100.0);
    assert(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice) == cudaSuccess);

    const double value = 0.5;
    test_atomicAdd_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    test_atomicSub_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost) == cudaSuccess);

    const double result    = static_cast<double>(h_res);
    const double expected  = 100.0;
    const double rel_error = ::cuda::std::fabs(result - expected) / expected;
    assert(rel_error <= 1e-14);

    assert(cudaFree(d_res) == cudaSuccess);
  }

  // Test 3: subtract a value from all threads (start at 1000.0).
  {
    dfloat* d_res = nullptr;
    assert(cudaMalloc(&d_res, sizeof(dfloat)) == cudaSuccess);
    dfloat h_res(1000.0);
    assert(cudaMemcpy(d_res, &h_res, sizeof(dfloat), cudaMemcpyHostToDevice) == cudaSuccess);

    const double value_to_sub = 0.25;
    test_atomicSub_accuracy_kernel_dd<<<num_blocks, num_threads>>>(d_res, value_to_sub);
    assert(cudaGetLastError() == cudaSuccess);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    assert(cudaMemcpy(&h_res, d_res, sizeof(dfloat), cudaMemcpyDeviceToHost) == cudaSuccess);

    const double result    = static_cast<double>(h_res);
    const double expected  = 1000.0 - (value_to_sub * total_threads);
    const double rel_error = ::cuda::std::fabs(result - expected) / ::cuda::std::fabs(expected);
    assert(rel_error <= 1e-14);

    assert(cudaFree(d_res) == cudaSuccess);
  }
}
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 840

int main(int, char**)
{
#if _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 840
  // force_include.h makes this main __host__ __device__ and runs it twice: on the host,
  // then inside a kernel. Only the host run can launch kernels and call the runtime API,
  // so NV_IS_HOST selects the driver of the test, not the code under test -- the atomics
  // themselves run on the GPU.
  NV_IF_TARGET(NV_IS_HOST, (run_atomicity(); run_accuracy();))
#endif // _CCCL_CUDA_COMPILATION() && __cccl_ptx_isa >= 840
  return 0;
}
