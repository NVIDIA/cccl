//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70
// UNSUPPORTED: nvrtc

// <cuda/atomic>

#define _LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH 1 // Force using the PTX is_local atomics path

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

/*
Test goals:
Pre-load registers with values that will be used to trigger the wrong codepath in local device atomics.

This test is architecture and driver dependent. It is not possible to reproduce this when compiled to SASS on 12.0, but
will repro on 12.8.

Compiled to SASS is an important point, compiling to PTX will show the failure to initialize the local test flag for
isspacep.local to 0, but that might be compiled out by the JIT compiler in the driver
*/
__global__ void __launch_bounds__(1024) device_test(char* gmem)
{
  constexpr int threads = 1024;

  __shared__ int hidx;
  __shared__ int histogram[threads];

  cuda::atomic<int, cuda::thread_scope_thread> xatom(0);

  constexpr int passes   = 16;
  constexpr int ops      = 32;
  constexpr int expected = passes * ops;

  if (threadIdx.x == 0)
  {
    hidx = 0;
    memset(histogram, sizeof(histogram), 0);
  }

  __syncthreads();

  for (xatom = 0; xatom.load() < passes; xatom++)
  {
    using A = cuda::atomic_ref<int, cuda::std::thread_scope_block>;
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 0]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 1]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 2]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 3]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 4]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 5]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 6]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 7]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 8]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 9]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 10]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 11]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 12]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 13]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 14]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 15]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 16]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 17]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 18]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 19]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 20]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 21]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 22]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 23]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 24]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 25]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 26]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 27]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 28]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 29]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 30]);
    A(histogram[A(hidx).fetch_add(1) % threads]).fetch_add(gmem[(xatom.load() * 8) + 31]);
  }

  __syncthreads();

  if (histogram[threadIdx.x] != expected)
  {
    printf("[%i] = %i\r\n", threadIdx.x, histogram[threadIdx.x]);
  }
  assert(histogram[threadIdx.x] == expected);
}

void launch_kernel()
{
  cudaError_t err;
  char* inptr = nullptr;
  CUDA_CALL(err, cudaGetLastError());
  CUDA_CALL(err, cudaMalloc(&inptr, 1024));
  CUDA_CALL(err, cudaMemset(inptr, 1, 1024));
  device_test<<<1, 1024>>>(inptr);
  CUDA_CALL(err, cudaGetLastError());
  CUDA_CALL(err, cudaDeviceSynchronize());
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (launch_kernel();))
  return 0;
}
