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

#define _LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH 1

#include <cuda/atomic>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

/*
Test goals:
Pre-load registers with values that will be used to trigger the wrong codepath in local device atomics.
*/
__global__ void __launch_bounds__(2048) device_test(char* gmem)
{
  __shared__ int hidx;
  __shared__ int histogram[1024];

  cuda::atomic<int, cuda::thread_scope_thread> xatom(0);

  if (threadIdx.x == 0)
  {
    hidx = 0;
    memset(histogram, sizeof(histogram), 0);
  }

  __syncthreads();

  for (xatom = 0; xatom.load() < 16; xatom++)
  {
    using A = cuda::atomic_ref<int, cuda::std::thread_scope_block>;
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 0]);
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 1]);
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 2]);
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 3]);
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 4]);
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 5]);
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 6]);
    A(histogram[A(hidx).fetch_add(1) % 1024]).fetch_add(gmem[(xatom.load() * 8) + 7]);
  }

  __syncthreads();
  printf("[%i] = %i\r\n", threadIdx.x, histogram[threadIdx.x]);
  assert(histogram[threadIdx.x] == 128);
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
#if !defined(_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE)
  NV_IF_TARGET(NV_IS_HOST, (launch_kernel();))
#endif
  return 0;
}
