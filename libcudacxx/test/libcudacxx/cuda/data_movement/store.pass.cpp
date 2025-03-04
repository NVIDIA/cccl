//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/data_movement>
#include <cuda/ptx>
#include <cuda/std/cassert>

__device__ unsigned memory;

template <typename Eviction>
__device__ void store_call(Eviction eviction)
{
  auto local = cuda::ptx::get_sreg_clock();
  cuda::device::store(local, &memory, eviction);
  __threadfence();
  assert(memory == local);
  __threadfence();
}

__global__ void store_kernel()
{
  store_call(cuda::device::eviction_none);
#if __CUDA_ARCH__ >= 700
  store_call(cuda::device::eviction_normal);
  store_call(cuda::device::eviction_unchanged);
  store_call(cuda::device::eviction_first);
  store_call(cuda::device::eviction_last);
  store_call(cuda::device::eviction_no_alloc);
#endif
}

//----------------------------------------------------------------------------------------------------------------------
// setup

int main(int, char**)
{
#if !defined(__CUDA_ARCH__)
  store_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
#endif
  return 0;
}
