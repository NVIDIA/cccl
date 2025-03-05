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

__device__ unsigned input;
__device__ unsigned output;

template <typename Access, typename Eviction, typename Prefetch>
__device__ void load_call(Access access, Eviction eviction, Prefetch prefetch)
{
  auto local = cuda::ptx::get_sreg_clock();
  input      = local;
  __threadfence();
  output = cuda::device::load(&input, access, eviction, prefetch);
  assert(output == local);
  __threadfence();
}

template <typename Access, typename Eviction>
__device__ void load_call(Access access, Eviction eviction)
{
  load_call(access, eviction, cuda::device::prefetch_spatial_none);
  load_call(access, eviction, cuda::device::prefetch_64B);
  load_call(access, eviction, cuda::device::prefetch_128B);
  load_call(access, eviction, cuda::device::prefetch_256B);
}

template <typename Access>
__device__ void load_call(Access access)
{
  load_call(access, cuda::device::eviction_none);
  load_call(access, cuda::device::eviction_normal);
  load_call(access, cuda::device::eviction_unchanged);
  load_call(access, cuda::device::eviction_first);
  load_call(access, cuda::device::eviction_last);
  load_call(access, cuda::device::eviction_no_alloc);
}

__global__ void load_kernel()
{
  load_call(cuda::device::read_write);
  load_call(cuda::device::read_only);
}

//----------------------------------------------------------------------------------------------------------------------
// setup

int main(int, char**)
{
#if !defined(__CUDA_ARCH__)
  load_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
#endif
  return 0;
}
