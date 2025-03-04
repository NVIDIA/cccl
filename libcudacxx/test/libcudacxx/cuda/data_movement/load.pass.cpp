//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/data_movement>
#include <cuda/std/cassert>

__device__ unsigned input;
__device__ unsigned output;

template <typename Behavior, typename Eviction, typename Prefetch>
__device__ void load_call(Behavior behavior, Eviction eviction, Prefetch prefetch)
{
#if __libcuda_ptx_isa > 0
  output = cuda::load(&input, behavior, eviction, prefetch);
  printf("%u\n", output);
#endif
}

template <typename Behavior, typename Eviction>
__device__ void load_call(Behavior behavior, Eviction eviction)
{
  load_call(behavior, eviction, cuda::prefetch_spatial_none);
  load_call(behavior, eviction, cuda::prefetch_64B);
  load_call(behavior, eviction, cuda::prefetch_128B);
  load_call(behavior, eviction, cuda::prefetch_256B);
}

template <typename Behavior>
__device__ void load_call(Behavior behavior)
{
  load_call(behavior, cuda::eviction_none);
  load_call(behavior, cuda::eviction_normal);
  load_call(behavior, cuda::eviction_unchanged);
  load_call(behavior, cuda::eviction_first);
  load_call(behavior, cuda::eviction_last);
  load_call(behavior, cuda::eviction_no_alloc);
}

__global__ void load_kernel()
{
  load_call(cuda::read_only);
  load_call(cuda::read_write);
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
