//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/data_movement>
#include <cuda/ptx>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include "test_macros.h"

__global__ void load_kernel()
{
  auto ptr1 = static_cast<const uint32_t*>(nullptr);
  unused(cuda::device::load(ptr1)); // nullptr

  __shared__ uint32_t smem[16];
  unused(cuda::device::load(smem)); // shared memory

  auto null_ptr = static_cast<const void*>(nullptr);
  auto ptr2     = reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(null_ptr) + 2);
  unused(cuda::device::load(ptr2)); // non aligned
}

__global__ void store_kernel()
{
  auto ptr1 = static_cast<uint32_t*>(nullptr);
  cuda::device::store(2u, ptr1); // nullptr

  // PTX compiler crashes
  //__shared__ uint32_t smem[16];
  // cuda::device::store(2u, static_cast<uint32_t*>(smem + 1)); // shared memory

  auto null_ptr = static_cast<void*>(nullptr);
  auto ptr2     = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(null_ptr) + 2);
  cuda::device::store(2u, ptr2); // non aligned
}

//----------------------------------------------------------------------------------------------------------------------
// setup

int main(int, char**)
{
#if !defined(__CUDA_ARCH__)
  load_kernel<<<1, 1>>>();
  store_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
#endif
  return 0;
}
