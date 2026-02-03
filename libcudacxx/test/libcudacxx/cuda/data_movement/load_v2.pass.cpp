//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__data_movement_v2/load.h>
#include <cuda/ptx>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include "test_macros.h"

template <typename T, size_t N>
__device__ void update(cuda::std::array<T, N>& array)
{
  for (size_t i = 0; i < N; ++i)
  {
    array[i] = static_cast<T>(cuda::ptx::get_sreg_clock());
  }
}

template <typename T, typename Access, typename L1Reuse, typename AccessProperty, typename Prefetch>
__device__ void load_call(T* input, T& value, Access access, L1Reuse l1_reuse, AccessProperty l2_hint, Prefetch prefetch)
{
  update(value);
  *input = value;
  __threadfence();
  [[maybe_unused]] auto result = cuda::device::load(input, access | l1_reuse | l2_hint | prefetch);
  assert(result == value);
  __threadfence();
}

template <typename T, typename Access, typename L1Reuse, typename L2Hint>
__device__ void load_call(T* input, T& value, Access access, L1Reuse l1_reuse, L2Hint l2_hint)
{
  load_call(input, value, access, l1_reuse, l2_hint, cuda::device::enable_l2_prefetch);
}

template <typename T, typename Access, typename L1Reuse>
__device__ void load_call(T* input, T& value, Access access, L1Reuse l1_reuse)
{
  load_call(input, value, access, l1_reuse, cuda::access_property::global{});
  load_call(input, value, access, l1_reuse, cuda::access_property::normal{});
  load_call(input, value, access, l1_reuse, cuda::access_property::streaming{});
  load_call(input, value, access, l1_reuse, cuda::access_property::persisting{});
}

template <typename T, typename Access>
__device__ void load_call(T* input, T& value, Access access)
{
  load_call(input, value, access, cuda::device::cache_reuse_low);
  load_call(input, value, access, cuda::device::cache_reuse_high);
  load_call(input, value, access, cuda::device::cache_no_reuse);
}

template <typename T>
__device__ void load_call(T* input, T& value)
{
  load_call(input, value, cuda::device::read_write);
  load_call(input, value, cuda::device::read_only);
}

__device__ uint8_t pointer[256];

__global__ void load_kernel()
{
  using Bytes1  = cuda::std::array<uint8_t, 1>;
  using Bytes2  = cuda::std::array<uint16_t, 1>;
  using Bytes4  = cuda::std::array<uint32_t, 1>;
  using Bytes8  = cuda::std::array<uint32_t, 2>;
  using Bytes16 = cuda::std::array<uint32_t, 4>;
  using Bytes32 = cuda::std::array<uint32_t, 8>;
  Bytes1 input1;
  Bytes2 input2;
  Bytes4 input4;
  Bytes8 input8;
  Bytes16 input16;
  Bytes32 input32;
  load_call(reinterpret_cast<Bytes1*>(pointer), input1);
  load_call(reinterpret_cast<Bytes2*>(pointer), input2);
  load_call(reinterpret_cast<Bytes4*>(pointer), input4);
  load_call(reinterpret_cast<Bytes8*>(pointer), input8);
  load_call(reinterpret_cast<Bytes16*>(pointer), input16);
  load_call(reinterpret_cast<Bytes32*>(pointer), input32);

  unused(cuda::device::load(reinterpret_cast<Bytes4*>(pointer), cuda::device::read_only));
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
