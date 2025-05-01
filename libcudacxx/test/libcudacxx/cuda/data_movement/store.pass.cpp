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

template <typename T, size_t N>
__device__ void update(cuda::std::array<T, N>& array)
{
  for (size_t i = 0; i < N; ++i)
  {
    array[i] = static_cast<T>(cuda::ptx::get_sreg_clock());
  }
}

template <typename T, typename Eviction, typename AccessProperty>
__device__ void store_call(T& value, T* output, Eviction eviction, AccessProperty property)
{
  update(value);
  cuda::device::store(value, output, eviction, property);
  __threadfence();
  assert(*output == value);
  __threadfence();
}

template <typename T, typename Prefetch>
__device__ void store_call(T& value, T* output, Prefetch prefetch)
{
  store_call(value, output, prefetch, cuda::access_property::global{});
  store_call(value, output, prefetch, cuda::access_property::streaming{});
  store_call(value, output, prefetch, cuda::access_property::persisting{});
}

template <typename T>
__device__ void store_call(T& value, T* output)
{
  store_call(value, output, cuda::device::L1_unchanged_reuse);
  store_call(value, output, cuda::device::L1_normal_reuse);
  store_call(value, output, cuda::device::L1_unchanged_reuse);
  store_call(value, output, cuda::device::L1_low_reuse);
  store_call(value, output, cuda::device::L1_high_reuse);
  store_call(value, output, cuda::device::L1_no_reuse);
}

__device__ uint8_t pointer[256];

__global__ void store_kernel()
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
  store_call(input1, reinterpret_cast<Bytes1*>(&pointer));
  store_call(input2, reinterpret_cast<Bytes2*>(&pointer));
  store_call(input4, reinterpret_cast<Bytes4*>(&pointer));
  store_call(input8, reinterpret_cast<Bytes8*>(&pointer));
  store_call(input16, reinterpret_cast<Bytes16*>(&pointer));
  store_call(input32, reinterpret_cast<Bytes32*>(&pointer));
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
