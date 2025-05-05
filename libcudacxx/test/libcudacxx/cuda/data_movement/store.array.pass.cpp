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
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

template <typename T, size_t N>
__device__ void update(cuda::std::array<T, N>& array)
{
  for (size_t i = 0; i < N; ++i)
  {
    array[i] = static_cast<T>(cuda::ptx::get_sreg_clock());
  }
}

template <size_t Align, typename T, size_t N, typename Eviction>
__device__ void store_call(cuda::std::array<T, N>& value, T* output, Eviction eviction)
{
  update(value);
  cuda::device::store<N>(output, value, cuda::aligned_size_t<Align>{Align}, eviction);
  __threadfence();
  auto result = *reinterpret_cast<cuda::std::array<T, N>*>(output);
  assert(result == value);
  __threadfence();
}

template <size_t Align, typename T, size_t N>
__device__ void store_call(cuda::std::array<T, N>& value, T* output)
{
  store_call<Align>(value, output, cuda::device::cache_reuse_unchanged);
  store_call<Align>(value, output, cuda::device::cache_reuse_normal);
  store_call<Align>(value, output, cuda::device::cache_reuse_unchanged);
  store_call<Align>(value, output, cuda::device::cache_reuse_low);
  store_call<Align>(value, output, cuda::device::cache_reuse_high);
  store_call<Align>(value, output, cuda::device::cache_no_reuse);
}

__device__ uint32_t pointer[256];

__global__ void store_kernel()
{
  cuda::std::array<uint32_t, 64> array;
  store_call<4>(array, pointer);
  store_call<8>(array, pointer);
  store_call<16>(array, pointer);
  store_call<32>(array, pointer);
  store_call<64>(array, pointer);
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
