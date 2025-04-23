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

template <size_t Align, size_t N, typename T, typename Access, typename Eviction, typename Prefetch>
__device__ void load_call(T* input, cuda::std::array<T, N>& value, Access access, Eviction eviction, Prefetch prefetch)
{
  update(value);
  *reinterpret_cast<cuda::std::array<T, N>*>(input) = value;
  __threadfence();
  auto result = cuda::device::load<N>(input, cuda::aligned_size_t<Align>{Align}, access, eviction, prefetch);
  assert(result == value);
  __threadfence();
}

template <size_t Align, size_t N, typename T, typename Access, typename Eviction>
__device__ void load_call(T* input, cuda::std::array<T, N>& value, Access access, Eviction eviction)
{
  load_call<Align>(input, value, access, eviction, cuda::device::prefetch_L2_none);
  load_call<Align>(input, value, access, eviction, cuda::device::prefetch_L2_64B);
  load_call<Align>(input, value, access, eviction, cuda::device::prefetch_L2_128B);
  load_call<Align>(input, value, access, eviction, cuda::device::prefetch_L2_256B);
}

template <size_t Align, size_t N, typename T, typename Access>
__device__ void load_call(T* input, cuda::std::array<T, N>& value, Access access)
{
  load_call<Align>(input, value, access, cuda::device::eviction_none);
  load_call<Align>(input, value, access, cuda::device::eviction_normal);
  load_call<Align>(input, value, access, cuda::device::eviction_unchanged);
  load_call<Align>(input, value, access, cuda::device::eviction_first);
  load_call<Align>(input, value, access, cuda::device::eviction_last);
  load_call<Align>(input, value, access, cuda::device::eviction_no_alloc);
}

template <size_t Align, size_t N, typename T>
__device__ void load_call(T* input, cuda::std::array<T, N>& value)
{
  load_call<Align>(input, value, cuda::device::read_write);
  load_call<Align>(input, value, cuda::device::read_only);
}

__device__ uint32_t pointer[256];

__global__ void load_kernel()
{
  cuda::std::array<uint32_t, 64> input;
  load_call<4>(pointer, input);
  load_call<8>(pointer, input);
  load_call<16>(pointer, input);
  load_call<32>(pointer, input);
  load_call<64>(pointer, input);

  cuda::std::array<uint32_t, 12> input2;
  load_call<4>(pointer, input2);
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
