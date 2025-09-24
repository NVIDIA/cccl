//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdlib>

template <class T>
__host__ __device__ volatile T* make_array(cuda::std::size_t n)
{
  auto ptr = static_cast<T*>(cuda::std::malloc(n * sizeof(T)));
  assert(ptr != nullptr);

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    ptr[i] = static_cast<T>(i);
  }

  return const_cast<volatile T*>(ptr);
}

__host__ __device__ void destroy_array(volatile void* ptr)
{
  cuda::std::free(const_cast<void*>(ptr));
}

__device__ __host__ void test()
{
  using T = int;

  constexpr cuda::std::size_t n      = 128;
  constexpr cuda::std::size_t nbytes = n * sizeof(T);

  // 1. Test on well aligned memory
  {
    auto ptr = make_array<T>(n);
    cuda::discard_memory(ptr, nbytes);
    destroy_array(ptr);
  }

  // 2. Test on misaligned begin address
  {
    auto ptr = make_array<T>(n);
    cuda::discard_memory(reinterpret_cast<volatile unsigned char*>(ptr) + 1, nbytes - 1);
    destroy_array(ptr);
  }

  // 3. Test on misaligned end address
  {
    auto ptr = make_array<T>(n);
    cuda::discard_memory(ptr, nbytes - 1);
    destroy_array(ptr);
  }

  // 4. Test on misaligned begin and end address
  {
    auto ptr = make_array<T>(n);
    cuda::discard_memory(reinterpret_cast<volatile unsigned char*>(ptr) + 1, nbytes - 2);
    destroy_array(ptr);
  }
}

int main(int, char**)
{
  test();
  return 0;
}
