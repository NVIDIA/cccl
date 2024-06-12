//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: sanitizer-new-delete

// It is not possible to overwrite device operator new
// UNSUPPORTED: true

// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(); // T is not array
//
// template<class T>
//   constexpr unique_ptr<T> make_unique_for_overwrite(size_t n); // T is U[]

// Test the object is not value initialized

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstddef>
#include <cuda/std/cstdlib>

#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4310) // cast truncates constant value
#endif // TEST_COMPILER_MSVC

constexpr char pattern = (char) 0xDE;

void* operator new(cuda::std::size_t count)
{
  void* ptr = malloc(count);
  for (cuda::std::size_t i = 0; i < count; ++i)
  {
    *(reinterpret_cast<char*>(ptr) + i) = pattern;
  }
  return ptr;
}

void* operator new[](cuda::std::size_t count)
{
  return ::operator new(count);
}

void operator delete(void* ptr) noexcept
{
  free(ptr);
}

void operator delete[](void* ptr) noexcept
{
  ::operator delete(ptr);
}

#ifdef TEST_COMPILER_GCC
void operator delete(void* ptr, cuda::std::size_t) noexcept
{
  free(ptr);
}
void operator delete[](void* ptr, cuda::std::size_t) noexcept
{
  ::operator delete(ptr);
}
#endif // TEST_COMPILER_GCC

__host__ __device__ void test()
{
  {
    decltype(auto) ptr = cuda::std::make_unique_for_overwrite<int>();
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<int>, decltype(ptr)>, "");
    NV_IF_TARGET(NV_IS_HOST, (assert(*(reinterpret_cast<char*>(ptr.get())) == pattern);))
  }
  {
    decltype(auto) ptr = cuda::std::make_unique_for_overwrite<int[]>(3);
    static_assert(cuda::std::same_as<cuda::std::unique_ptr<int[]>, decltype(ptr)>, "");
    NV_IF_TARGET(NV_IS_HOST, (assert(*(reinterpret_cast<char*>(&ptr[0])) == pattern);))
    NV_IF_TARGET(NV_IS_HOST, (assert(*(reinterpret_cast<char*>(&ptr[1])) == pattern);))
    NV_IF_TARGET(NV_IS_HOST, (assert(*(reinterpret_cast<char*>(&ptr[2])) == pattern);))
  }
}

int main(int, char**)
{
  test();

  return 0;
}
