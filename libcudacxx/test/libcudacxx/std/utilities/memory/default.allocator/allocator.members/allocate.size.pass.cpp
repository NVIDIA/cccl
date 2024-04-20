//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// <memory>

// allocator:
// constexpr T* allocate(size_t n);

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
template <typename T>
__host__ __device__ void test_max(cuda::std::size_t count)
{
  cuda::std::allocator<T> a;
  try
  {
    TEST_IGNORE_NODISCARD a.allocate(count);
    assert(false);
  }
  catch (const cuda::std::bad_array_new_length&)
  {}
}

template <typename T>
__host__ __device__ void test()
{
  // Bug 26812 -- allocating too large
  typedef cuda::std::allocator<T> A;
  typedef cuda::std::allocator_traits<A> AT;
  A a;
  test_max<T>(AT::max_size(a) + 1); // just barely too large
  test_max<T>(AT::max_size(a) * 2); // significantly too large
  test_max<T>(((cuda::std::size_t) -1) / sizeof(T) + 1); // multiply will overflow
  test_max<T>((cuda::std::size_t) -1); // way too large
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test<double>();));
  NV_IF_TARGET(NV_IS_HOST, (test<const double>();));
#endif // !TEST_HAS_NO_EXCEPTIONS

  return 0;
}
