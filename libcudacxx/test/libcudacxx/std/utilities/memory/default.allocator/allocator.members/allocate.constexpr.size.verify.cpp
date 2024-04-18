//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// allocator:
// constexpr T* allocate(size_type n);

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cuda/std/__memory_>
#include <cuda/std/cstddef>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool test()
{
  typedef cuda::std::allocator<T> A;
  typedef cuda::std::allocator_traits<A> AT;
  A a;
  TEST_IGNORE_NODISCARD a.allocate(AT::max_size(a) + 1); // just barely too large
  TEST_IGNORE_NODISCARD a.allocate(AT::max_size(a) * 2); // significantly too large
  TEST_IGNORE_NODISCARD a.allocate(((cuda::std::size_t) -1) / sizeof(T) + 1); // multiply will overflow
  TEST_IGNORE_NODISCARD a.allocate((cuda::std::size_t) -1); // way too large

  return true;
}

__host__ __device__ void f()
{
  static_assert(test<double>()); // expected-error-re {{{{(static_assert|static assertion)}} expression is not an
                                 // integral constant expression}}
  LIBCPP_STATIC_ASSERT(test<const double>()); // expected-error-re {{{{(static_assert|static assertion)}} expression is
                                              // not an integral constant expression}}
}

int main(int, char**)
{
  return 0;
}
