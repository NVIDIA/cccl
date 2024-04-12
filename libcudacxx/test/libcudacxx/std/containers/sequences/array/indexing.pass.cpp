//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// reference operator[](size_type); // constexpr in C++17
// Libc++ marks it as noexcept

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C c = {1, 2, 3.5};
    LIBCPP_ASSERT_NOEXCEPT(c[0]);
    ASSERT_SAME_TYPE(C::reference, decltype(c[0]));
    C::reference r1 = c[0];
    assert(r1 == 1);
    r1 = 5.5;
    assert(c.front() == 5.5);

    C::reference r2 = c[2];
    assert(r2 == 3.5);
    r2 = 7.5;
    assert(c.back() == 7.5);
  }

  // Test operator[] "works" on zero sized arrays
  {
    {
      typedef double T;
      typedef cuda::std::array<T, 0> C;
      C c = {};
      LIBCPP_ASSERT_NOEXCEPT(c[0]);
      ASSERT_SAME_TYPE(C::reference, decltype(c[0]));
      if (c.size() > (0))
      { // always false
#if !defined(TEST_COMPILER_MSVC)
        C::reference r = c[0];
        unused(r);
#endif // !TEST_COMPILER_MSVC
      }
    }
    {
      typedef double T;
      typedef cuda::std::array<const T, 0> C;
      C c = {};
      LIBCPP_ASSERT_NOEXCEPT(c[0]);
      ASSERT_SAME_TYPE(C::reference, decltype(c[0]));
      if (c.size() > (0))
      { // always false
#if !defined(TEST_COMPILER_MSVC)
        C::reference r = c[0];
        unused(r);
#endif // !TEST_COMPILER_MSVC
      }
    }
  }

  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER >= 2014
  static_assert(tests(), "");
#endif
  return 0;
}
