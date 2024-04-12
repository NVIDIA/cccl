//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// const_reference operator[](size_type) const; // constexpr in C++14
// Libc++ marks it as noexcept

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
  {
    typedef double T;
    typedef cuda::std::array<T, 3> C;
    C const c = {1, 2, 3.5};
    LIBCPP_ASSERT_NOEXCEPT(c[0]);
    ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));
    C::const_reference r1 = c[0];
    assert(r1 == 1);
    C::const_reference r2 = c[2];
    assert(r2 == 3.5);
  }
  // Test operator[] "works" on zero sized arrays
  {
    {
      typedef double T;
      typedef cuda::std::array<T, 0> C;
      C const c = {};
      LIBCPP_ASSERT_NOEXCEPT(c[0]);
      ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));
      if (c.size() > (0))
      { // always false
#if !defined(TEST_COMPILER_MSVC)
        C::const_reference r = c[0];
        unused(r);
#endif // !TEST_COMPILER_MSVC
      }
    }
    {
      typedef double T;
      typedef cuda::std::array<T const, 0> C;
      C const c = {};
      LIBCPP_ASSERT_NOEXCEPT(c[0]);
      ASSERT_SAME_TYPE(C::const_reference, decltype(c[0]));
      if (c.size() > (0))
      { // always false
#if !defined(TEST_COMPILER_MSVC)
        C::const_reference r = c[0];
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
#if TEST_STD_VER >= 2014 && !defined(TEST_COMPILER_MSVC_2017)
  static_assert(tests(), "");
#endif
  return 0;
}
