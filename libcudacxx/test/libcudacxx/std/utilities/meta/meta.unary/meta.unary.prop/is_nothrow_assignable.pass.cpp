//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_assignable

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_is_nothrow_assignable()
{
  static_assert((cuda::std::is_nothrow_assignable<T, U>::value), "");
#if TEST_STD_VER > 2011
  static_assert((cuda::std::is_nothrow_assignable_v<T, U>), "");
#endif
}

template <class T, class U>
__host__ __device__ void test_is_not_nothrow_assignable()
{
  static_assert((!cuda::std::is_nothrow_assignable<T, U>::value), "");
#if TEST_STD_VER > 2011
  static_assert((!cuda::std::is_nothrow_assignable_v<T, U>), "");
#endif
}

struct A
{};

struct B
{
  __host__ __device__ void operator=(A);
};

struct C
{
  __host__ __device__ void operator=(C&); // not const
};

int main(int, char**)
{
  test_is_nothrow_assignable<int&, int&>();
  test_is_nothrow_assignable<int&, int>();
#if !defined(_LIBCUDACXX_HAS_NOEXCEPT_SFINAE)
  // The `__has_nothrow_assign`-based fallback for can't handle this case.
  test_is_nothrow_assignable<int&, double>();
#endif

  test_is_not_nothrow_assignable<int, int&>();
  test_is_not_nothrow_assignable<int, int>();

  test_is_not_nothrow_assignable<A, B>();
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  test_is_not_nothrow_assignable<B, A>();
  test_is_not_nothrow_assignable<C, C&>();
#endif // !TEST_COMPILER_BROKEN_SMF_NOEXCEPT

  return 0;
}
