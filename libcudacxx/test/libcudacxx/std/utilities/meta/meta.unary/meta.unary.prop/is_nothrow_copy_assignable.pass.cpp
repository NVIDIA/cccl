//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_nothrow_copy_assignable

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_has_nothrow_assign()
{
  static_assert(cuda::std::is_nothrow_copy_assignable<T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_nothrow_copy_assignable_v<T>, "");
#endif
}

template <class T>
__host__ __device__ void test_has_not_nothrow_assign()
{
  static_assert(!cuda::std::is_nothrow_copy_assignable<T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_nothrow_copy_assignable_v<T>, "");
#endif
}

class Empty
{};

struct NotEmpty
{
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

struct A
{
  __host__ __device__ A& operator=(const A&);
};

int main(int, char**)
{
  test_has_nothrow_assign<int&>();
  test_has_nothrow_assign<Union>();
  test_has_nothrow_assign<Empty>();
  test_has_nothrow_assign<int>();
  test_has_nothrow_assign<double>();
  test_has_nothrow_assign<int*>();
  test_has_nothrow_assign<const int*>();
  test_has_nothrow_assign<NotEmpty>();
  test_has_nothrow_assign<bit_zero>();

  test_has_not_nothrow_assign<const int>();
  test_has_not_nothrow_assign<void>();
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  test_has_not_nothrow_assign<A>();
#endif // !TEST_COMPILER_BROKEN_SMF_NOEXCEPT

  return 0;
}
