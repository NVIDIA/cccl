//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// has_nothrow_move_constructor

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_nothrow_move_constructible()
{
  static_assert(cuda::std::is_nothrow_move_constructible<T>::value, "");
  static_assert(cuda::std::is_nothrow_move_constructible<const T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_nothrow_move_constructible_v<T>, "");
  static_assert(cuda::std::is_nothrow_move_constructible_v<const T>, "");
#endif
}

template <class T>
__host__ __device__ void test_has_not_nothrow_move_constructor()
{
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  static_assert(!cuda::std::is_nothrow_move_constructible<T>::value, "");
  static_assert(!cuda::std::is_nothrow_move_constructible<const T>::value, "");
#endif // !TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  static_assert(!cuda::std::is_nothrow_move_constructible<volatile T>::value, "");
  static_assert(!cuda::std::is_nothrow_move_constructible<const volatile T>::value, "");
#if TEST_STD_VER > 2011
#  ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  static_assert(!cuda::std::is_nothrow_move_constructible_v<T>, "");
  static_assert(!cuda::std::is_nothrow_move_constructible_v<const T>, "");
#  endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  static_assert(!cuda::std::is_nothrow_move_constructible_v<volatile T>, "");
  static_assert(!cuda::std::is_nothrow_move_constructible_v<const volatile T>, "");
#endif // TEST_STD_VER > 2011
}

class Empty
{};

union Union
{};

struct bit_zero
{
  int : 0;
};

struct A
{
  __host__ __device__ A(const A&);
};

int main(int, char**)
{
  test_has_not_nothrow_move_constructor<void>();
  test_has_not_nothrow_move_constructor<A>();

  test_is_nothrow_move_constructible<int&>();
  test_is_nothrow_move_constructible<Union>();
  test_is_nothrow_move_constructible<Empty>();
  test_is_nothrow_move_constructible<int>();
  test_is_nothrow_move_constructible<double>();
  test_is_nothrow_move_constructible<int*>();
  test_is_nothrow_move_constructible<const int*>();
  test_is_nothrow_move_constructible<bit_zero>();

  return 0;
}
