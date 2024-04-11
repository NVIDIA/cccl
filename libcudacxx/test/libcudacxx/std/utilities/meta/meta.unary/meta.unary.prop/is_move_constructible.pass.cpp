//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_move_constructible

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_move_constructible()
{
  static_assert(cuda::std::is_move_constructible<T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_move_constructible_v<T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_move_constructible()
{
  static_assert(!cuda::std::is_move_constructible<T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_move_constructible_v<T>, "");
#endif
}

class Empty
{};

class NotEmpty
{
public:
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
public:
  __host__ __device__ virtual ~Abstract() = 0;
};

struct A
{
  __host__ __device__ A(const A&);
};

struct B
{
  __host__ __device__ B(B&&);
};

int main(int, char**)
{
#if !defined(TEST_COMPILER_GCC) || TEST_STD_VER < 2020
  test_is_not_move_constructible<char[3]>();
  test_is_not_move_constructible<char[]>();
#endif
  test_is_not_move_constructible<void>();
  test_is_not_move_constructible<Abstract>();

  test_is_move_constructible<A>();
  test_is_move_constructible<int&>();
  test_is_move_constructible<Union>();
  test_is_move_constructible<Empty>();
  test_is_move_constructible<int>();
  test_is_move_constructible<double>();
  test_is_move_constructible<int*>();
  test_is_move_constructible<const int*>();
  test_is_move_constructible<NotEmpty>();
  test_is_move_constructible<bit_zero>();
  test_is_move_constructible<B>();

  return 0;
}
