//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_trivially_constructible;

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_trivially_constructible()
{
  static_assert((cuda::std::is_trivially_constructible<T>::value), "");
#if TEST_STD_VER > 2011
  static_assert((cuda::std::is_trivially_constructible_v<T>), "");
#endif
}

template <class T, class A0>
__host__ __device__ void test_is_trivially_constructible()
{
  static_assert((cuda::std::is_trivially_constructible<T, A0>::value), "");
#if TEST_STD_VER > 2011
  static_assert((cuda::std::is_trivially_constructible_v<T, A0>), "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_trivially_constructible()
{
  static_assert((!cuda::std::is_trivially_constructible<T>::value), "");
#if TEST_STD_VER > 2011
  static_assert((!cuda::std::is_trivially_constructible_v<T>), "");
#endif
}

template <class T, class A0>
__host__ __device__ void test_is_not_trivially_constructible()
{
  static_assert((!cuda::std::is_trivially_constructible<T, A0>::value), "");
#if TEST_STD_VER > 2011
  static_assert((!cuda::std::is_trivially_constructible_v<T, A0>), "");
#endif
}

template <class T, class A0, class A1>
__host__ __device__ void test_is_not_trivially_constructible()
{
  static_assert((!cuda::std::is_trivially_constructible<T, A0, A1>::value), "");
#if TEST_STD_VER > 2011
  static_assert((!cuda::std::is_trivially_constructible_v<T, A0, A1>), "");
#endif
}

struct A
{
  __host__ __device__ explicit A(int);
  __host__ __device__ A(int, double);
};

int main(int, char**)
{
  test_is_trivially_constructible<int>();
  test_is_trivially_constructible<int, const int&>();

  test_is_not_trivially_constructible<A, int>();
  test_is_not_trivially_constructible<A, int, double>();
  test_is_not_trivially_constructible<A>();

  return 0;
}
