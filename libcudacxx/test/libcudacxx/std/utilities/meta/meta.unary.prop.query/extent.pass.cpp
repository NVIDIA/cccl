//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// extent

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, unsigned A>
__host__ __device__ void test_extent()
{
  static_assert((cuda::std::extent<T>::value == A), "");
  static_assert((cuda::std::extent<const T>::value == A), "");
  static_assert((cuda::std::extent<volatile T>::value == A), "");
  static_assert((cuda::std::extent<const volatile T>::value == A), "");
  static_assert((cuda::std::extent_v<T> == A), "");
  static_assert((cuda::std::extent_v<const T> == A), "");
  static_assert((cuda::std::extent_v<volatile T> == A), "");
  static_assert((cuda::std::extent_v<const volatile T> == A), "");
}

template <class T, unsigned A>
__host__ __device__ void test_extent1()
{
  static_assert((cuda::std::extent<T, 1>::value == A), "");
  static_assert((cuda::std::extent<const T, 1>::value == A), "");
  static_assert((cuda::std::extent<volatile T, 1>::value == A), "");
  static_assert((cuda::std::extent<const volatile T, 1>::value == A), "");
  static_assert((cuda::std::extent_v<T, 1> == A), "");
  static_assert((cuda::std::extent_v<const T, 1> == A), "");
  static_assert((cuda::std::extent_v<volatile T, 1> == A), "");
  static_assert((cuda::std::extent_v<const volatile T, 1> == A), "");
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  test_extent<void, 0>();
  test_extent<int&, 0>();
  test_extent<Class, 0>();
  test_extent<int*, 0>();
  test_extent<const int*, 0>();
  test_extent<int, 0>();
  test_extent<double, 0>();
  test_extent<bool, 0>();
  test_extent<unsigned, 0>();

  test_extent<int[2], 2>();
  test_extent<int[2][4], 2>();
  test_extent<int[][4], 0>();

  test_extent1<int, 0>();
  test_extent1<int[2], 0>();
  test_extent1<int[2][4], 4>();
  test_extent1<int[][4], 4>();

  return 0;
}
