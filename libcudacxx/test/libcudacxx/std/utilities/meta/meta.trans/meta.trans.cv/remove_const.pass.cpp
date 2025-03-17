//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_const

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_remove_const_imp()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::remove_const<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::remove_const_t<T>>);
}

template <class T>
__host__ __device__ void test_remove_const()
{
  test_remove_const_imp<T, T>();
  test_remove_const_imp<const T, T>();
  test_remove_const_imp<volatile T, volatile T>();
  test_remove_const_imp<const volatile T, volatile T>();
}

int main(int, char**)
{
  test_remove_const<void>();
  test_remove_const<int>();
  test_remove_const<int[3]>();
  test_remove_const<int&>();
  test_remove_const<const int&>();
  test_remove_const<int*>();
  test_remove_const<const int*>();

  return 0;
}
