//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_cv

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_remove_cv_imp()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::remove_cv<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::remove_cv_t<T>>);
}

template <class T>
__host__ __device__ void test_remove_cv()
{
  test_remove_cv_imp<T, T>();
  test_remove_cv_imp<const T, T>();
  test_remove_cv_imp<volatile T, T>();
  test_remove_cv_imp<const volatile T, T>();
}

int main(int, char**)
{
  test_remove_cv<void>();
  test_remove_cv<int>();
  test_remove_cv<int[3]>();
  test_remove_cv<int&>();
  test_remove_cv<const int&>();
  test_remove_cv<int*>();
  test_remove_cv<const int*>();

  return 0;
}
