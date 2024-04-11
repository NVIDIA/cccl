//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_volatile

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_remove_volatile_imp()
{
  ASSERT_SAME_TYPE(U, typename cuda::std::remove_volatile<T>::type);
#if TEST_STD_VER > 2011
  ASSERT_SAME_TYPE(U, cuda::std::remove_volatile_t<T>);
#endif
}

template <class T>
__host__ __device__ void test_remove_volatile()
{
  test_remove_volatile_imp<T, T>();
  test_remove_volatile_imp<const T, const T>();
  test_remove_volatile_imp<volatile T, T>();
  test_remove_volatile_imp<const volatile T, const T>();
}

int main(int, char**)
{
  test_remove_volatile<void>();
  test_remove_volatile<int>();
  test_remove_volatile<int[3]>();
  test_remove_volatile<int&>();
  test_remove_volatile<const int&>();
  test_remove_volatile<int*>();
  test_remove_volatile<volatile int*>();

  return 0;
}
