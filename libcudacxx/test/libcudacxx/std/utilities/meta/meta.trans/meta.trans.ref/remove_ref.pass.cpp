//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// remove_reference

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_remove_reference()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::remove_reference<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::remove_reference_t<T>>);
}

int main(int, char**)
{
  test_remove_reference<void, void>();
  test_remove_reference<int, int>();
  test_remove_reference<int[3], int[3]>();
  test_remove_reference<int*, int*>();
  test_remove_reference<const int*, const int*>();

  test_remove_reference<int&, int>();
  test_remove_reference<const int&, const int>();
  test_remove_reference<int (&)[3], int[3]>();
  test_remove_reference<int*&, int*>();
  test_remove_reference<const int*&, const int*>();

  test_remove_reference<int&&, int>();
  test_remove_reference<const int&&, const int>();
  test_remove_reference<int (&&)[3], int[3]>();
  test_remove_reference<int*&&, int*>();
  test_remove_reference<const int*&&, const int*>();

  return 0;
}
