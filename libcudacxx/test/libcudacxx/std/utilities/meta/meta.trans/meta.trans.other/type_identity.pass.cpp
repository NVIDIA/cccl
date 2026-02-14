//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// type_identity

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_type_identity()
{
  static_assert(cuda::std::is_same_v<T, typename cuda::std::type_identity<T>::type>);
  static_assert(cuda::std::is_same_v<T, cuda::std::type_identity_t<T>>);
}

int main(int, char**)
{
  test_type_identity<void>();
  test_type_identity<int>();
  test_type_identity<const volatile int>();
  test_type_identity<int*>();
  test_type_identity<int[3]>();
  test_type_identity<const int[3]>();

  test_type_identity<void (*)()>();
  test_type_identity<int(int) const>();
  test_type_identity<int(int) volatile>();
  test_type_identity<int(int) &>();
  test_type_identity<int(int) &&>();

  return 0;
}
