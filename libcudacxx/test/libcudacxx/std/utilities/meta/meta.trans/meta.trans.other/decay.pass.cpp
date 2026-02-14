//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// decay

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, class U>
__host__ __device__ void test_decay()
{
  static_assert(cuda::std::is_same_v<U, typename cuda::std::decay<T>::type>);
  static_assert(cuda::std::is_same_v<U, cuda::std::decay_t<T>>);
}

int main(int, char**)
{
  test_decay<void, void>();
  test_decay<int, int>();
  test_decay<const volatile int, int>();
  test_decay<int*, int*>();
  test_decay<int[3], int*>();
  test_decay<const int[3], const int*>();
  test_decay<void(), void (*)()>();
  test_decay<int(int) const, int(int) const>();
  test_decay<int(int) volatile, int(int) volatile>();
  test_decay<int(int) &, int(int) &>();
  test_decay<int(int) &&, int(int) &&>();

  return 0;
}
