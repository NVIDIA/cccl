//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// alignment_of

#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T, unsigned A>
__host__ __device__ void test_alignment_of()
{
  const unsigned AlignofResult = TEST_ALIGNOF(T);
  static_assert(AlignofResult == A, "Golden value does not match result of alignof keyword");
  static_assert(cuda::std::alignment_of<T>::value == AlignofResult, "");
  static_assert(cuda::std::alignment_of<T>::value == A, "");
  static_assert(cuda::std::alignment_of<const T>::value == A, "");
  static_assert(cuda::std::alignment_of<volatile T>::value == A, "");
  static_assert(cuda::std::alignment_of<const volatile T>::value == A, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::alignment_of_v<T> == A, "");
  static_assert(cuda::std::alignment_of_v<const T> == A, "");
  static_assert(cuda::std::alignment_of_v<volatile T> == A, "");
  static_assert(cuda::std::alignment_of_v<const volatile T> == A, "");
#endif
}

class Class
{
public:
  __host__ __device__ ~Class();
};

int main(int, char**)
{
  test_alignment_of<int&, 4>();
  test_alignment_of<Class, 1>();
  test_alignment_of<int*, sizeof(intptr_t)>();
  test_alignment_of<const int*, sizeof(intptr_t)>();
  test_alignment_of<char[3], 1>();
  test_alignment_of<int, 4>();
  // The test case below is a hack. It's hard to detect what golden value
  // we should expect. In most cases it should be 8. But in i386 builds
  // with Clang >= 8 or GCC >= 8 the value is '4'.
  test_alignment_of<double, TEST_ALIGNOF(double)>();
  test_alignment_of<bool, 1>();
  test_alignment_of<unsigned, 4>();

  return 0;
}
