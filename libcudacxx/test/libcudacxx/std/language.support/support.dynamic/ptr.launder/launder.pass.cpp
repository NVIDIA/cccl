//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvcc-11.1

// <new>

// template <class T> constexpr T* launder(T* p) noexcept;

#include <cuda/std/__new_>
#include <cuda/std/cassert>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL int gi   = 5;
STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL float gf = 8.f;

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  assert(cuda::std::launder(&gi) == &gi);
  assert(cuda::std::launder(&gf) == &gf);

  const int* i   = &gi;
  const float* f = &gf;
  static_assert(cuda::std::is_same<decltype(i), decltype(cuda::std::launder(i))>::value, "");
  static_assert(cuda::std::is_same<decltype(f), decltype(cuda::std::launder(f))>::value, "");

  assert(cuda::std::launder(i) == i);
  assert(cuda::std::launder(f) == f);

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014 && defined(_CCCL_BUILTIN_LAUNDER)
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014 && _CCCL_BUILTIN_LAUNDER

  return 0;
}
