//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <new>

// template <class T> constexpr T* launder(T* p) noexcept;

#include <cuda/std/__new_>
#include <cuda/std/cassert>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE constexpr int gi   = 5;
TEST_GLOBAL_VARIABLE constexpr float gf = 8.f;

__host__ __device__ constexpr bool test()
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
#if defined(_CCCL_BUILTIN_LAUNDER)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_LAUNDER

  return 0;
}
