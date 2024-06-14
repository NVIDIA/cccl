//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// negate

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr with_device_op operator-(const with_device_op&)
  {
    return {};
  }
  __device__ constexpr operator bool() const
  {
    return true;
  }
};

__global__ void test_global_kernel()
{
  const cuda::std::negate<with_device_op> f;
  assert(f({}));
}

int main(int, char**)
{
  typedef cuda::std::negate<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<F::argument_type, int>::value), "");
  static_assert((cuda::std::is_same<F::result_type, int>::value), "");
#endif
  assert(f(36) == -36);

  typedef cuda::std::negate<> F2;
  const F2 f2 = F2();
  assert(f2(36) == -36);
  assert(f2(36L) == -36);
  assert(f2(36.0) == -36);
#if TEST_STD_VER > 2011
  constexpr int foo = cuda::std::negate<int>()(3);
  static_assert(foo == -3, "");

  constexpr double bar = cuda::std::negate<>()(3.0);
  static_assert(bar == -3.0, "");
#endif

  return 0;
}
