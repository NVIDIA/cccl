//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// divides

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr with_device_op operator/(const with_device_op&, const with_device_op&)
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
  const cuda::std::divides<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  typedef cuda::std::divides<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(f(36, 4) == 9);

  typedef cuda::std::divides<> F2;
  const F2 f2 = F2();
  assert(f2(36, 4) == 9);
  assert(f2(36.0, 4) == 9);
  assert(f2(18, 4.0) == 4.5); // exact in binary
  constexpr int foo = cuda::std::divides<int>()(3, 2);
  static_assert(foo == 1, "");

  constexpr double bar = cuda::std::divides<>()(3.0, 2);
  static_assert(bar == 1.5, ""); // exact in binary

  return 0;
}
