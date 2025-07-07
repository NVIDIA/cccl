//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// bit_or

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr with_device_op operator|(const with_device_op&, const with_device_op&)
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
  const cuda::std::bit_or<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  typedef cuda::std::bit_or<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(f(0xEA95, 0xEA95) == 0xEA95);
  assert(f(0xEA95, 0x58D3) == 0xFAD7);
  assert(f(0x58D3, 0xEA95) == 0xFAD7);
  assert(f(0x58D3, 0) == 0x58D3);
  assert(f(0xFFFF, 0x58D3) == 0xFFFF);

  typedef cuda::std::bit_or<> F2;
  const F2 f2 = F2();
  assert(f2(0xEA95, 0xEA95) == 0xEA95);
  assert(f2(0xEA95L, 0xEA95) == 0xEA95);
  assert(f2(0xEA95, 0xEA95L) == 0xEA95);

  assert(f2(0xEA95, 0x58D3) == 0xFAD7);
  assert(f2(0xEA95L, 0x58D3) == 0xFAD7);
  assert(f2(0xEA95, 0x58D3L) == 0xFAD7);

  assert(f2(0x58D3, 0xEA95) == 0xFAD7);
  assert(f2(0x58D3L, 0xEA95) == 0xFAD7);
  assert(f2(0x58D3, 0xEA95L) == 0xFAD7);

  assert(f2(0x58D3, 0) == 0x58D3);
  assert(f2(0x58D3L, 0) == 0x58D3);
  assert(f2(0x58D3, 0L) == 0x58D3);

  assert(f2(0xFFFF, 0x58D3) == 0xFFFF);
  assert(f2(0xFFFFL, 0x58D3) == 0xFFFF);
  assert(f2(0xFFFF, 0x58D3L) == 0xFFFF);
  constexpr int foo = cuda::std::bit_or<int>()(0x58D3, 0xEA95);
  static_assert(foo == 0xFAD7, "");

  constexpr int bar = cuda::std::bit_or<>()(0x58D3L, 0xEA95);
  static_assert(bar == 0xFAD7, "");

  return 0;
}
