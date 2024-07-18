//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// bit_xor

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr with_device_op operator^(const with_device_op&, const with_device_op&)
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
  const cuda::std::bit_xor<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  {
    typedef cuda::std::bit_xor<int> F;
    const F f = F();
#if TEST_STD_VER <= 2017
    static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
    static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
    static_assert((cuda::std::is_same<int, F::result_type>::value), "");
#endif
    assert(f(0xEA95, 0xEA95) == 0);
    assert(f(0xEA95, 0x58D3) == 0xB246);
    assert(f(0x58D3, 0xEA95) == 0xB246);
    assert(f(0x58D3, 0) == 0x58D3);
    assert(f(0xFFFF, 0x58D3) == 0xA72C);
  }

  {
    typedef cuda::std::bit_xor<> F2;
    const F2 f = F2();
    assert(f(0xEA95, 0xEA95) == 0);
    assert(f(0xEA95L, 0xEA95) == 0);
    assert(f(0xEA95, 0xEA95L) == 0);

    assert(f(0xEA95, 0x58D3) == 0xB246);
    assert(f(0xEA95L, 0x58D3) == 0xB246);
    assert(f(0xEA95, 0x58D3L) == 0xB246);

    assert(f(0x58D3, 0xEA95) == 0xB246);
    assert(f(0x58D3L, 0xEA95) == 0xB246);
    assert(f(0x58D3, 0xEA95L) == 0xB246);

    assert(f(0x58D3, 0) == 0x58D3);
    assert(f(0x58D3L, 0) == 0x58D3);
    assert(f(0x58D3, 0L) == 0x58D3);

    assert(f(0xFFFF, 0x58D3) == 0xA72C);
    assert(f(0xFFFFL, 0x58D3) == 0xA72C);
    assert(f(0xFFFF, 0x58D3L) == 0xA72C);
#if TEST_STD_VER > 2011
    constexpr int foo = cuda::std::bit_xor<int>()(0x58D3, 0xEA95);
    static_assert(foo == 0xB246, "");

    constexpr int bar = cuda::std::bit_xor<>()(0x58D3L, 0xEA95);
    static_assert(bar == 0xB246, "");
#endif
  }

  return 0;
}
