//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// less

// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"
#if !TEST_COMPILER(NVRTC)
#  include "pointer_comparison_test_helper.hpp"
#endif // !TEST_COMPILER(NVRTC)

// ensure that we allow `__device__` functions too
struct with_device_op
{
  __device__ friend constexpr bool operator<(const with_device_op&, const with_device_op&)
  {
    return true;
  }
};

__global__ void test_global_kernel()
{
  const cuda::std::less<with_device_op> f;
  assert(f({}, {}));
}

int main(int, char**)
{
  typedef cuda::std::less<int> F;
  const F f = F();
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
  static_assert((cuda::std::is_same<bool, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(!f(36, 36));
  assert(!f(36, 6));
  assert(f(6, 36));
  NV_IF_TARGET(NV_IS_HOST,
               (
                 // test total ordering of int* for less<int*> and less<void>.
                 do_pointer_comparison_test<int, cuda::std::less>();))

  typedef cuda::std::less<> F2;
  const F2 f2 = F2();
  assert(!f2(36, 36));
  assert(!f2(36, 6));
  assert(f2(6, 36));
  assert(!f2(36, 6.0));
  assert(!f2(36.0, 6));
  assert(f2(6, 36.0));
  assert(f2(6.0, 36));
  constexpr bool foo = cuda::std::less<int>()(36, 36);
  static_assert(!foo, "");

  constexpr bool bar = cuda::std::less<>()(36.0, 36);
  static_assert(!bar, "");

  return 0;
}
