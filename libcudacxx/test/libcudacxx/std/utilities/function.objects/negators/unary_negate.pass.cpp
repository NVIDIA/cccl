//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// unary_negate

#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS
#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_NEGATORS
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

// ensure that we allow `__device__` functions too
struct with_device_op
{
  using argument_type = int;
  using result_type   = bool;
  __device__ constexpr bool operator()(const int&) const
  {
    return true;
  }
};

__global__ void test_global_kernel()
{
  const cuda::std::unary_negate<with_device_op> f{with_device_op{}};
  assert(!f(36));
}

int main(int, char**)
{
  typedef cuda::std::unary_negate<cuda::std::logical_not<int>> F;
  const F f = F(cuda::std::logical_not<int>());
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<F::argument_type, int>::value), "");
  static_assert((cuda::std::is_same<F::result_type, bool>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(f(36));
  assert(!f(0));

  return 0;
}
