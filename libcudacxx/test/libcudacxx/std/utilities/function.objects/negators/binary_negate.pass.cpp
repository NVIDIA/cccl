//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// binary_negate

#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_BINDER_TYPEDEFS
#define _LIBCUDACXX_ENABLE_CXX20_REMOVED_NEGATORS
// ADDITIONAL_COMPILE_DEFINITIONS: _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

// ensure that we allow `__device__` functions too
struct with_device_op
{
  using first_argument_type  = int;
  using second_argument_type = int;
  using result_type          = bool;
  __device__ constexpr bool operator()(const int& lhs, const int& rhs) const
  {
    return lhs && rhs;
  }
};

__global__ void test_global_kernel()
{
  const cuda::std::binary_negate<with_device_op> f{with_device_op{}};
  assert(!f(36, 36));
}

int main(int, char**)
{
  typedef cuda::std::binary_negate<cuda::std::logical_and<int>> F;
  const F f = F(cuda::std::logical_and<int>());
#if TEST_STD_VER <= 2017
  static_assert((cuda::std::is_same<int, F::first_argument_type>::value), "");
  static_assert((cuda::std::is_same<int, F::second_argument_type>::value), "");
  static_assert((cuda::std::is_same<bool, F::result_type>::value), "");
#endif // TEST_STD_VER <= 2017
  assert(!f(36, 36));
  assert(f(36, 0));
  assert(f(0, 36));
  assert(f(0, 0));

  return 0;
}
