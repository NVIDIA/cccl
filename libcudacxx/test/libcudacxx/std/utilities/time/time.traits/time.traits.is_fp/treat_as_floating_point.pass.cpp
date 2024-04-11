//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// treat_as_floating_point

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

template <class T>
__host__ __device__ void test()
{
  static_assert(
    (cuda::std::is_base_of<cuda::std::is_floating_point<T>, cuda::std::chrono::treat_as_floating_point<T>>::value), "");
#if TEST_STD_VER > 2014
  static_assert(cuda::std::is_floating_point<T>::value == cuda::std::chrono::treat_as_floating_point_v<T>, "");
#endif
}

struct A
{};

int main(int, char**)
{
  test<int>();
  test<unsigned>();
  test<char>();
  test<bool>();
  test<float>();
  test<double>();
  test<long double>();
  test<A>();

  return 0;
}
