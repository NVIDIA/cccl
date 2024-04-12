//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// template <class Rep2>
//   explicit duration(const Rep2& r);

#include "../../rep.h"

#include <cuda/std/cassert>
#include <cuda/std/chrono>

#include "test_macros.h"

template <class D, class R>
__host__ __device__ void test(R r)
{
  D d(r);
  assert(d.count() == r);
  constexpr D d2(R(2));
  static_assert(d2.count() == 2, "");
}

int main(int, char**)
{
  test<cuda::std::chrono::duration<int>>(5);
  test<cuda::std::chrono::duration<int, cuda::std::ratio<3, 2>>>(5);
  test<cuda::std::chrono::duration<Rep, cuda::std::ratio<3, 2>>>(Rep(3));
  test<cuda::std::chrono::duration<double, cuda::std::ratio<2, 3>>>(5.5);

  return 0;
}
