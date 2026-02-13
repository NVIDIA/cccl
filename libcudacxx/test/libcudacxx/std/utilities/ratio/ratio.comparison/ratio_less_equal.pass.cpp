//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_less_equal

#include <cuda/std/ratio>

#include "test_macros.h"

template <class Rat1, class Rat2, bool result>
__host__ __device__ void test()
{
  static_assert((result == cuda::std::ratio_less_equal<Rat1, Rat2>::value), "");
  static_assert((result == cuda::std::ratio_less_equal_v<Rat1, Rat2>), "");
}

int main(int, char**)
{
  {
    using R1 = cuda::std::ratio<1, 1>;
    using R2 = cuda::std::ratio<1, 1>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1>;
    using R2 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1>;
    using R2 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL>;
    using R2 = cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<1, 1>;
    using R2 = cuda::std::ratio<1, -1>;
    test<R1, R2, false>();
  }
  {
    using R1 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1>;
    using R2 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1>;
    test<R1, R2, false>();
  }
  {
    using R1 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1>;
    using R2 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL>;
    using R2 = cuda::std::ratio<1, -0x7FFFFFFFFFFFFFFFLL>;
    test<R1, R2, false>();
  }

  return 0;
}
