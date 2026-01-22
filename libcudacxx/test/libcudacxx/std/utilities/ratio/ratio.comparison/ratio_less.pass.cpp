//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_less

#include <cuda/std/ratio>

#include "test_macros.h"

template <class Rat1, class Rat2, bool result>
__host__ __device__ void test()
{
  static_assert((result == cuda::std::ratio_less<Rat1, Rat2>::value), "");
  static_assert((result == cuda::std::ratio_less_v<Rat1, Rat2>), "");
}

int main(int, char**)
{
  {
    using R1 = cuda::std::ratio<1, 1>;
    using R2 = cuda::std::ratio<1, 1>;
    test<R1, R2, false>();
  }
  {
    using R1 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1>;
    using R2 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1>;
    test<R1, R2, false>();
  }
  {
    using R1 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1>;
    using R2 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1>;
    test<R1, R2, false>();
  }
  {
    using R1 = cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL>;
    using R2 = cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL>;
    test<R1, R2, false>();
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
  {
    using R1 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL>;
    using R2 = cuda::std::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL>;
    using R2 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL>;
    test<R1, R2, false>();
  }
  {
    using R1 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL>;
    using R2 = cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL>;
    using R2 = cuda::std::ratio<0x7FFFFFFFFFFFFFFELL, 0x7FFFFFFFFFFFFFFDLL>;
    test<R1, R2, true>();
  }
  {
    using R1 = cuda::std::ratio<641981, 1339063>;
    using R2 = cuda::std::ratio<1291640, 2694141LL>;
    test<R1, R2, false>();
  }
  {
    using R1 = cuda::std::ratio<1291640, 2694141LL>;
    using R2 = cuda::std::ratio<641981, 1339063>;
    test<R1, R2, true>();
  }

  return 0;
}
