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
#if TEST_STD_VER > 2014
  static_assert((result == cuda::std::ratio_less_v<Rat1, Rat2>), "");
#endif
}

int main(int, char**)
{
  {
    typedef cuda::std::ratio<1, 1> R1;
    typedef cuda::std::ratio<1, 1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<1, 1> R1;
    typedef cuda::std::ratio<1, -1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef cuda::std::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFDLL, 0x7FFFFFFFFFFFFFFCLL> R1;
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 0x7FFFFFFFFFFFFFFELL> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFELL, 0x7FFFFFFFFFFFFFFDLL> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<641981, 1339063> R1;
    typedef cuda::std::ratio<1291640, 2694141LL> R2;
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<1291640, 2694141LL> R1;
    typedef cuda::std::ratio<641981, 1339063> R2;
    test<R1, R2, true>();
  }

  return 0;
}
