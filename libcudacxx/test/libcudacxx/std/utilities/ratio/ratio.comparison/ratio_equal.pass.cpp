//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_equal

#include <cuda/std/ratio>

#include "test_macros.h"

template <class Rat1, class Rat2, bool result>
__host__ __device__ void test()
{
  static_assert((result == cuda::std::ratio_equal<Rat1, Rat2>::value), "");
#if TEST_STD_VER > 2014
  static_assert((result == cuda::std::ratio_equal_v<Rat1, Rat2>), "");
#endif
}

int main(int, char**)
{
  {
    typedef cuda::std::ratio<1, 1> R1;
    typedef cuda::std::ratio<1, 1> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 1> R2;
    test<R1, R2, true>();
  }
  {
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, true>();
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
    test<R1, R2, false>();
  }
  {
    typedef cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL> R1;
    typedef cuda::std::ratio<1, -0x7FFFFFFFFFFFFFFFLL> R2;
    test<R1, R2, false>();
  }

  return 0;
}
