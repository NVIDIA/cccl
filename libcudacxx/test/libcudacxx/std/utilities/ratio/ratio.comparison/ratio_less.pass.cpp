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
TEST_FUNC constexpr void test()
{
  static_assert((result == cuda::std::ratio_less<Rat1, Rat2>::value));
  static_assert((result == cuda::std::ratio_less_v<Rat1, Rat2>) );
}

TEST_FUNC constexpr bool test()
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
  {
    using R1 = cuda::std::ratio<0>;
    using R2 = cuda::std::ratio<1, 0x7FFFFFFFFFFFFFFFLL>;
    test<R1, R2, true>();
    test<R2, R1, false>();
  }
  {
    using R1 = cuda::std::ratio<-1, 0x7FFFFFFFFFFFFFFFLL>;
    using R2 = cuda::std::ratio<0>;
    test<R1, R2, true>();
    test<R2, R1, false>();
  }
  {
    using R1 = cuda::std::ratio<0>;
    using R2 = cuda::std::ratio<0, 5>;
    test<R1, R2, false>();
  }
  {
    // Ratios of consecutive Fibonacci numbers have the longest continued fraction expansions
    using R1 = cuda::std::ratio<7540113804746346429LL, 4660046610375530309LL>;
    using R2 = cuda::std::ratio<4660046610375530309LL, 2880067194370816120LL>;
    test<R1, R2, true>();
    test<R2, R1, false>();
  }
  {
    using R1 = cuda::std::ratio<-7540113804746346429LL, 4660046610375530309LL>;
    using R2 = cuda::std::ratio<-4660046610375530309LL, 2880067194370816120LL>;
    test<R1, R2, false>();
    test<R2, R1, true>();
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
