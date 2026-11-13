//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_multiply

#include <cuda/std/ratio>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class R1, class R2, long long N, long long D>
TEST_FUNC constexpr void test()
{
  using R = cuda::std::ratio_multiply<R1, R2>;
  static_assert(cuda::std::is_same_v<R, cuda::std::ratio<N, D>>);
  static_assert(R::num == N && R::den == D);
}

TEST_FUNC constexpr bool test()
{
  test<cuda::std::ratio<1, 1>, cuda::std::ratio<1, 1>, 1, 1>();
  test<cuda::std::ratio<1, 2>, cuda::std::ratio<1, 1>, 1, 2>();
  test<cuda::std::ratio<-1, 2>, cuda::std::ratio<1, 1>, -1, 2>();
  test<cuda::std::ratio<1, -2>, cuda::std::ratio<1, 1>, -1, 2>();
  test<cuda::std::ratio<1, 2>, cuda::std::ratio<-1, 1>, -1, 2>();
  test<cuda::std::ratio<1, 2>, cuda::std::ratio<1, -1>, -1, 2>();
  test<cuda::std::ratio<56987354, 467584654>, cuda::std::ratio<544668, 22145>, 15519594064236LL, 5177331081415LL>();
  test<cuda::std::ratio<0>, cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL>, 0, 1>();
  test<cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 2>, cuda::std::ratio<2, 0x7FFFFFFFFFFFFFFFLL>, 1, 1>();
  test<cuda::std::ratio<-0x7FFFFFFFFFFFFFFFLL, 3>, cuda::std::ratio<3, 0x7FFFFFFFFFFFFFFFLL>, -1, 1>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
