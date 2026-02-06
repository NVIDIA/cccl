//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_multiply

#include <cuda/std/ratio>

int main(int, char**)
{
  {
    using R1 = cuda::std::ratio<1, 1>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_multiply<R1, R2>::type;
    static_assert(R::num == 1 && R::den == 1, "");
  }
  {
    using R1 = cuda::std::ratio<1, 2>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_multiply<R1, R2>::type;
    static_assert(R::num == 1 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<-1, 2>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_multiply<R1, R2>::type;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<1, -2>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_multiply<R1, R2>::type;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<1, 2>;
    using R2 = cuda::std::ratio<-1, 1>;
    using R  = cuda::std::ratio_multiply<R1, R2>::type;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<1, 2>;
    using R2 = cuda::std::ratio<1, -1>;
    using R  = cuda::std::ratio_multiply<R1, R2>::type;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<56987354, 467584654>;
    using R2 = cuda::std::ratio<544668, 22145>;
    using R  = cuda::std::ratio_multiply<R1, R2>::type;
    static_assert(R::num == 15519594064236LL && R::den == 5177331081415LL, "");
  }

  return 0;
}
