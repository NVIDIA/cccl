//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_subtract

// XFAIL: (nvcc-10 || nvcc-11.0 || nvcc-11.1 || nvcc-11.2) && !nvrtc

// NVCC emits completely nonsensical code for the host compiler for this test
// because some of the specializations of ratio appear multiple times, whether
// as arguments or as results of the calculations. expect this to fail until
// the compiler fixes it.

#include <cuda/std/ratio>

int main(int, char**)
{
  {
    using R1 = cuda::std::ratio<1, 1>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == 0 && R::den == 1, "");
  }
  {
    using R1 = cuda::std::ratio<1, 2>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == -1 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<-1, 2>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == -3 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<1, -2>;
    using R2 = cuda::std::ratio<1, 1>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == -3 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<1, 2>;
    using R2 = cuda::std::ratio<-1, 1>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == 3 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<1, 2>;
    using R2 = cuda::std::ratio<1, -1>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == 3 && R::den == 2, "");
  }
  {
    using R1 = cuda::std::ratio<56987354, 467584654>;
    using R2 = cuda::std::ratio<544668, 22145>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == -126708206685271LL && R::den == 5177331081415LL, "");
  }
  {
    using R1 = cuda::std::ratio<0>;
    using R2 = cuda::std::ratio<0>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == 0 && R::den == 1, "");
  }
  {
    using R1 = cuda::std::ratio<1>;
    using R2 = cuda::std::ratio<0>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == 1 && R::den == 1, "");
  }
  {
    using R1 = cuda::std::ratio<0>;
    using R2 = cuda::std::ratio<1>;
    using R  = cuda::std::ratio_subtract<R1, R2>::type;
    static_assert(R::num == -1 && R::den == 1, "");
  }

  return 0;
}
