//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_divide

#include <cuda/std/ratio>

int main(int, char**)
{
  using R1 = cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1>;
  using R2 = cuda::std::ratio<1, 2>;
  using R  = cuda::std::ratio_divide<R1, R2>::type;

  return 0;
}
