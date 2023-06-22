//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio_multiply

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

#include <cuda/std/ratio>

int main(int, char**)
{
    typedef cuda::std::ratio<0x7FFFFFFFFFFFFFFFLL, 1> R1;
    typedef cuda::std::ratio<2, 1> R2;
    typedef cuda::std::ratio_multiply<R1, R2>::type R;

  return 0;
}
