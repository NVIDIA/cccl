//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test ratio:  The template argument D shall not be zero

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

#include <cuda/std/ratio>
#include <cuda/std/cstdint>

int main(int, char**)
{
    const cuda::std::intmax_t t1 = cuda::std::ratio<1, 0>::num;

  return 0;
}
