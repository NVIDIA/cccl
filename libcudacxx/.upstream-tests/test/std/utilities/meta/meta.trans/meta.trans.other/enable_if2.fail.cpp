//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// .fail. expects compilation to fail, but this would only fail at runtime with NVRTC
// UNSUPPORTED: nvrtc

// UNSUPPORTED: c++98, c++03, c++11
// type_traits

// enable_if

#include <cuda/std/type_traits>

int main(int, char**)
{
    typedef cuda::std::enable_if_t<false> A;

  return 0;
}
