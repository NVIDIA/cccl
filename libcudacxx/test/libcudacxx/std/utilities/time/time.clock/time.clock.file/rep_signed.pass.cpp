//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14, c++17
// XFAIL: dylib-has-no-filesystem

// File clock is unsupported in NVRTC
// UNSUPPORTED: nvrtc

// <cuda/std/chrono>

// file_clock

// rep should be signed

#include <cuda/std/cassert>
#include <cuda/std/chrono>

int main(int, char**)
{
  static_assert(cuda::std::is_signed<cuda::std::chrono::file_clock::rep>::value, "");
  assert(cuda::std::chrono::file_clock::duration::min() < cuda::std::chrono::file_clock::duration::zero());

  return 0;
}
