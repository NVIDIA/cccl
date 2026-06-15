//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

// fetch_min is only provided for integral and pointer atomics, not floating-point.

#include <cuda/std/atomic>

int main(int, char**)
{
  cuda::std::atomic<float> a(0.0f);
  a.fetch_min(0.0f); // expected-error: no member named 'fetch_min'
  return 0;
}
