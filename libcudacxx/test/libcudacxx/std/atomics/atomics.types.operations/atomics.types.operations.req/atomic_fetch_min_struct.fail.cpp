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

// fetch_min is only provided for integral and pointer atomics, not user-defined types.

#include <cuda/std/atomic>

struct UserType
{
  int value;
};

int main(int, char**)
{
  cuda::std::atomic<UserType> a(UserType{0});
  a.fetch_min(UserType{0}); // expected-error: no member named 'fetch_min'
  return 0;
}
