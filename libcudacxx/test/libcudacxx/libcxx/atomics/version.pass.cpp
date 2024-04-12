//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads, pre-sm-60
// UNSUPPORTED: windows && pre-sm-70

// <cuda/std/atomic>

#include <cuda/std/atomic>

#include "test_macros.h"

#ifndef _LIBCUDACXX_VERSION
#  error _LIBCUDACXX_VERSION not defined
#endif

int main(int, char**)
{
  return 0;
}
