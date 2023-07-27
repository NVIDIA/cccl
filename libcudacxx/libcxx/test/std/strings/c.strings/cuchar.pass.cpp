//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// XFAIL: *

// If we're just building the test and not executing it, it should pass.
// UNSUPPORTED: no_execute

// <cuchar>

#include <cuchar>

#include "test_macros.h"

int main(int, char**)
{

  return 0;
}
