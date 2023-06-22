//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/version>

#include <cuda/std/version>

#include "test_macros.h"

#if !defined(_LIBCUDACXX_VERSION)
#error "_LIBCUDACXX_VERSION must be defined after including <cuda/std/version>"
#endif

int main(int, char**)
{

  return 0;
}
