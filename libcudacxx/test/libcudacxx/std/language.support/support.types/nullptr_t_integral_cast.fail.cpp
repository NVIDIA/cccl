//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// typedef decltype(nullptr) nullptr_t;

#include <cuda/std/cstddef>

int main(int, char**)
{
  cuda::std::ptrdiff_t i = static_cast<cuda::std::ptrdiff_t>(nullptr);

  return 0;
}
