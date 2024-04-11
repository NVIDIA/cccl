//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>
// UNSUPPORTED: c++20

// not1

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/cassert>
#include <cuda/std/functional>

int main(int, char**)
{
  typedef cuda::std::logical_not<int> F;
  assert(cuda::std::not1(F())(36));
  assert(!cuda::std::not1(F())(0));

  return 0;
}
