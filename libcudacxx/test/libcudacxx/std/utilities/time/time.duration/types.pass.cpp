//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// duration

// Test nested types

// typedef Rep rep;
// typedef Period period;

#include <cuda/std/chrono>
#include <cuda/std/type_traits>

int main(int, char**)
{
  typedef cuda::std::chrono::duration<long, cuda::std::ratio<3, 2>> D;
  static_assert((cuda::std::is_same<D::rep, long>::value), "");
  static_assert((cuda::std::is_same<D::period, cuda::std::ratio<3, 2>>::value), "");

  return 0;
}
