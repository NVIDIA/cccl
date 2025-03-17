//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// conditional

#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_same_v<char, cuda::std::conditional<true, char, int>::type>);
  static_assert(cuda::std::is_same_v<int, cuda::std::conditional<false, char, int>::type>);
  static_assert(cuda::std::is_same_v<char, cuda::std::conditional_t<true, char, int>>);
  static_assert(cuda::std::is_same_v<int, cuda::std::conditional_t<false, char, int>>);

  return 0;
}
