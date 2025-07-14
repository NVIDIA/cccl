//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(sizeof(cuda::std::pair<float, int>) == sizeof(float) + sizeof(int));
  static_assert(sizeof(cuda::std::pair<cuda::std::pair<float, int>, cuda::std::pair<float, int>>)
                == 2 * sizeof(cuda::std::pair<float, int>));
  static_assert(sizeof(cuda::std::pair<cuda::std::pair<float, int>, cuda::std::pair<float, int>>)
                == sizeof(float) * 2 + sizeof(int) * 2);

  return 0;
}
