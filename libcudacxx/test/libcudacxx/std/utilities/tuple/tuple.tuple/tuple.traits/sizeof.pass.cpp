//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(sizeof(cuda::std::tuple<float>) == sizeof(float));
  static_assert(sizeof(cuda::std::tuple<cuda::std::tuple<float>, cuda::std::tuple<float>>)
                == sizeof(cuda::std::tuple<float, float>));
  static_assert(sizeof(cuda::std::tuple<cuda::std::tuple<float>, cuda::std::tuple<float>>) == sizeof(float) * 2);

  return 0;
}
