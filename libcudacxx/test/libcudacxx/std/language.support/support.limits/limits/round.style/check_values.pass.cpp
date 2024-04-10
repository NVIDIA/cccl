//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test numeric_limits

// float_denorm_style

#include <cuda/std/limits>

#include "test_macros.h"

typedef char one;
struct two
{
  one _[2];
};

__host__ __device__ one test(cuda::std::float_denorm_style);
__host__ __device__ two test(int);

int main(int, char**)
{
  static_assert(cuda::std::denorm_indeterminate == -1, "cuda::std::denorm_indeterminate == -1");
  static_assert(cuda::std::denorm_absent == 0, "cuda::std::denorm_absent == 0");
  static_assert(cuda::std::denorm_present == 1, "cuda::std::denorm_present == 1");
  static_assert(sizeof(test(cuda::std::denorm_present)) == 1, "sizeof(test(cuda::std::denorm_present)) == 1");
  static_assert(sizeof(test(1)) == 2, "sizeof(test(1)) == 2");

  return 0;
}
