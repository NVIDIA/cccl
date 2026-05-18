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

// denorm_style has been deprecated since C++23
#if _CCCL_STD_VER >= 2023
_CCCL_SUPPRESS_DEPRECATED_PUSH
#endif // _CCCL_STD_VER >= 2023

using one = char;
struct two
{
  one _[2];
};

TEST_FUNC one test(cuda::std::float_denorm_style);
TEST_FUNC two test(int);

int main(int, char**)
{
  static_assert(cuda::std::denorm_indeterminate == -1, "cuda::std::denorm_indeterminate == -1");
  static_assert(cuda::std::denorm_absent == 0, "cuda::std::denorm_absent == 0");
  static_assert(cuda::std::denorm_present == 1, "cuda::std::denorm_present == 1");
  static_assert(sizeof(test(cuda::std::denorm_present)) == 1, "sizeof(test(cuda::std::denorm_present)) == 1");
  static_assert(sizeof(test(1)) == 2, "sizeof(test(1)) == 2");

  return 0;
}
