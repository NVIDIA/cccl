//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Static error check: `subst(1) & noop` is rejected like any `subst(1) & s`.
 *
 * the identity `s & noop = s` is absorbed only after the checks on `s` run.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::sump;

int main()
{
  errsink(subst(1) \& noop) ->* [] { return 0; };
  return EXIT_FAILURE;
}
