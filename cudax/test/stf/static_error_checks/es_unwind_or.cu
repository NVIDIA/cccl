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
 * @brief Static error check: nothing may follow `unwind` in a `|` alternation.
 *
 * `unwind` always diverges, so the right arm can never run.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::sump;

int main()
{
  errsink(unwind | subst(0))->*[] {
    return 0;
  };
  return EXIT_FAILURE;
}
