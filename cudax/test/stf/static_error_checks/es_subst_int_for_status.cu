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
 * @brief Static error check: `subst` on a status operand must substitute a value of the status type.
 *
 * An `int` is not a `cudaError_t`; `subst(cudaSuccess)` is the spelling that works.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::exception_policies;

int main()
{
  on_error(subst(0))->*cudaErrorInvalidValue;
  return EXIT_FAILURE;
}
