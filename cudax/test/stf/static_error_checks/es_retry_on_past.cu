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
 * @brief Static error check: `retry` must reject a status operand.
 *
 * A status is a past result: there is no action left to repeat, so `retry` on one
 * is rejected instead of quietly yielding the failing status again.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::exception_policies;

int main()
{
  on_error(retry)->*cudaErrorInvalidValue;
  return EXIT_FAILURE;
}
