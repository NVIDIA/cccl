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
 * @brief Static error check: a type selector must not be applied to a status operand.
 *
 * A past result cannot throw, so `if_isa` could never match.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::sump;

int main()
{
  errsink(if_isa<::std::bad_alloc>(noop))->*cudaErrorInvalidValue;
  return EXIT_FAILURE;
}
