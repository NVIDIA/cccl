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
 * @brief Static error check: the policy must cover every kind of failure the operand can produce.
 *
 * The callable is not `noexcept`, so it can throw, but `when_equal` only treats failing statuses.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::exception_policies;

void may_throw();

int main()
{
  on_error(when_equal(cudaErrorNotReady)(::std::ignore))->*[] {
    may_throw();
    return cudaSuccess;
  };
  return EXIT_FAILURE;
}
