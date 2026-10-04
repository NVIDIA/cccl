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
 * @brief Static error check: `when_equal` must match a status of the values' own type.
 *
 * The values are `cudaError_t`; the operand is a `CUresult`.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::eh;

int main()
{
  errsink(when_equal(cudaErrorNotReady)(::std::ignore))->*CUDA_ERROR_NOT_READY;
  return EXIT_FAILURE;
}
