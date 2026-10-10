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
 * @brief Static error check: cuda_try must reject calls where both the first-
 *        and last-output-parameter forms apply for the supplied user arguments.
 *
 * `cudaMemGetInfo(size_t* free, size_t* total)` has two non-const pointer
 * parameters. Supplying a single `size_t*` is consistent with synthesizing
 * either the first or the last output, so cuda_try cannot disambiguate.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  size_t user_size = 0;
  cuda_try<cudaMemGetInfo>(&user_size);
  return EXIT_FAILURE;
}
