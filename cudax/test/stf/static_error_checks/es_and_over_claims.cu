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
 * @brief Static error check: `&` treats only the channels both of its sides treat.
 *
 * On an exception, `when_equal` forwards and `&` stops there, so `thrown(...)` never runs:
 * the composite has no treatment for exceptions although the callable can throw.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;
using namespace cuda::experimental::stf::eh;

void may_throw();

int main()
{
  errsink(when_equal(cudaErrorNotReady)(::std::ignore) & thrown(::std::ignore)) << [&] {
    may_throw();
    return cudaErrorNotReady;
  };
  return EXIT_FAILURE;
}
