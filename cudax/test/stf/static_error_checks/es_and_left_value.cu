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
 * @brief Static error check: the left side of `&` may not choose a value.
 *
 * `subst(1) & notify` would choose 1 and then let `notify` override it with the default; the
 * value-producing policy belongs on the right (`notify & subst(1)`), or in a `|` fallback.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  const int v = errsink(eh::subst(1) & eh::notify)->*[]() -> int {
    throw ::std::runtime_error("x");
  };
  return v;
}
