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
 * @brief Static error check: exec_place_scope should not accept data_place
 */

#include <cuda/experimental/__places/places.cuh>

using namespace cuda::experimental::places;

int main()
{
  // expected-error {{"exec_place_scope cannot be constructed from data_place; use data_place::affine_exec_place() to
  // get the exec_place first"}}
  exec_place_scope scope(data_place::device(0));
  return EXIT_FAILURE;
}
