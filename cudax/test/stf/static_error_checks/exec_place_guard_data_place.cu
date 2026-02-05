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
 * @brief Static error check: exec_place_guard should not accept data_place
 */

#include <cuda/experimental/__stf/places/places.cuh>

using namespace cuda::experimental::stf;

int main()
{
  // This should fail to compile: exec_place_guard only accepts exec_place, not data_place
  exec_place_guard guard(data_place::device(0));
  return EXIT_FAILURE;
}
