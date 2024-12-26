//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Illustrate how to use the void data interface
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  auto token = ctx.logical_data(shape_of<void_interface>());
  ctx.task(token.write())->*[](cudaStream_t, auto) {

  };

  void_interface sync;
  auto token2 = ctx.logical_data(sync);

  auto token3 = ctx.logical_token();
  ctx.task(token2.write(), token.read())->*[](cudaStream_t, auto, auto) {

  };

  // Do not pass useless arguments by removing void_interface arguments
  // Note that the rw() access is possible even if there was no prior write()
  // or actual underlying data.
  ctx.task(token3.rw(), token.read())->*[](cudaStream_t) {

  };

  ctx.finalize();
}
