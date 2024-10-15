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

  auto ltask_res = ctx.logical_data(shape_of<void_interface>());
  ctx.task(ltask_res.write())->*[](cudaStream_t, auto) {

  };

  void_interface sync;
  auto ltask2_res = ctx.logical_data(sync);
  ctx.task(ltask2_res.write(), ltask_res.read())->*[](cudaStream_t, auto, auto) {

  };

  ctx.finalize();
}
