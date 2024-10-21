//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <iostream>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;

  // Contiguous 1D
  double* X     = new double[1024];
  auto handle_X = ctx.logical_data(make_slice(X, 1024));

  // Contiguous 2D
  double* X2     = new double[1024 * 1024];
  auto handle_X2 = ctx.logical_data(make_slice(X2, std::tuple{1024, 1024}, 1024));

  // Contiguous 3D
  double* X4     = new double[128 * 128 * 128];
  auto handle_X4 = ctx.logical_data(make_slice(X4, std::tuple{128, 128, 128}, 128, 128 * 128));

  // Discontiguous 2D
  double* X3     = new double[128 * 8];
  auto handle_X3 = ctx.logical_data(make_slice(X3, std::tuple{64, 8}, 128));

  // Discontiguous 3D
  double* X5     = new double[32 * 4 * 4];
  auto handle_X5 = ctx.logical_data(make_slice(X5, std::tuple{16, 4, 4}, 32, 32 * 4));

  double* X6     = new double[32 * 4 * 4];
  auto handle_X6 = ctx.logical_data(make_slice(X6, std::tuple{32, 2, 4}, 32, 32 * 4));

  double* X7 = new double[128 * 128 * 128];
  cuda_safe_call(cudaHostRegister(X7, 128 * 128 * 128, cudaHostRegisterPortable));
  auto handle_X7 = ctx.logical_data(make_slice(X7, std::tuple{128, 128, 128}, 128, 128 * 128));

  // Detect that this was already pinned
  double* X9 = new double[1024];
  cuda_safe_call(cudaHostRegister(X9, 1024, cudaHostRegisterPortable));
  auto handle_X9 = ctx.logical_data(make_slice(X9, 1024));

  // Detect that this was already pinned
  double* X8 = new double[4 * 4 * 4];
  cuda_safe_call(cudaHostRegister(X8, 4 * 4 * 4, cudaHostRegisterPortable));
  auto handle_X8 = ctx.logical_data(make_slice(X8, std::tuple{1, 4, 4}, 4, 4 * 4));

  ctx.finalize();
}
