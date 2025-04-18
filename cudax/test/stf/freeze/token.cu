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
 * @brief Freeze token
 *
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  auto ltoken = ctx.token();

  auto ftoken = ctx.freeze(ltoken);

  cudaStream_t stream = ctx.pick_stream();
  // This makes any future operations in this CUDA stream depend on the
  // availability of the token
  [[maybe_unused]] auto dtoken = ftoken.get(data_place::current_device(), stream);
  ftoken.unfreeze(stream);

  ctx.finalize();
}
