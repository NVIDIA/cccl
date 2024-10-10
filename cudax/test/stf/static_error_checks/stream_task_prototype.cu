//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;

  auto A = ctx.logical_data<int>(size_t(1));
  auto B = ctx.logical_data<int>(size_t(1));

  // Task with an invalid prototype
  ctx.task(A.rw(), B.rw())->*[](cudaStream_t s, auto A) {

  };

  ctx.finalize();
}
