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
  context ctx;

  int nqpoints = 3;
  auto ltoken  = ctx.token();

  ctx.parallel_for(exec_place::host(), box(5), ltoken.read())->*[nqpoints] __host__(size_t) {
    _CCCL_ASSERT(nqpoints == 3, "invalid value");
  };

  ctx.finalize();
}
