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

  auto A = ctx.logical_data<int>(size_t(1));

  // We cannot do a parallel for on a device lambda without a CUDA compiler
  // as we normally expect an extended lambda (which is also missing here)
  ctx.parallel_for(A.shape(), A.write())->*[](size_t i, auto A) {
    A(i) = 0;
  };

  ctx.finalize();
}
