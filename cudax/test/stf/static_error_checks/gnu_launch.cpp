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

  auto A = ctx.logical_data<int>(size_t(128));

  // We cannot do a launch on a device lambda without a CUDA compiler
  // as we normally expect an extended lambda (which is also missing here)
  ctx.launch(par(128), A.write())->*[](auto th, auto A) { A(th.rank(i) = 0;
  };

  ctx.finalize();
}
