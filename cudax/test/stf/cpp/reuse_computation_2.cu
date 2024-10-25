//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/utility/run_once.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
  context ctx;

  const int N  = 16;
  size_t niter = 12;

  int A[N];

  for (int i = 0; i < N; i++)
  {
    A[i] = 2 * i + 1;
  }

  auto lres = ctx.logical_data(A);

  for (size_t k = 0; k < niter; k++)
  {
    auto ltmp = ctx.logical_data(lres.shape());
    ctx.parallel_for(ltmp.shape(), ltmp.write())->*[k] __device__(size_t i, auto tmp) {
      tmp(i) = (k % 2) * i;
    };

    ctx.parallel_for(lres.shape(), ltmp.read(), lres.rw())->*[] __device__(size_t i, auto tmp, auto res) {
      res(i) += tmp(i);
    };
  }

  for (size_t k = 0; k < niter; k++)
  {
    auto ltmp = run_once(k)->*[&](size_t k) {
      auto out = ctx.logical_data(lres.shape());
      ctx.parallel_for(out.shape(), out.write())->*[k] __device__(size_t i, auto tmp) {
        tmp(i) = (k % 2) * i;
      };
      return out;
    };

    ctx.parallel_for(lres.shape(), ltmp.read(), lres.rw())->*[] __device__(size_t i, auto tmp, auto res) {
      res(i) += tmp(i);
    };
  }

  ctx.finalize();

  for (int i = 0; i < N; i++)
  {
    EXPECT(A[i] == (2 * i + 1) + 2 * i * niter / 2);
  }
}
