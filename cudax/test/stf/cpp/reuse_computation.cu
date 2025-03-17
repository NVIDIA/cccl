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
    ctx.parallel_for(ltmp.shape(), ltmp.write())->*[] __device__(size_t i, auto tmp) {
      tmp(i) = i;
    };

    ctx.parallel_for(lres.shape(), ltmp.read(), lres.rw())->*[] __device__(size_t i, auto tmp, auto res) {
      res(i) += tmp(i);
    };
  }

  for (size_t k = 0; k < niter; k++)
  {
    auto ltmp = run_once()->*[&]() {
      // Ensure this is only done once !
      static bool done = false;
      EXPECT(!done);
      done = true;

      auto ltmp = ctx.logical_data(lres.shape());
      ctx.parallel_for(ltmp.shape(), ltmp.write())->*[] __device__(size_t i, auto tmp) {
        tmp(i) = i;
      };
      return ltmp;
    };

    auto ltmp2 = run_once(size_t(k % 4))->*[&](size_t val) {
      // fprintf(stderr, "COMPUTE FOR %ld\n", val);

      auto ltmp = ctx.logical_data(lres.shape());
      ctx.parallel_for(ltmp.shape(), ltmp.write())->*[val] __device__(size_t i, auto tmp) {
        tmp(i) = val;
      };
      return ltmp;
    };

    ctx.parallel_for(lres.shape(), ltmp.read(), lres.rw())->*[] __device__(size_t i, auto tmp, auto res) {
      res(i) += tmp(i);
    };
  }

  ctx.finalize();

  for (int i = 0; i < N; i++)
  {
    EXPECT(A[i] == (2 * i + 1) + 2 * i * niter);
  }
}
