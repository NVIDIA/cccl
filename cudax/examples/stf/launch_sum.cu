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
 * @brief A reduction kernel written using launch
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

int main()
{
  context ctx;

  const size_t N = 128 * 1024 * 1024;

  std::vector<double> X(N);
  double sum = 0.0;

  double ref_sum = 0.0;

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = sin((double) ind);
    ref_sum += X[ind];
  }

  auto lX   = ctx.logical_data(&X[0], {N});
  auto lsum = ctx.logical_data(&sum, {1});

  auto number_devices = 1; //
  auto where          = exec_place::repeat(exec_place::device(0), number_devices);

  auto spec = par<16>(con<32>());

  ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
    // Each thread computes the sum of elements assigned to it
    double local_sum = 0.0;
    for (size_t i = th.rank(); i < x.size(); i += th.size())
    {
      local_sum += x(i);
    }

    auto ti = th.inner();

    __shared__ double block_sum[th.static_width(1)];
    block_sum[ti.rank()] = local_sum;

    for (size_t s = ti.size() / 2; s > 0; s /= 2)
    {
      ti.sync();
      if (ti.rank() < s)
      {
        block_sum[ti.rank()] += block_sum[ti.rank() + s];
      }
    }

    if (ti.rank() == 0)
    {
      atomicAdd(&sum(0), block_sum[0]);
    }
  };

  ctx.finalize();

  EXPECT(fabs(sum - ref_sum) < 0.0001);
}
