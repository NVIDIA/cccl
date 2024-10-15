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
 * @brief A reduction kernel written using launch and CUB
 */

#include <cub/cub.cuh>

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

  auto number_devices = 2;
  auto where          = exec_place::repeat(exec_place::device(0), number_devices);

  auto spec = par<32>(con<128>());

  ctx.launch(spec, where, lX.read(), lsum.rw())->*[] _CCCL_DEVICE(auto th, auto x, auto sum) {
    // Each thread computes the sum of elements assigned to it
    double local_sum = 0.0;
    for (auto ind : th.apply_partition(shape(x)))
    {
      local_sum += x(ind);
    }

    using BlockReduce = cub::BlockReduce<double, th.static_width(1)>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (th.inner().rank() == 0)
    {
      atomicAdd(&sum(0), block_sum);
    }
  };

  ctx.finalize();

  EXPECT(fabs(sum - ref_sum) < 0.0001);
}
