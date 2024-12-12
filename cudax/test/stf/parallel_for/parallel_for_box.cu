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
#include <cuda/experimental/__stf/utility/dimensions.cuh>

using namespace cuda::experimental::stf;

int main()
{
  stream_ctx ctx;

  const size_t N = 16;
  const size_t M = 16;

  /*
   * First test an explicit shape by only specifying its size
   */

  std::vector<double> A(M * N);
  auto lA = ctx.logical_data(make_slice(&A[0], std::tuple<size_t, size_t>{M, N}, M));

  ctx.parallel_for(lA.shape(), lA.write())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto sA) {
    sA(i, j) = 42.0;
  };

  // Create a subset of a limited size
  box subset_shape(3, 4);
  ctx.parallel_for(subset_shape, lA.rw())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto sA) {
    sA(i, j) = 13.0;
  };

  bool checkedA = false;
  ctx.host_launch(lA.rw())->*[&checkedA](auto sA) {
    for (size_t j = 0; j < sA.extent(1); j++)
    {
      for (size_t i = 0; i < sA.extent(0); i++)
      {
        double expected = (i < 3 && j < 4) ? 13.0 : 42.0;
        if (sA(i, j) != expected)
        {
          fprintf(stderr, "sA(%zu,%zu) = %lf, expected %lf\n", i, j, sA(i, j), expected);
        }
      }
    }

    checkedA = true;
  };

  std::vector<double> B(M * N);
  auto lB = ctx.logical_data(make_slice(&B[0], std::tuple<size_t, size_t>{M, N}, M));

  ctx.parallel_for(lB.shape(), lB.write())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto sB) {
    sB(i, j) = 42.0;
  };

  // Create a subset of a limited size
  box subset_shape_2({2, 5}, {5, 8});
  ctx.parallel_for(subset_shape_2, lB.rw())->*[=] _CCCL_DEVICE(size_t i, size_t j, auto sB) {
    sB(i, j) = 13.0;
  };

  bool checkedB = false;
  ctx.host_launch(lB.rw())->*[&checkedB](auto sB) {
    for (size_t j = 0; j < sB.extent(1); j++)
    {
      for (size_t i = 0; i < sB.extent(0); i++)
      {
        double expected = (2 <= i && i < 5 && 5 <= j && j < 8) ? 13.0 : 42.0;
        if (sB(i, j) != expected)
        {
          fprintf(stderr, "sB(%zu,%zu) = %lf, expected %lf\n", i, j, sB(i, j), expected);
        }
      }
    }

    checkedB = true;
  };

  ctx.finalize();

  assert(checkedA);
  assert(checkedB);
}
