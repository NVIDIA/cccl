//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

//! \file
//!
//! \brief Freeze data and store it as a frozen_logical_data_untyped object

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

int X0(int i)
{
  return 17 * i + 45;
}

__global__ void mult(slice<int> s, int val)
{
  int tid      = blockIdx.x * blockDim.x + threadIdx.x;
  int nthreads = gridDim.x * blockDim.x;

  for (int i = tid; i < s.size(); i += nthreads)
  {
    s(i) *= val;
  }
}

int main()
{
  stream_ctx ctx;

  cudaStream_t stream = ctx.pick_stream();

  const int N = 16;
  int X[N];

  for (int i = 0; i < N; i++)
  {
    X[i] = X0(i);
  }

  auto lX = ctx.logical_data(X).set_symbol("X");
  auto lY = ctx.logical_data(lX.shape()).set_symbol("Y");

  for (int k = 0; k < 4; k++)
  {
    logical_data_untyped lX_untyped = lX;
    auto fx                         = ctx.freeze(lX_untyped, access_mode::rw, data_place::current_device());

    _CCCL_ASSERT(fx.get_access_mode() == access_mode::rw, "invalid access mode");

    auto dX = fx.template get<slice<int>>(data_place::current_device(), stream);
    mult<<<8, 4, 0, stream>>>(dX, 4);
    fx.unfreeze(stream);

    ctx.parallel_for(lX.shape(), lX.read(), lY.write()).set_symbol("Y=X")->*[] __device__(size_t i, auto x, auto y) {
      y(i) = x(i);
    };

    ctx.parallel_for(lX.shape(), lY.rw()).set_symbol("Y+=1")->*[] __device__(size_t i, auto y) {
      y(i) += 1;
    };
  }

  ctx.finalize();
}
