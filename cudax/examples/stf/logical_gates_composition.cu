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
 * @brief Composition of boolean operations applied on logical data
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// z = AND(x,y)
logical_data<slice<int>> AND(context& ctx, logical_data<slice<int>> x, logical_data<slice<int>> y)
{
  assert(x.shape().size() == y.shape().size());

  auto z = ctx.logical_data(x.shape());

  std::string symbol = "(" + x.get_symbol() + " & " + y.get_symbol() + ")";
  z.set_symbol(symbol);

  ctx.parallel_for(z.shape(), x.read(), y.read(), z.write()).set_symbol("AND")->*
    [] __device__(size_t i, auto dx, auto dy, auto dz) {
      dz(i) = dx(i) & dy(i);
    };

  return z;
}

// y = NOT(x)
logical_data<slice<int>> NOT(context& ctx, logical_data<slice<int>> x)
{
  auto y = ctx.logical_data(x.shape());

  std::string symbol = "( !" + x.get_symbol() + ")";
  y.set_symbol(symbol);

  ctx.parallel_for(y.shape(), x.read(), y.write()).set_symbol("NOT")->*[] __device__(size_t i, auto dx, auto dy) {
    dy(i) = ~dx(i);
  };

  return y;
}

int main()
{
  const size_t n = 12;

  int X[n], Y[n], Z[n];

  context ctx;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);
  auto lZ = ctx.logical_data(Z);

  lX.set_symbol("X");
  lY.set_symbol("Y");
  lZ.set_symbol("Z");

  auto lB = AND(ctx, AND(ctx, lX, lY), AND(ctx, lX, lZ));
  auto lC = AND(ctx, NOT(ctx, AND(ctx, lB, NOT(ctx, lY))), lX);

  ctx.finalize();
}
