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
 *
 * @brief Ensure we can nest push/pop section to author composable code
 *
 */

#include <cuda/experimental/__stf/utility/stackable_ctx.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

// Z += X*Y
void fma_lib(stackable_ctx& sctx,
             stackable_logical_data<slice<int>>& lX,
             stackable_logical_data<slice<int>>& lY,
             stackable_logical_data<slice<int>>& lZ)
{
  stackable_ctx::graph_scope scope{sctx};
  lX.push(access_mode::read);
  lY.push(access_mode::read);
  sctx.parallel_for(lZ.shape(), lZ.rw(), lX.read(), lY.read())->*[] __device__(size_t i, auto z, auto x, auto y) {
    z(i) += x(i) * y(i);
  };
}

// Z += (XiYi) for all i
void dot_lib(stackable_ctx& sctx,
             ::std::vector<stackable_logical_data<slice<int>>>& vecx,
             ::std::vector<stackable_logical_data<slice<int>>>& vecy,
             stackable_logical_data<slice<int>>& Z)
{
  stackable_ctx::graph_scope scope{sctx};
  for (size_t i = 0; i < vecx.size(); i++)
  {
    // Force read push to stress test nested context handling
    vecx[i].push(access_mode::read);
    vecy[i].push(access_mode::read);

    fma_lib(sctx, vecx[i], vecy[i], Z);
  }
}

int main()
{
  stackable_ctx sctx;

  size_t sz = 1024;
  ::std::vector<int> X(sz), Y(sz);

  ::std::vector<stackable_logical_data<slice<int>>> vecx, vecy;

  int expected = 0;

  for (size_t i = 0; i < sz; i++)
  {
    X[i] = i;
    vecx.push_back(sctx.logical_data(make_slice(&X[i], 1)));

    Y[i] = (i - 1);
    vecy.push_back(sctx.logical_data(make_slice(&Y[i], 1)));

    expected += i * (i - 1);
  }

  int result   = 0;
  auto lresult = sctx.logical_data(make_slice(&result, 1));

  dot_lib(sctx, vecx, vecy, lresult);

  sctx.finalize();

  _CCCL_ASSERT(result == expected, "invalid result");
}
