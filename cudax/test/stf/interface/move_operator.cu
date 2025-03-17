//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

using namespace cuda::experimental::stf;

template <typename T>
class foo
{
public:
  template <typename Ctx>
  foo(Ctx& ctx, T* array, size_t n)
      : h(ctx.logical_data(array, n))
  {}

private:
  logical_data_untyped h;
};

template <typename Ctx>
void run()
{
  Ctx ctx;

  const int N = 16;
  double X[N], Y[N], Z[N];

  for (size_t ind = 0; ind < N; ind++)
  {
    X[ind] = 0.0;
    Y[ind] = 0.0;
    Z[ind] = 0.0;
  }

  // Move logical_data_untyped directly
  logical_data_untyped h1 = ctx.logical_data(X);
  logical_data_untyped h2(std::move(h1));

  // Ensures the methodology used in the move ctor of logical_data_untyped is working
  // with multiple handles...
  logical_data_untyped h3 = ctx.logical_data(Y);
  logical_data_untyped h4(std::move(h3));

  // Make sure a class containing a logical_data_untyped is movable
  foo A = foo(ctx, &Z[0], N);
  foo B = std::move(A);

  ctx.finalize();
}

int main()
{
  run<stream_ctx>();
  run<graph_ctx>();
}
