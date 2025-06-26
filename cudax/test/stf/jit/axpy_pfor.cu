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
 * @brief JIT version of the AXPY example
 *
 */

#include <cuda/experimental/__stf/nvrtc/jit_utils.cuh>
#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

double X0(int i)
{
  return sin((double) i);
}

double Y0(int i)
{
  return cos((double) i);
}

int main()
{
  const size_t N = 16;
  double X[N], Y[N];

  for (size_t i = 0; i < N; i++)
  {
    X[i] = X0(i);
    Y[i] = Y0(i);
  }

  context ctx;
  double alpha = 3.14;

  auto lX = ctx.logical_data(X);
  auto lY = ctx.logical_data(Y);

  /* Compute Y = Y + alpha X */
  parallel_for_scope_jit(ctx, exec_place::current_device(), lY.shape(), lX.read(), lY.rw())->*[alpha]() {
    const char* header_template = R"(
      #include <cuda/experimental/__stf/nvrtc/slice.cuh>
      )";

    ::std::ostringstream body_stream;
    body_stream << R"(
      (size_t i, auto dX, auto dY) {
        dY(i) += )"
                << alpha << R"(* dX(i);
      })";

    return ::std::pair(::std::string(header_template), body_stream.str());
  };

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    assert(fabs(Y[i] - (Y0(i) + alpha * X0(i))) < 0.0001);
    assert(fabs(X[i] - X0(i)) < 0.0001);
  }
}
