//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Compute square roots via Newton's method using while_graph_scope
 *
 * This is a minimal example of an iterative solver with convergence
 * checking in a stackable context.  Each iteration applies the
 * Babylonian step  x <- (x + S/x) / 2  and reduces the maximum
 * absolute change across all elements.  The while loop exits once
 * the change drops below a tolerance.
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving example: while_graph_scope requires CUDA 12.4+.\n");
  return 0;
#else
  stackable_ctx ctx;

  constexpr size_t N   = 1024;
  constexpr double tol = 1e-12;

  ::std::vector<double> host_S(N);
  ::std::vector<double> host_X(N);

  for (size_t i = 0; i < N; i++)
  {
    host_S[i] = 1.0 + static_cast<double>(i);
    host_X[i] = host_S[i]; // initial guess x0 = S
  }

  auto lS = ctx.logical_data(make_slice(host_S.data(), N)).set_symbol("S");
  lS.set_read_only();

  auto lX       = ctx.logical_data(make_slice(host_X.data(), N)).set_symbol("X");
  auto lmax_err = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("max_err");

  {
    auto while_guard = ctx.while_graph_scope();

    // Babylonian step: x = (x + S/x) / 2, reduce max |change|
    ctx.parallel_for(box(N), lX.rw(), lS.read(), lmax_err.reduce(reducer::maxval<double>{}))
        ->*[] __device__(size_t i, auto x, auto s, auto& max_err) {
              double x_old = x(i);
              double x_new = 0.5 * (x_old + s(i) / x_old);
              x(i)         = x_new;
              max_err      = fabs(x_new - x_old);
            };

    while_guard.update_cond(lmax_err.read())->*[tol] __device__(auto max_err) {
      return (*max_err > tol);
    };
  }

  ctx.finalize();

  for (size_t i = 0; i < N; i++)
  {
    double expected = sqrt(1.0 + static_cast<double>(i));
    EXPECT(fabs(host_X[i] - expected) < 1e-8);
  }

  return 0;
#endif
}
