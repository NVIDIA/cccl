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
 * @brief Generic Newton Solver
 */

#include <cuda/experimental/stf.cuh>

#include "cg_solver.cuh"
#include "dot.cuh" 
  
using namespace cuda::experimental::stf;

/**
 * Generic Newton solver for nonlinear systems F(x) = 0
 *
 * @tparam ctx_t STF context type
 * @tparam ResidualCallback Callback to compute residual F(x)
 * @tparam JacobianCallback Callback to assemble Jacobian J = ∂F/∂x
 *
 * The callbacks must be callable with these signatures:
 * - ResidualCallback: void fn(ctx_t&, const vector_t<double>& x, const vector_t<double>& x_prev, vector_t<double>&
 * residual)
 * - JacobianCallback: void fn(ctx_t&, const vector_t<double>& x, vector_t<double>& jacobian_values)
 */
template <typename ctx_t, typename ResidualCallback, typename JacobianCallback>
void newton_solver(
  ctx_t& ctx,
  vector_t<double>& U,
  vector_t<double>& csr_values,
  const vector_t<size_t>& csr_row_offsets,
  const vector_t<size_t>& csr_col_ind,
  ResidualCallback compute_residual_fn,
  JacobianCallback assemble_jacobian_fn,
  size_t max_newton = 20,
  double newton_tol = 1e-10,
  size_t max_cg     = 100)
{
  auto U_prev = ctx.logical_data(U.shape()).set_symbol("U_prev");

  ctx.parallel_for(U.shape(), U_prev.write(), U.read()).set_symbol("init_guess")
      ->*[] __device__(size_t i, auto dU_prev, auto dU) {
            dU_prev(i) = dU(i);
          };

  auto newton_norm2 = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("newton_norm2");
  auto newton_iter  = ctx.logical_data(shape_of<scalar_view<size_t>>()).set_symbol("newton_iter");
  ctx.parallel_for(box(1), newton_iter.write()).set_symbol("init_newton_iter")->*[] _CCCL_DEVICE(size_t i, auto diter) {
    *diter = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    auto residual = ctx.logical_data(U.shape()).set_symbol("residual");
    auto delta    = ctx.logical_data(U.shape()).set_symbol("delta");

    // Compute residual F(U)
    compute_residual_fn(ctx, U, U_prev, residual);

    // Compute Newton residual norm for convergence check
    DOT(ctx, residual, residual, newton_norm2);

    // Assemble Jacobian J = ∂F/∂U
    assemble_jacobian_fn(ctx, U, csr_values);

    auto rhs = ctx.logical_data(U.shape()).set_symbol("rhs");

    // Set up RHS: rhs = -F(U)
    ctx.parallel_for(rhs.shape(), rhs.write(), residual.read()).set_symbol("rhs = -residual")
        ->*[] __device__(size_t i, auto drhs, auto dresidual) {
              drhs(i) = -dresidual(i);
            };

    csr_matrix<double> A(csr_values, csr_row_offsets, csr_col_ind);

    // Solve linear system: J * delta = -F(U)
    double cg_tol = 1e-8;
    cg_solver(ctx, A, delta, rhs, cg_tol, max_cg);

    // Newton update: U = U + delta (no special boundary handling needed)
    ctx.parallel_for(U.shape(), U.rw(), delta.read()).set_symbol("newton_update")
        ->*[] __device__(size_t i, auto dU, auto ddelta) {
              dU(i) += ddelta(i);
            };

    while_guard.update_cond(newton_norm2.read(), newton_iter.rw())
        ->*[newton_tol, max_newton] __device__(auto dnorm2, auto diter) {
              (*diter)++; // increment iteration counter
              bool converged = (*dnorm2 < newton_tol * newton_tol);
              return !converged && (*diter < max_newton);
            };
  }
}


