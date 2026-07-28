//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once
/**
 * @file
 * @brief Sparse conjugate gradient algorithm
 */

#include <cuda/experimental/stf.cuh>

#include "dot.cuh"

using namespace cuda::experimental::stf;

#if !_CCCL_CTK_BELOW(12, 4)
template <typename ctx_t, typename T>
void cg_solver(ctx_t& ctx, csr_matrix<T>& A, vector_t<T>& X, vector_t<T>& B, double cg_tol = 1e-10, size_t max_cg = 1000)
{
  // Initial guess X = 0 (better for Newton corrections)
  ctx.parallel_for(X.shape(), X.write()).set_symbol("init_guess")->*[] _CCCL_DEVICE(size_t i, auto dX) {
    dX(i) = 0.0;
  };

  // Residual R initialized to B
  auto R = ctx.logical_data(B.shape()).set_symbol("R");
  ctx.parallel_for(R.shape(), R.write(), B.read()).set_symbol("R=B")->*[] _CCCL_DEVICE(size_t i, auto dR, auto dB) {
    dR(i) = dB(i);
  };

  // R = R - A*X
  auto Ax = ctx.logical_data(X.shape()).set_symbol("Ax");
  SPMV(ctx, A, X, Ax);
  ctx.parallel_for(R.shape(), R.rw(), Ax.read()).set_symbol("R -= Ax")->*[] _CCCL_DEVICE(size_t i, auto dR, auto dAx) {
    dR(i) -= dAx(i);
  };

  // P = R;
  auto P = ctx.logical_data(R.shape()).set_symbol("P");
  ctx.parallel_for(P.shape(), P.write(), R.read()).set_symbol("P=R")->*[] _CCCL_DEVICE(size_t i, auto dP, auto dR) {
    dP(i) = dR(i);
  };

  // RSOLD = R'*R
  auto rsold = ctx.logical_data(shape_of<scalar_view<T>>()).set_symbol("rsold");
  DOT(ctx, R, R, rsold);

  // CG iteration counter
  auto cg_iter = ctx.logical_data(shape_of<scalar_view<int>>()).set_symbol("cg_iter");
  ctx.parallel_for(box(1), cg_iter.write()).set_symbol("init_cg_iter")->*[] _CCCL_DEVICE(size_t i, auto diter) {
    *diter = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    // Ap = A*P
    auto Ap = ctx.logical_data(P.shape()).set_symbol("Ap");
    SPMV(ctx, A, P, Ap);

    // We don't compute alpha explicitly
    // alpha = rsold / (p' * Ap);
    auto pAp = ctx.logical_data(shape_of<scalar_view<T>>()).set_symbol("pAp");
    DOT(ctx, P, Ap, pAp);

    // x = x + alpha * p;
    ctx.parallel_for(X.shape(), X.rw(), rsold.read(), pAp.read(), P.read()).set_symbol("X+=alpha*P")
        ->*[] _CCCL_DEVICE(size_t i, auto dX, auto drsold, auto dpAp, auto dP) {
              T alpha = (*drsold / *dpAp);
              dX(i) += alpha * dP(i);
            };

    // r = r - alpha * Ap;
    ctx.parallel_for(R.shape(), R.rw(), rsold.read(), pAp.read(), Ap.read()).set_symbol("R-=alpha*Ap")
        ->*[] _CCCL_DEVICE(size_t i, auto dR, auto drsold, auto dpAp, auto dAp) {
              T alpha = (*drsold / *dpAp);
              dR(i) -= alpha * dAp(i);
            };

    // rsnew = r' * r;
    auto rsnew = ctx.logical_data(shape_of<scalar_view<T>>()).set_symbol("rsnew");
    DOT(ctx, R, R, rsnew);

    while_guard.update_cond(rsnew.read(), cg_iter.rw())->*[cg_tol, max_cg] __device__(auto drsnew, auto diter) {
      (*diter)++; // increment iteration counter
      bool converged = (*drsnew < cg_tol * cg_tol);
      // printf("CG iter %d: RES %e (tol=%e)\n", *diter, sqrt(*drsnew), cg_tol);
      return !converged && (*diter < max_cg);
    };

    // p = r + (rsnew / rsold) * p;
    ctx.parallel_for(P.shape(), P.rw(), R.read(), rsnew.read(), rsold.read()).set_symbol("P=r+(rsnew/rsold)*P")
        ->*[] _CCCL_DEVICE(size_t i, auto dP, auto dR, auto drsnew, auto drsold) {
              dP(i) = dR(i) + (*drsnew / *drsold) * dP(i);
            };

    // update old residual
    ctx.parallel_for(box(1), rsold.write(), rsnew.read()).set_symbol("update_rsold")
        ->*[] _CCCL_DEVICE(size_t i, auto drsold, auto drsnew) {
              *drsold = *drsnew;
            };
  }
}

template <typename ctx_t, typename T>
void cg_solver_no_while(
  ctx_t& ctx, csr_matrix<T>& A, vector_t<T>& X, vector_t<T>& B, double cg_tol = 1e-10, size_t max_cg = 1000)
{
  // Initial guess X = 0 (better for Newton corrections)
  ctx.parallel_for(X.shape(), X.write()).set_symbol("init_guess")->*[] _CCCL_DEVICE(size_t i, auto dX) {
    dX(i) = 0.0;
  };

  // Residual R initialized to B
  auto R = ctx.logical_data(B.shape()).set_symbol("R");
  ctx.parallel_for(R.shape(), R.write(), B.read()).set_symbol("R=B")->*[] _CCCL_DEVICE(size_t i, auto dR, auto dB) {
    dR(i) = dB(i);
  };

  // R = R - A*X
  auto Ax = ctx.logical_data(X.shape()).set_symbol("Ax");
  SPMV(ctx, A, X, Ax);
  ctx.parallel_for(R.shape(), R.rw(), Ax.read()).set_symbol("R -= Ax")->*[] _CCCL_DEVICE(size_t i, auto dR, auto dAx) {
    dR(i) -= dAx(i);
  };

  // P = R;
  auto P = ctx.logical_data(R.shape()).set_symbol("P");
  ctx.parallel_for(P.shape(), P.write(), R.read()).set_symbol("P=R")->*[] _CCCL_DEVICE(size_t i, auto dP, auto dR) {
    dP(i) = dR(i);
  };

  // RSOLD = R'*R
  auto rsold = ctx.logical_data(shape_of<scalar_view<T>>()).set_symbol("rsold");
  DOT(ctx, R, R, rsold);

  size_t iter = 0;
  auto rsnew  = ctx.logical_data(shape_of<scalar_view<T>>()).set_symbol("rsnew");

  do
  {
    // Ap = A*P
    auto Ap = ctx.logical_data(P.shape()).set_symbol("Ap");
    SPMV(ctx, A, P, Ap);

    // We don't compute alpha explicitly
    // alpha = rsold / (p' * Ap);
    auto pAp = ctx.logical_data(shape_of<scalar_view<T>>()).set_symbol("pAp");
    DOT(ctx, P, Ap, pAp);

    // x = x + alpha * p;
    ctx.parallel_for(X.shape(), X.rw(), rsold.read(), pAp.read(), P.read()).set_symbol("X+=alpha*P")
        ->*[] _CCCL_DEVICE(size_t i, auto dX, auto drsold, auto dpAp, auto dP) {
              T alpha = (*drsold / *dpAp);
              dX(i) += alpha * dP(i);
            };

    // r = r - alpha * Ap;
    ctx.parallel_for(R.shape(), R.rw(), rsold.read(), pAp.read(), Ap.read()).set_symbol("R-=alpha*Ap")
        ->*[] _CCCL_DEVICE(size_t i, auto dR, auto drsold, auto dpAp, auto dAp) {
              T alpha = (*drsold / *dpAp);
              dR(i) -= alpha * dAp(i);
            };

    // rsnew = r' * r;
    DOT(ctx, R, R, rsnew);

    // p = r + (rsnew / rsold) * p;
    ctx.parallel_for(P.shape(), P.rw(), R.read(), rsnew.read(), rsold.read()).set_symbol("P=r+(rsnew/rsold)*P")
        ->*[] _CCCL_DEVICE(size_t i, auto dP, auto dR, auto drsnew, auto drsold) {
              dP(i) = dR(i) + (*drsnew / *drsold) * dP(i);
            };

    // update old residual
    ctx.parallel_for(box(1), rsold.write(), rsnew.read()).set_symbol("update_rsold")
        ->*[] _CCCL_DEVICE(size_t i, auto drsold, auto drsnew) {
              *drsold = *drsnew;
            };
  } while ((++iter < max_cg) && (ctx.wait(rsnew) > cg_tol * cg_tol));
}

#endif
