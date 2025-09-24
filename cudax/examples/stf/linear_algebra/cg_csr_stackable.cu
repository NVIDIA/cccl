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
 * @brief Sparse conjugate gradient algorithm
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

using vector_t  = stackable_logical_data<slice<double>>;
using scalar_t  = stackable_logical_data<scalar_view<double>>;
using context_t = stackable_ctx;

struct csr_matrix
{
  csr_matrix(
    context_t& ctx, size_t num_rows, size_t num_nonzeros, double* values, size_t* row_offsets, size_t* column_indices)
  {
    val_handle = ctx.logical_data(make_slice(values, num_nonzeros));
    col_handle = ctx.logical_data(make_slice(column_indices, num_nonzeros));
    row_handle = ctx.logical_data(make_slice(row_offsets, num_rows + 1));

    val_handle.set_symbol("csr_val");
    col_handle.set_symbol("csr_col");
    row_handle.set_symbol("csr_row");
  }

  /* Description of the CSR */
  mutable stackable_logical_data<slice<double>> val_handle;
  mutable stackable_logical_data<slice<size_t>> row_handle;
  mutable stackable_logical_data<slice<size_t>> col_handle;
};

// Note that a and b might be the same logical data
void DOT(context_t& ctx, vector_t& a, vector_t& b, scalar_t& res)
{
  ctx.parallel_for(a.shape(), a.read(), b.read(), res.reduce(reducer::sum<double>{})).set_symbol("DOT")->*
    [] __device__(size_t i, auto da, auto db, double& dres) {
      dres += da(i) * db(i);
    };
};

void SPMV(context_t& ctx, csr_matrix& a, vector_t& x, vector_t& y)
{
  ctx.parallel_for(y.shape(), a.val_handle.read(), a.col_handle.read(), a.row_handle.read(), x.read(), y.write())
      .set_symbol("SPMV")
      ->*[] _CCCL_DEVICE(size_t row, auto da_val, auto da_col, auto da_row, auto dx, auto dy) {
            int row_start = da_row(row);
            int row_end   = da_row(row + 1);

            double sum = 0.0;
            for (int elt = row_start; elt < row_end; elt++)
            {
              sum += da_val(elt) * dx(da_col(elt));
            }

            dy(row) = sum;
          };
}

/* genTridiag: generate a random tridiagonal symmetric matrix
   from :
   https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/conjugateGradientCudaGraphs/conjugateGradientCudaGraphs.cu
 */
void genTridiag(size_t* I, size_t* J, double* val, size_t N, size_t nz)
{
  const double d = 2.0;

  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = drand48() + d;
  val[1] = drand48();
  int start;

  for (size_t i = 1; i < N; i++)
  {
    if (i > 1)
    {
      I[i] = I[i - 1] + 3;
    }
    else
    {
      I[1] = 2;
    }

    start        = (i - 1) * 3 + 2;
    J[start]     = i - 1;
    J[start + 1] = i;

    if (i < N - 1)
    {
      J[start + 2] = i + 1;
    }

    val[start]     = val[start - 1];
    val[start + 1] = drand48() + d;

    if (i < N - 1)
    {
      val[start + 2] = drand48();
    }
  }

  I[N] = nz;
}

void cg_solver(context_t& ctx, csr_matrix& A, vector_t& X, vector_t& B)
{
  // Initial guess X = 1
  ctx.parallel_for(X.shape(), X.write()).set_symbol("init_guess")->*[] _CCCL_DEVICE(size_t i, auto dX) {
    dX(i) = 1.0;
  };

  // Residual R initialized to B
  auto R = ctx.logical_data(B.shape());
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
  auto rsold = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("rsold");
  DOT(ctx, R, R, rsold);

  {
    auto while_guard = ctx.while_graph_scope();

    // Ap = A*P
    auto Ap = ctx.logical_data(P.shape()).set_symbol("Ap");
    SPMV(ctx, A, P, Ap);

    // We don't compute alpha explicitly
    // alpha = rsold / (p' * Ap);
    auto pAp = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("pAp");
    DOT(ctx, P, Ap, pAp);

    // x = x + alpha * p;
    ctx.parallel_for(X.shape(), X.rw(), rsold.read(), pAp.read(), P.read()).set_symbol("X+=alpha*P")
        ->*[] _CCCL_DEVICE(size_t i, auto dX, auto drsold, auto dpAp, auto dP) {
              double alpha = (*drsold / *dpAp);
              dX(i) += alpha * dP(i);
            };

    // r = r - alpha * Ap;
    ctx.parallel_for(R.shape(), R.rw(), rsold.read(), pAp.read(), Ap.read()).set_symbol("R-=alpha*Ap")
        ->*[] _CCCL_DEVICE(size_t i, auto dR, auto drsold, auto dpAp, auto dAp) {
              double alpha = (*drsold / *dpAp);
              dR(i) -= alpha * dAp(i);
            };

    // rsnew = r' * r;
    auto rsnew = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("rsnew");
    DOT(ctx, R, R, rsnew);

    while_guard.update_cond(rsnew.read())->*[] __device__(auto drsnew) {
      printf("RES %e\n", *drsnew);
      bool converged = (*drsnew < 1e-13);
      return !converged;
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

int main(int argc, char** argv)
{
  size_t N = 10485760;

  context_t ctx;

  if (argc > 1)
  {
    N = atoi(argv[1]);
    fprintf(stderr, "N = %zu\n", N);
  }

  size_t nz = (N - 2) * 3 + 4;

  size_t* row_offsets;
  size_t* column_indices;
  double* values;
  cuda_safe_call(cudaHostAlloc(&row_offsets, (N + 1) * sizeof(size_t), cudaHostAllocMapped));
  cuda_safe_call(cudaHostAlloc(&column_indices, nz * sizeof(size_t), cudaHostAllocMapped));
  cuda_safe_call(cudaHostAlloc(&values, nz * sizeof(double), cudaHostAllocMapped));

  // Generate a random matrix that is supposed to be invertible
  genTridiag(row_offsets, column_indices, values, N, nz);

  csr_matrix A(ctx, N, nz, values, row_offsets, column_indices);

  auto X = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("X");
  auto B = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("B");

  // RHS
  ctx.parallel_for(B.shape(), B.write()).set_symbol("B assembly")->*[] __device__(size_t i, auto dB) {
    dB(i) = 1.0;
  };

  cg_solver(ctx, A, X, B);

  ctx.finalize();
}
