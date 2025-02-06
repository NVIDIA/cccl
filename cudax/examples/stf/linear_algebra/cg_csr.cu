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

struct csr_matrix
{
  csr_matrix(const context& _ctx,
             size_t num_rows,
             size_t num_nonzeros,
             double* values,
             size_t* row_offsets,
             size_t* column_indices)
      : ctx(_ctx)
  {
    val_handle = ctx.logical_data(make_slice(values, num_nonzeros));
    col_handle = ctx.logical_data(make_slice(column_indices, num_nonzeros));
    row_handle = ctx.logical_data(make_slice(row_offsets, num_rows + 1));
  }

  /* Description of the CSR */
  mutable logical_data<slice<double>> val_handle;
  mutable logical_data<slice<size_t>> row_handle;
  mutable logical_data<slice<size_t>> col_handle;
  mutable context ctx;
};

class vector
{
public:
  vector(const context& _ctx, size_t N)
      : ctx(_ctx)
  {
    handle = ctx.logical_data(shape_of<slice<double>>(N));
  }

  static void copy_vector(const vector& from, vector& to)
  {
    to.ctx.parallel_for(to.handle.shape(), to.handle.write(), from.handle.read()).set_symbol("copy_vector")
        ->*[] _CCCL_DEVICE(size_t i, slice<double> dto, slice<const double> dfrom) {
              dto(i) = dfrom(i);
            };
  }

  // Copy constructor
  vector(const vector& a)
  {
    ctx    = a.ctx;
    handle = ctx.logical_data(a.handle.shape());
    copy_vector(a, *this);
  }

  // this -= rhs
  vector& operator-=(const vector& rhs)
  {
    ctx.parallel_for(handle.shape(), handle.rw(), rhs.handle.read()).set_symbol("-= vector")
        ->*[] _CCCL_DEVICE(size_t i, auto dthis, auto drhs) {
              dthis(i) -= drhs(i);
            };

    return *this;
  }

  size_t size() const
  {
    return handle.shape().size();
  }

  mutable logical_data<slice<double>> handle;
  mutable context ctx;
};

class scalar
{
public:
  scalar(const context& _ctx)
      : ctx(_ctx)
  {
    handle = ctx.logical_data(shape_of<scalar_view<double>>());
  }

  static void copy_scalar(const scalar& from, scalar& to)
  {
    to.ctx.parallel_for(box(1), to.handle.write(), from.handle.read())->*[] _CCCL_DEVICE(size_t i, auto dto, auto dfrom) {
      *dto = *dfrom;
    };
  }

  // Assign constructor
  scalar& operator=(scalar const& rhs)
  {
    ctx    = rhs.ctx;
    handle = ctx.logical_data(rhs.handle.shape());
    return *this;
  }

  // Copy constructor
  scalar(const scalar& a)
  {
    handle = ctx.logical_data(a.handle.shape());
    ctx    = a.ctx;
    copy_scalar(a, *this);
  }

  scalar& operator=(scalar&& a)
  {
    handle = mv(a.handle);
    ctx    = mv(a.ctx);
    return *this;
  }

  scalar operator/(scalar const& rhs) const
  {
    // Submit a task that computes this/rhs
    scalar res(ctx);
    res.ctx.parallel_for(box(1), handle.read(), rhs.handle.read(), res.handle.write())
        ->*[] _CCCL_DEVICE(size_t i, auto dthis, auto drhs, auto dres) {
              *dres = *dthis / *drhs;
            };

    return res;
  }

  scalar operator-() const
  {
    // Submit a task that computes -s
    scalar res(ctx);
    res.ctx.parallel_for(box(1), handle.read(), res.handle.write())->*[] _CCCL_DEVICE(size_t i, auto dthis, auto dres) {
      *dres = -*dthis;
    };

    return res;
  }

  mutable logical_data<scalar_view<double>> handle;
  mutable context ctx;
};

scalar DOT(vector& a, vector& b)
{
  scalar res(a.ctx);

  a.ctx.parallel_for(a.handle.shape(), a.handle.read(), b.handle.read(), res.handle.reduce(reducer::sum<double>{}))
      ->*[] __device__(size_t i, auto da, auto db, double& dres) {
            dres += da(i) * db(i);
          };

  return res;
};

// Y = Y + alpha * X
void AXPY(const scalar& alpha, vector& x, vector& y)
{
  y.ctx.parallel_for(x.handle.shape(), alpha.handle.read(), x.handle.read(), y.handle.rw())
      ->*[] _CCCL_DEVICE(size_t i, auto dalpha, auto dx, auto dy) {
            dy(i) += *dalpha * dx(i);
          };
};

// Y = alpha*Y + X
void SCALE_AXPY(const scalar& alpha, const vector& x, vector& y)
{
  y.ctx.parallel_for(x.handle.shape(), alpha.handle.read(), x.handle.read(), y.handle.rw())
      ->*[] _CCCL_DEVICE(size_t i, auto dalpha, auto dx, auto dy) {
            dy(i) = *dalpha * dy(i) + dx(i);
          };
};

vector SPMV(csr_matrix& a, vector& x)
{
  vector y(x.ctx, x.size());

  y.ctx.parallel_for(
    y.handle.shape(), a.val_handle.read(), a.col_handle.read(), a.row_handle.read(), x.handle.read(), y.handle.write())
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

  return y;
}

void cg(csr_matrix& A, vector& X, vector& B)
{
  context ctx = A.ctx;
  size_t N    = B.size();
  vector R    = B;

  // R = R - A*X
  auto Ax = SPMV(A, X);
  R -= Ax;

  vector P = R;

  // RSOLD = R'*R
  scalar rsold = DOT(R, R);

  const int MAXITER = N;
  for (int k = 0; k < MAXITER; k++)
  {
    // Ap = A*P
    auto Ap = SPMV(A, P);

    // alpha = rsold / (p' * Ap);
    scalar alpha = rsold / DOT(P, Ap);

    // x = x + alpha * p;
    AXPY(alpha, P, X);

    // r = r - alpha * Ap;
    AXPY(-alpha, Ap, R);

    // rsnew = r' * r;
    scalar rsnew = DOT(R, R);

    // Read the residual on the CPU, and halt the iterative process if we have converged
    // (note that this will block the submission of tasks)
    double err = ctx.wait(rsnew.handle);
    if (err < 1e-10)
    {
      // We have converged
      fprintf(stderr, "Successfully converged (err = %le)\n", err);
      break;
    }

    // p = r + (rsnew / rsold) * p;
    SCALE_AXPY(rsnew / rsold, R, P);

    rsold = mv(rsnew);
  }
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

void cg_solver(size_t N, size_t nz, size_t* row_offsets, size_t* column_indices, double* values)
{
  context ctx;

  csr_matrix A(ctx, N, nz, values, row_offsets, column_indices);

  vector X(ctx, N), B(ctx, N);

  // RHS
  ctx.parallel_for(B.handle.shape(), B.handle.write())->*[] _CCCL_DEVICE(size_t i, auto dB) {
    dB(i) = 1.0;
  };

  // Initial guess
  ctx.parallel_for(X.handle.shape(), X.handle.write())->*[] _CCCL_DEVICE(size_t i, auto dX) {
    dX(i) = 1.0;
  };

  cg(A, X, B);

  ctx.finalize();
}

int main(int argc, char** argv)
{
  size_t N = 10485760;

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

  // Solve the system using the CG algorithm
  cg_solver(N, nz, row_offsets, column_indices, values);
}
