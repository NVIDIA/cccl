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

stream_ctx ctx;

class csr_matrix
{
public:
  csr_matrix(size_t num_rows, size_t num_nonzeros, double* values, size_t* row_offsets, size_t* column_indices)
  {
    val_handle = ctx.logical_data(make_slice(values, num_nonzeros));
    col_handle = ctx.logical_data(make_slice(column_indices, num_nonzeros));
    row_handle = ctx.logical_data(make_slice(row_offsets, num_rows + 1));
  }

  /* Description of the CSR */
  logical_data<slice<double>> val_handle;
  logical_data<slice<size_t>> row_handle;
  logical_data<slice<size_t>> col_handle;
};

class vector
{
public:
  vector(size_t N)
  {
    handle = ctx.logical_data(shape_of<slice<double>>(N));
  }

  static void copy_vector(const vector& from, vector& to)
  {
    ctx.parallel_for(to.handle.shape(), to.handle.write(), from.handle.read()).set_symbol("copy_vector")
        ->*[] _CCCL_DEVICE(size_t i, slice<double> dto, slice<double> dfrom) {
              dto(i) = dfrom(i);
            };
  }

  // Copy constructor
  vector(const vector& a)
  {
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

  logical_data<slice<double>> handle;
};

class scalar
{
public:
  scalar()
  {
    handle = ctx.logical_data(shape_of<slice<double>>(1));
  }

  static void copy_scalar(const scalar& from, scalar& to)
  {
    ctx.parallel_for(to.handle.shape(), to.handle.write(), from.handle.read())
        ->*[] _CCCL_DEVICE(size_t i, slice<double> dto, slice<double> dfrom) {
              dto(i) = dfrom(i);
            };
  }

  // Copy constructor
  scalar(const scalar& a)
  {
    handle = ctx.logical_data(shape_of<slice<double>>(1));
    copy_scalar(a, *this);
  }

  scalar operator/(scalar const& rhs) const
  {
    // Submit a task that computes this/rhs
    scalar res;
    ctx.parallel_for(handle.shape(), handle.read(), rhs.handle.read(), res.handle.write())
        ->*[] _CCCL_DEVICE(size_t i, auto dthis, auto drhs, auto dres) {
              dres(i) = dthis(i) / drhs(i);
            };

    return res;
  }

  scalar operator-() const
  {
    // Submit a task that computes -s
    scalar res;
    ctx.parallel_for(handle.shape(), handle.read(), res.handle.write())
        ->*[] _CCCL_DEVICE(size_t i, auto dthis, auto dres) {
              dres(i) = -dthis(i);
            };

    return res;
  }

  /* This methods reads the content of the logical data in a synchronous manner, and returns its value. */
  double get() const
  {
    double ret;
    ctx.task(exec_place::host, handle.read())->*[&](cudaStream_t stream, auto dval) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      ret = dval(0);
    };
    return ret;
  }

  logical_data<slice<double>> handle;
};

scalar DOT(vector& a, vector& b)
{
  scalar res;

  /* We initialize the result with a trivial kernel so that we do not need a
   * cooperative kernel launch for the reduction */
  ctx.parallel_for(box(1), res.handle.write())->*[] _CCCL_DEVICE(size_t, auto dres) {
    dres(0) = 0.0;
  };

  auto spec = par<128>(con<32>());
  ctx.launch(spec, exec_place::current_device(), a.handle.read(), b.handle.read(), res.handle.rw())
      ->*[] _CCCL_DEVICE(auto th, auto da, auto db, auto dres) {
            // Each thread computes the dot product of the elements assigned to it
            double local_sum = 0.0;
            for (auto i : th.apply_partition(shape(da), std::tuple<blocked_partition, cyclic_partition>()))
            {
              local_sum += da(i) * db(i);
            }

            auto ti = th.inner();
            __shared__ double block_sum[th.static_width(1)];
            block_sum[ti.rank()] = local_sum;

            /* Reduce within blocks */
            for (size_t s = ti.size() / 2; s > 0; s /= 2)
            {
              ti.sync();
              if (ti.rank() < s)
              {
                block_sum[ti.rank()] += block_sum[ti.rank() + s];
              }
            }

            /* Every first thread of a block writes its contribution */
            if (ti.rank() == 0)
            {
              atomicAdd(&dres(0), block_sum[0]);
            }
          };

  return res;
};

// Y = Y + alpha * X
void AXPY(const scalar& alpha, vector& x, vector& y)
{
  ctx.parallel_for(x.handle.shape(), alpha.handle.read(), x.handle.read(), y.handle.rw())
      ->*[] _CCCL_DEVICE(size_t i, auto dalpha, auto dx, auto dy) {
            dy(i) += dalpha(0) * dx(i);
          };
};

// Y = alpha*Y + X
void SCALE_AXPY(const scalar& alpha, const vector& x, vector& y)
{
  ctx.parallel_for(x.handle.shape(), alpha.handle.read(), x.handle.read(), y.handle.rw())
      ->*[] _CCCL_DEVICE(size_t i, auto dalpha, auto dx, auto dy) {
            dy(i) = dalpha(0) * dy(i) + dx(i);
          };
};

void SPMV(csr_matrix& a, vector& x, vector& y)
{
  ctx.parallel_for(
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
}

void cg(csr_matrix& A, vector& X, vector& B)
{
  size_t N = B.size();
  vector R = B;

  // R = R - A*X
  vector Ax(N);
  SPMV(A, X, Ax);
  R -= Ax;

  vector P = R;

  // RSOLD = R'*R
  scalar rsold = DOT(R, R);

  const int MAXITER = N;
  for (int k = 0; k < MAXITER; k++)
  {
    vector Ap(N);

    // Ap = A*P
    SPMV(A, P, Ap);

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
    double err = sqrt(rsnew.get());
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

  csr_matrix A(N, nz, values, row_offsets, column_indices);

  vector X(N), B(N);

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
