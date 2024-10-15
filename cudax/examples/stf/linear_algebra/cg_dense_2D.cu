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
 * @brief Conjugate gradient for a tiled dense matrix
 */

#include <cuda/experimental/stf.cuh>

using namespace cuda::experimental::stf;

static cublasHandle_t cublas_handle;

stream_ctx ctx;

class matrix
{
public:
  matrix(size_t N)
      : N(N)
  {
    h_addr.reset(new double[N * N]);
    cuda_safe_call(cudaHostRegister(h_addr.get(), N * N * sizeof(double), cudaHostRegisterPortable));
    handle = to_shared(ctx.logical_data(make_slice(h_addr.get(), std::tuple{N, N}, N)));
  }

  void fill(const std::function<double(int, int)>& f)
  {
    ctx.task(exec_place::host, handle->write())->*[&f](cudaStream_t stream, auto ds) {
      cuda_safe_call(cudaStreamSynchronize(stream));

      for (size_t col = 0; col < ds.extent(1); col++)
      {
        for (size_t row = 0; row < ds.extent(0); row++)
        {
          ds(row, col) = f(row, col);
        }
      }
    };
  }

  size_t N;
  std::unique_ptr<double[]> h_addr;
  std::shared_ptr<logical_data<slice<double, 2>>> handle;
};

class vector
{
public:
  vector(size_t N, size_t _block_size, bool is_tmp = false)
      : N(N)
      , block_size(_block_size)
      , nblocks((N + block_size - 1) / block_size)
  {
    handles.resize(nblocks);

    if (is_tmp)
    {
      // There is no physical backing for this temporary vector
      for (int b = 0; b < nblocks; b++)
      {
        size_t bs  = std::min(N - block_size * b, block_size);
        handles[b] = to_shared(ctx.logical_data(shape_of<slice<double>>(bs)));
      }
    }
    else
    {
      h_addr.reset(new double[N]);
      cuda_safe_call(cudaHostRegister(h_addr.get(), N * sizeof(double), cudaHostRegisterPortable));
      for (size_t b = 0; b < nblocks; b++)
      {
        size_t bs  = std::min(N - block_size * b, block_size);
        handles[b] = to_shared(ctx.logical_data(make_slice(&h_addr[block_size * b], bs)));
      }
    }
  }

  // Copy constructor
  vector(const vector& a)
      : N(a.N)
      , block_size(a.block_size)
      , nblocks(a.nblocks)
  {
    handles.resize(nblocks);

    for (int b = 0; b < nblocks; b++)
    {
      size_t bs  = std::min(N - block_size * b, block_size);
      handles[b] = to_shared(ctx.logical_data(shape_of<slice<double>>(bs)));

      ctx.task(handles[b]->write(), a.handles[b]->read())->*[bs](cudaStream_t stream, auto dthis, auto da) {
        // There are likely much more efficient ways.
        cuda_safe_call(cudaMemcpyAsync(
          dthis.data_handle(), da.data_handle(), bs * sizeof(double), cudaMemcpyDeviceToDevice, stream));
      };
    }
  }

  void fill(const std::function<double(int)>& f)
  {
    size_t bs = block_size;
    for (int b = 0; b < nblocks; b++)
    {
      ctx.task(exec_place::host, handles[b]->write())->*[&f, b, bs](cudaStream_t stream, auto ds) {
        cuda_safe_call(cudaStreamSynchronize(stream));

        for (int local_row = 0; local_row < ds.extent(0); local_row++)
        {
          ds(local_row) = f(local_row + b * bs);
        }
      };
    }
  }

  size_t N;
  size_t block_size;
  size_t nblocks;

  mutable std::vector<std::shared_ptr<logical_data<slice<double>>>> handles;
  std::unique_ptr<double[]> h_addr;
};

__global__ void scalar_div(const double* a, const double* b, double* c)
{
  *c = *a / *b;
}

// A += B
__global__ void scalar_add(double* a, const double* b)
{
  *a = *a + *b;
}

__global__ void scalar_minus(const double* a, double* res)
{
  *res = -(*a);
}

class scalar
{
public:
  scalar(bool is_tmp = false)
  {
    size_t s = sizeof(double);

    if (is_tmp)
    {
      // There is no physical backing for this temporary vector
      handle = to_shared(ctx.logical_data(shape_of<slice<double>>(1)));
    }
    else
    {
      h_addr.reset(new double);
      cuda_safe_call(cudaHostRegister(h_addr.get(), s, cudaHostRegisterPortable));
      handle = to_shared(ctx.logical_data(make_slice(h_addr.get(), 1)));
    }
  }

  scalar(scalar&&)            = default;
  scalar& operator=(scalar&&) = default;

  // Copy constructor
  scalar(const scalar& a)
  {
    handle = to_shared(ctx.logical_data(shape_of<slice<double>>(1)));

    ctx.task(handle->write(), a.handle->read())->*[](cudaStream_t stream, auto dthis, auto da) {
      // There are likely much more efficient ways.
      cuda_safe_call(
        cudaMemcpyAsync(dthis.data_handle(), da.data_handle(), sizeof(double), cudaMemcpyDeviceToDevice, stream));
    };
  }

  scalar operator/(scalar const& rhs) const
  {
    // Submit a task that computes this/rhs
    scalar res(true);
    ctx.task(handle->read(), rhs.handle->read(), res.handle->write())
        ->*[](cudaStream_t stream, auto da, auto db, auto dres) {
              scalar_div<<<1, 1, 0, stream>>>(da.data_handle(), db.data_handle(), dres.data_handle());
            };

    return res;
  }

  // this += rhs
  scalar& operator+=(const scalar& rhs)
  {
    ctx.task(handle->rw(), rhs.handle->read())->*[](cudaStream_t stream, auto dthis, auto drhs) {
      scalar_add<<<1, 1, 0, stream>>>(dthis.data_handle(), drhs.data_handle());
    };

    return *this;
  }

  scalar operator-() const
  {
    // Submit a task that computes -s
    scalar res(true);
    ctx.task(handle->read(), res.handle->write())->*[](cudaStream_t stream, auto dthis, auto dres) {
      scalar_minus<<<1, 1, 0, stream>>>(dthis.data_handle(), dres.data_handle());
    };

    return res;
  }

  // Get value on the host
  double get_value()
  {
    double val;
    ctx.task(exec_place::host, handle->read())->*[&val](cudaStream_t stream, auto ds) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      val = ds(0);
    };

    return val;
  }

  mutable std::shared_ptr<logical_data<slice<double>>> handle;
  std::unique_ptr<double> h_addr;
};

class scalar DOT(vector& a, class vector& b)
{
  assert(a.nblocks == b.nblocks);
  scalar global_res(true);

  // Loop over all blocks,
  for (int bid = 0; bid < a.nblocks; bid++)
  {
    scalar res(true);

    // Note that it works even if a.handle == b.handle because they have the same acces mode
    ctx.task(a.handles[bid]->read(), b.handles[bid]->read(), res.handle->write())
        ->*[](cudaStream_t stream, auto da, auto db, auto dres) {
              cuda_safe_call(cublasSetStream(cublas_handle, stream));
              cuda_safe_call(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
              cuda_safe_call(
                cublasDdot(cublas_handle, da.extent(0), da.data_handle(), 1, db.data_handle(), 1, dres.data_handle()));
            };

    if (bid == 0)
    {
      // First access requires an assigment because it was not initialized
      global_res = std::move(res);
    }
    else
    {
      global_res += res;
    }
  }

  return global_res;
};

// Y = Y + alpha * X
void AXPY(const class scalar& alpha, class vector& x, class vector& y)
{
  assert(x.N == y.N);
  assert(x.nblocks == y.nblocks);

  for (int b = 0; b < x.nblocks; b++)
  {
    ctx.task(alpha.handle->read(), x.handles[b]->read(), y.handles[b]->rw())
        ->*
      [](cudaStream_t stream, auto dalpha, auto dx, auto dy) {
        auto nx = dx.extent(0);
        cuda_safe_call(cublasSetStream(cublas_handle, stream));
        cuda_safe_call(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
        cuda_safe_call(cublasDaxpy(cublas_handle, nx, dalpha.data_handle(), dx.data_handle(), 1, dy.data_handle(), 1));
      };
  }
};

// Y = alpha*Y + X
void SCALE_AXPY(const scalar& alpha, const class vector& x, class vector& y)
{
  assert(x.N == y.N);
  assert(x.nblocks == y.nblocks);

  for (int b = 0; b < x.nblocks; b++)
  {
    ctx.task(alpha.handle->read(), x.handles[b]->read(), y.handles[b]->rw())
        ->*[](cudaStream_t stream, auto dalpha, auto dx, auto dy) {
              cuda_safe_call(cublasSetStream(cublas_handle, stream));

              auto nx = dx.extent(0);

              // Y = alpha Y
              cuda_safe_call(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
              cuda_safe_call(cublasDscal(cublas_handle, nx, dalpha.data_handle(), dy.data_handle(), 1));

              // Y = Y + X
              const double one = 1.0;
              cuda_safe_call(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
              cuda_safe_call(cublasDaxpy(cublas_handle, nx, &one, dx.data_handle(), 1, dy.data_handle(), 1));
            };
  }
};

// y = alpha Ax + beta y
void GEMV(double alpha, class matrix& a, class vector& x, double beta, class vector& y)
{
  assert(a.N == x.N);
  assert(x.N == y.N);

  size_t block_size = x.block_size;
  assert(block_size == y.block_size);

  for (int row_y = 0; row_y < y.nblocks; row_y++)
  {
    for (int row_x = 0; row_x < x.nblocks; row_x++)
    {
      double local_beta = (row_x == 0) ? beta : 1.0;

      // If beta is null, then this is a write only mode
      auto y_mode = local_beta == 0.0 ? access_mode::write : access_mode::rw;

      ctx.task(a.handle->read(), x.handles[row_x]->read(), task_dep<slice<double>>(*(y.handles[row_y].get()), y_mode))
          ->*[alpha, local_beta, row_x, row_y, block_size](cudaStream_t stream, auto da, auto dx, auto dy) {
                auto nx              = dx.extent(0);
                auto ny              = dy.extent(0);
                auto ldA             = da.stride(1);
                const double* Ablock = &da(row_y * block_size, row_x * block_size);

                cuda_safe_call(cublasSetStream(cublas_handle, stream));
                cuda_safe_call(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
                cuda_safe_call(cublasDgemv(
                  cublas_handle,
                  CUBLAS_OP_N,
                  ny,
                  nx,
                  &alpha,
                  Ablock,
                  ldA,
                  dx.data_handle(),
                  1,
                  &local_beta,
                  dy.data_handle(),
                  1));
              };
    }
  }
}

void cg(matrix& A, vector& X, vector& B)
{
  int N = A.N;

  assert(N == X.N);
  assert(N == B.N);

  vector R = B;

  // R = R - A*X
  GEMV(-1.0, A, X, 1.0, R);

  vector P = R;

  // RSOLD = R'*R
  scalar rsold = DOT(R, R);

  int MAXITER = N;

  if (getenv("MAXITER"))
  {
    MAXITER = atoi(getenv("MAXITER"));
  }

  for (int k = 0; k < MAXITER; k++)
  {
    vector Ap(N, P.block_size, true);

    // Ap = A*P
    GEMV(1.0, A, P, 0.0, Ap);

    // alpha = rsold / (p' * Ap);
    scalar alpha = rsold / DOT(P, Ap);

    // x = x + alpha * p;
    AXPY(alpha, P, X);

    // r = r - alpha * Ap;
    AXPY(-alpha, Ap, R);

    // rsnew = r' * r;
    scalar rsnew = DOT(R, R);

    // Read the residual on the CPU, and halt the iterative process if we have converged
    {
      double err;
      ctx.task(exec_place::host, rsnew.handle->read())->*[&err](cudaStream_t stream, auto dres) {
        cuda_safe_call(cudaStreamSynchronize(stream));
        err = sqrt(dres(0));
      };

      if (err < 1e-10)
      {
        // We have converged
        // fprintf(stderr, "Successfully converged (err = %le)\n", err);
        break;
      }
    }

    // p = r + (rsnew / rsold) * p;
    SCALE_AXPY(rsnew / rsold, R, P);

    rsold = std::move(rsnew);
  }
}

int main(int argc, char** argv)
{
  size_t N = 1024;

  if (argc > 1)
  {
    N = atoi(argv[1]);
    fprintf(stderr, "N = %zu\n", N);
  }

  size_t block_size = N / 4;

  if (argc > 2)
  {
    block_size = atoi(argv[2]);
    fprintf(stderr, "block_size = %zu\n", block_size);
  }

  // Do this lazily ?
  cuda_safe_call(cublasCreate(&cublas_handle));

  matrix A(N);
  A.fill([&](int row, int col) {
    return (1.0 / (row + col + 1) + (row == col ? 0.1 : 0.0));
  });

  vector B(N, block_size);
  vector X(N, block_size);

  B.fill([&](int /*unused*/) {
    return 1.0;
  });

  X.fill([&](int /*unused*/) {
    return 0.0;
  });

  cg(A, X, B);

  ctx.finalize();
}
