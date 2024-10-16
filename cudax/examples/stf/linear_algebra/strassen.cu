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
 * @brief Strassen matrix multiplication algorithm
 *
 * This demonstrates how CUDASTF helps combining many interdependant tasks and
 * deal with temporary data.
 */

#include <cuda/experimental/stf.cuh>

static const size_t BLOCKSIZE = 1024;

using namespace cuda::experimental::stf;

using logical_matrix = logical_data<slice<double, 2>>;

inline size_t get_m(logical_matrix& s)
{
  return s.shape().extent(0);
}

inline size_t get_n(logical_matrix& s)
{
  return s.shape().extent(1);
}

// XXX global for the sake of simplicity, yet ...
static std::vector<cublasHandle_t> cublas_handle;

cublasHandle_t get_cublas_handle()
{
  int dev;
  cuda_safe_call(cudaGetDevice(&dev));
  return cublas_handle[dev];
}

// C = AB
void MULT_CLASSIC(context& ctx, logical_matrix& A, logical_matrix& B, logical_matrix& C)
{
  ctx.task(A.read(), B.read(), C.write()).set_symbol("MULT")->*[](cudaStream_t s, auto a, auto b, auto c) {
    cuda_safe_call(cublasSetStream(get_cublas_handle(), s));

    size_t N = a.extent(0);

    const double zero = 0.0;
    const double one  = 1.0;
    cuda_safe_call(cublasDgemm(
      get_cublas_handle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      N,
      N,
      N,
      &one,
      a.data_handle(),
      a.stride(1),
      b.data_handle(),
      b.stride(1),
      &zero,
      c.data_handle(),
      c.stride(1)));
  };
}

// A = A + alpha B
template <typename T>
__global__ void add_kernel(int m, int n, T* A, int ld_A, T alpha, const T* B, int ld_B)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x)
  {
    for (int idy = threadIdx.y + blockIdx.y * blockDim.y; idy < m; idy += blockDim.y * gridDim.y)
    {
      A[idy + idx * ld_A] += alpha * B[idy + idx * ld_B];
    }
  }
}

// Compute A = A + B
template <typename T>
void ADD(context& ctx, logical_matrix& A, T alpha, logical_matrix& B)
{
  ctx.task(A.rw(), B.read()).set_symbol("ADD")->*[&](cudaStream_t s, auto a, auto b) {
    int m_A = a.extent(0);
    int n_A = a.extent(1);

    int ld_A = a.stride(1);
    int ld_B = b.stride(1);

    T* addr_A       = a.data_handle();
    const T* addr_B = b.data_handle();

    add_kernel<<<16, 16, 0, s>>>(m_A, n_A, addr_A, ld_A, alpha, addr_B, ld_B);
  };
}

template <typename T>
__global__ void copy_kernel(int m, int n, const T* src, int ld_src, T* dst, int ld_dst)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n; idx += blockDim.x * gridDim.x)
  {
    for (int idy = threadIdx.y + blockIdx.y * blockDim.y; idy < m; idy += blockDim.y * gridDim.y)
    {
      dst[idy + idx * ld_dst] = src[idy + idx * ld_src];
    }
  }
}

// row and col = 0 or 1
template <typename T>
void COPY_TO_SUBMATRIX(context& ctx, logical_data<slice<T, 2>>& A, logical_data<slice<T, 2>>& subA, int row, int col)
{
  // To copy to a subset, this is a write only access, so that we did not need a valid copy for subA before ...
  ctx.task(A.read(), subA.write()).set_symbol("COPY_TO")->*[&](cudaStream_t s, auto a, auto subA) {
    int ld_A             = a.stride(1);
    int ld_subA          = subA.stride(1);
    int m_subA           = subA.extent(0);
    int n_subA           = subA.extent(1);
    T* addr_subA         = subA.data_handle();
    const T* addr_A_base = a.data_handle();
    const T* addr_A      = addr_A_base + row * m_subA + col * n_subA * ld_A;

    // subA = A_row,col
    copy_kernel<<<16, 16, 0, s>>>(m_subA, n_subA, addr_A, ld_A, addr_subA, ld_subA);
  };
}

template <typename T>
void COPY_FROM_SUBMATRICES(context& ctx, logical_data<slice<T, 2>>& A, logical_data<slice<T, 2>> subA[2][2])
{
  // To copy to a subset, this is a write only access, so that we did not need a valid copy for subA before ...
  // When copying from a subset to the whole matrix, we need a RW because we only modify a part of the matrix
  ctx.task(A.write(), subA[0][0].read(), subA[0][1].read(), subA[1][0].read(), subA[1][1].read()).set_symbol("COPY_FROM")
      ->*[&](cudaStream_t s, auto a, auto a00, auto a01, auto a10, auto a11) {
            int ld_A       = a.stride(1);
            T* addr_A_base = a.data_handle();

            for (int col = 0; col < 2; col++)
            {
              for (int row = 0; row < 2; row++)
              {
                auto& subA         = col == 0 ? (row == 0 ? a00 : a10) : (row == 0 ? a01 : a11);
                int m_subA         = subA.extent(0);
                int n_subA         = subA.extent(1);
                int ld_subA        = subA.stride(1);
                const T* addr_subA = subA.data_handle();
                T* addr_A          = addr_A_base + row * m_subA + col * n_subA * ld_A;

                // A_row,col= subA
                copy_kernel<<<16, 16, 0, s>>>(m_subA, n_subA, addr_subA, ld_subA, addr_A, ld_A);
              }
            }
          };
}

template <typename T>
void COPY_MATRIX(context& ctx, logical_data<slice<T, 2>>& dst, logical_data<slice<T, 2>>& src)
{
  // This is a write only access, so that we did not need a valid copy for subA before ...
  ctx.task(dst.write(), src.read()).set_symbol("COPY")->*[&](cudaStream_t s, auto d_dst, auto d_src) {
    int ld_src = d_dst.stride(1);
    int ld_dst = d_src.stride(1);

    auto m = d_src.extent(0);
    assert(m == d_dst.extent(0));

    auto n = d_src.extent(1);
    assert(n == d_dst.extent(1));

    const T* addr_src = d_src.data_handle();
    T* addr_dst       = d_dst.data_handle();

    copy_kernel<<<16, 16, 0, s>>>(m, n, addr_src, ld_src, addr_dst, ld_dst);
  };
}

void MULT(context& ctx, logical_matrix& A, logical_matrix& B, logical_matrix& C);

void MULT_REC_NAIVE(context& ctx, logical_matrix& A, logical_matrix& B, logical_matrix& C)
{
  logical_matrix subA[2][2], subB[2][2], subC[2][2];

  size_t N = get_m(A);

  assert(get_m(A) == get_n(A));
  assert(get_m(B) == get_n(B));
  assert(get_m(C) == get_n(C));

  assert(N % 2 == 0);

  size_t half_N = N / 2;

  // These are TMP data which don't have a valid copy yet
  for (int col = 0; col < 2; col++)
  {
    for (int row = 0; row < 2; row++)
    {
      subA[row][col] = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
      subB[row][col] = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
      subC[row][col] = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

      COPY_TO_SUBMATRIX(ctx, A, subA[row][col], row, col);
      COPY_TO_SUBMATRIX(ctx, B, subB[row][col], row, col);
    }
  }

  for (int col = 0; col < 2; col++)
  {
    for (int row = 0; row < 2; row++)
    {
      for (int k = 0; k < 2; k++)
      {
        auto Ck = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
        MULT(ctx, subA[row][k], subB[k][col], Ck);

        ADD(ctx, subC[row][col], 1.0, Ck);
      }

      // C_row,col = subC[row][col]
      COPY_FROM_SUBMATRICES(ctx, C, subC);
    }
  }
}

void MULT_STRASSEN(context& ctx, logical_matrix& A, logical_matrix& B, logical_matrix& C)
{
  /*
   *  STRASSEN ALGORITHM
   *
   *   M1 = (A00 + A11)(B00 + B11)
   *   M2 = (A10 + A11)B00
   *   M3 = A00(B01 - B11)
   *   M4 = A11(B10 - B00)
   *   M5 = (A00 + A01)B11
   *   M6 = (A10 - A00)(B00 + B01)
   *   M7 = (A01 - A11)(B10 + B11)
   *
   *   C00 = M1 + M4 - M5 + M7
   *   C01 = M3 + M5
   *   C10 = M2 + M4
   *   C11 = M1 - M2 + M3 + M6
   *
   */
  size_t N = get_m(A);
  assert(N % 2 == 0);
  size_t half_N = N / 2;

  logical_matrix subA[2][2], subB[2][2], subC[2][2];
  auto M1 = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
  auto M2 = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
  auto M3 = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
  auto M4 = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
  auto M5 = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
  auto M6 = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
  auto M7 = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

  assert(get_m(A) == get_n(A));
  assert(get_m(B) == get_n(B));
  assert(get_m(C) == get_n(C));

  // These are TMP data which don't have a valid copy yet
  for (int col = 0; col < 2; col++)
  {
    for (int row = 0; row < 2; row++)
    {
      subA[row][col] = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
      subB[row][col] = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
      subC[row][col] = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

      COPY_TO_SUBMATRIX(ctx, A, subA[row][col], row, col);
      COPY_TO_SUBMATRIX(ctx, B, subB[row][col], row, col);
    }
  }

  // M1 = (A00 + A11)(B00 + B11)
  {
    auto left  = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N)),
         right = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

    COPY_MATRIX(ctx, left, subA[0][0]);
    ADD(ctx, left, 1.0, subA[1][1]);

    COPY_MATRIX(ctx, right, subB[0][0]);
    ADD(ctx, right, 1.0, subB[1][1]);

    MULT(ctx, left, right, M1);
  }

  // M2 = (A10 + A11)B00
  {
    auto left = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

    COPY_MATRIX(ctx, left, subA[1][0]);
    ADD(ctx, left, 1.0, subA[1][1]);

    MULT(ctx, left, subB[0][0], M2);
  }

  // M3 = A00(B01 - B11)
  {
    auto right = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

    COPY_MATRIX(ctx, right, subB[0][1]);
    ADD(ctx, right, -1.0, subB[1][1]);

    MULT(ctx, subA[0][0], right, M3);
  }

  // M4 = A11(B10 - B00)
  {
    auto right = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

    COPY_MATRIX(ctx, right, subB[1][0]);
    ADD(ctx, right, -1.0, subB[0][0]);

    MULT(ctx, subA[1][1], right, M4);
  }

  // M5 = (A00 + A01)B11
  {
    auto left = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

    COPY_MATRIX(ctx, left, subA[0][0]);
    ADD(ctx, left, 1.0, subA[0][1]);

    MULT(ctx, left, subB[1][1], M5);
  }

  // M6 = (A10 - A00)(B00 + B01)
  {
    auto left  = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N)),
         right = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

    COPY_MATRIX(ctx, left, subA[1][0]);
    ADD(ctx, left, -1.0, subA[1][1]);

    COPY_MATRIX(ctx, right, subB[0][0]);
    ADD(ctx, right, 1.0, subB[0][1]);

    MULT(ctx, left, right, M6);
  }

  // M7 = (A01 - A11)(B10 + B11)
  {
    auto left  = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));
    auto right = ctx.logical_data(shape_of<slice<double, 2>>(half_N, half_N));

    COPY_MATRIX(ctx, left, subA[0][1]);
    ADD(ctx, left, -1.0, subA[1][1]);

    COPY_MATRIX(ctx, right, subB[1][0]);
    ADD(ctx, right, 1.0, subB[1][1]);

    MULT(ctx, left, right, M7);
  }

  // C00 = M1 + M4 - M5 + M7
  COPY_MATRIX(ctx, subC[0][0], M1);
  ADD(ctx, subC[0][0], 1.0, M4);
  ADD(ctx, subC[0][0], -1.0, M5);
  ADD(ctx, subC[0][0], -1.0, M5);
  ADD(ctx, subC[0][0], 1.0, M7);

  // C01 = M3 + M5
  COPY_MATRIX(ctx, subC[0][1], M3);
  ADD(ctx, subC[0][1], 1.0, M5);

  // C10 = M2 + M4
  COPY_MATRIX(ctx, subC[1][0], M2);
  ADD(ctx, subC[1][0], 1.0, M4);

  // C11 = M1 - M2 + M3 + M6
  COPY_MATRIX(ctx, subC[1][1], M1);
  ADD(ctx, subC[1][1], -1.0, M2);
  ADD(ctx, subC[1][1], 1.0, M3);
  ADD(ctx, subC[1][1], 1.0, M6);

  // Write back subsets of C to C
  COPY_FROM_SUBMATRICES(ctx, C, subC);
}

void MULT(context& ctx, logical_matrix& A, logical_matrix& B, logical_matrix& C)
{
  size_t N = get_m(A);

  if (N <= BLOCKSIZE)
  {
    MULT_CLASSIC(ctx, A, B, C);
  }
  else
  {
    // MULT_REC_NAIVE(ctx, A, B, C);
    MULT_STRASSEN(ctx, A, B, C);
  }
}

void strassen_test(context& ctx, size_t N)
{
  double* A = new double[N * N];
  double* B = new double[N * N];
  double* C = new double[N * N];

  int ldA = N;
  int ldB = N;
  int ldC = N;

  cuda_safe_call(cudaHostRegister(A, N * N * sizeof(double), cudaHostRegisterPortable));
  cuda_safe_call(cudaHostRegister(B, N * N * sizeof(double), cudaHostRegisterPortable));
  cuda_safe_call(cudaHostRegister(C, N * N * sizeof(double), cudaHostRegisterPortable));

  for (int col = 0; col < N; col++)
  {
    for (int row = 0; row < N; row++)
    {
      A[row + N * col] = 1.0;
      B[row + N * col] = -1.0;
      C[row + N * col] = 0.0;
    }
  }

  auto descA = ctx.logical_data(make_slice(A, std::tuple{N, N}, ldA)),
       descB = ctx.logical_data(make_slice(B, std::tuple{N, N}, ldB)),
       descC = ctx.logical_data(make_slice(C, std::tuple{N, N}, ldC));
  descA.set_symbol("A");
  descB.set_symbol("B");
  descC.set_symbol("C");

  std::chrono::steady_clock::time_point start, stop;

  ctx.host_launch(descC.read())->*[&](auto /* ignored */) {
    start = std::chrono::steady_clock::now();
  };

  MULT(ctx, descA, descB, descC);

  ctx.host_launch(descC.read())->*[&](auto /* ignored */) {
    stop = std::chrono::steady_clock::now();
  };

  ctx.finalize();

  std::chrono::duration<double> duration = stop - start;
  fprintf(stderr, "Elapsed: %.2lf ms\n", duration.count() * 1000.0);
}

int main(int argc, char** argv)
{
  long N = 2 * BLOCKSIZE;

  if (argc > 1)
  {
    N = atoi(argv[1]);
  }

  bool use_graphs = false;
  if (argc > 2)
  {
    use_graphs = (atoi(argv[2]) > 0);
  }

  // Set up CUBLAS
  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));
  cublas_handle.resize(ndevs);
  for (int d = 0; d < ndevs; d++)
  {
    cuda_safe_call(cudaSetDevice(d));
    cuda_safe_call(cublasCreate(&cublas_handle[d]));
  }

  cuda_safe_call(cudaSetDevice(0));

  context ctx;
  if (use_graphs)
  {
    ctx = graph_ctx();
  }

  strassen_test(ctx, N);
}
