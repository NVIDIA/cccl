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
 * @brief This example implements a Cholesky decomposition over multiple devices using CUBLAS and CUSOLVER
 *
 * It also illustrates how we can use CUDASTF to allocate temporary data for CUSOLVER in CUDASTF tasks
 */

#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <nvtx3/nvToolsExt.h>

#define TILED

using namespace cuda::experimental::stf;

// Global for the sake of simplicity !
stream_ctx ctx;

/* Get a CUBLAS handle valid on the current device, or initialize it lazily */
cublasHandle_t& get_cublas_handle()
{
  int dev;
  cuda_safe_call(cudaGetDevice(&dev));

  static std::unordered_map<int, cublasHandle_t> cublas_handles;
  auto& result = cublas_handles[dev];
  if (result == cublasHandle_t())
  { // not found, default value inserted
    // Lazy initialization, and save the handle for future use
    cuda_safe_call(cublasCreate(&result));
  }
  return result;
}

/* Get a CUSOLVER handle valid on the current device, or initialize it lazily */
cusolverDnHandle_t& get_cusolver_handle()
{
  int dev;
  cuda_safe_call(cudaGetDevice(&dev));

  static std::unordered_map<int, cusolverDnHandle_t> cusolver_handles;
  auto& result = cusolver_handles[dev];
  if (result == cusolverDnHandle_t())
  { // not found, default value inserted
    // Lazy initialization, and save the handle for future use
    cuda_safe_call(cusolverDnCreate(&result));
  }
  return result;
}

template <typename T>
class matrix
{
public:
  matrix(int NROWS, int NCOLS, int BLOCKSIZE_ROWS, int BLOCKSIZE_COLS, bool is_sym, const char* _symbol = "matrix")
  {
    symbol = _symbol;

    sym_matrix = is_sym;

    m  = NROWS;
    mb = BLOCKSIZE_ROWS;

    n  = NCOLS;
    nb = BLOCKSIZE_COLS;

    assert(m % mb == 0);
    assert(n % nb == 0);

    // cuda_safe_call(cudaMallocHost(&h_array, m*n*sizeof(T)));
    // fprintf(stderr, "Allocating %ld x %ld x %ld = %ld bytes (%f GB) on host for %s\n", m, n, sizeof(T), s,
    //        s / (1024.0 * 1024.0 * 1024.0), _symbol);
    h_array.resize(m * n);
    cuda_safe_call(cudaHostRegister(&h_array[0], h_array.size() * sizeof(T), cudaHostRegisterPortable));

    // Compute the number of blocks
    mt = m / mb;
    nt = n / nb;

    handles.resize(mt * nt);

    for (int colb = 0; colb < nt; colb++)
    {
      int low_rowb = sym_matrix ? colb : 0;
      for (int rowb = low_rowb; rowb < mt; rowb++)
      {
        T* addr_h = get_block_h(rowb, colb);
        auto& h   = handle(rowb, colb);

#ifdef TILED
        // tiles are stored contiguously
        size_t ld = mb;
#else
        size_t ld = m;
#endif
        std::ignore = ld; // work around bug in compiler
        h           = ctx.logical_data(make_slice(addr_h, std::tuple{mb, nb}, ld));
        h.set_symbol(std::string(symbol) + "_" + std::to_string(rowb) + "_" + std::to_string(colb));
      }
    }

    cuda_safe_call(cudaGetDeviceCount(&ndevs));
    for (int a = 1; a * a <= ndevs; a++)
    {
      if (ndevs % a == 0)
      {
        grid_p = a;
        grid_q = ndevs / a;
      }
    }

    assert(grid_p * grid_q == ndevs);

    // std::cout << "FOUND " << ndevs << " DEVICES "
    //          << "p=" << grid_p << " q=" << grid_q << std::endl;
  }

  int get_preferred_devid(int row, int col)
  {
    return (row % grid_p) + (col % grid_q) * grid_p;
  }

  auto& handle(int row, int col)
  {
    return handles[row + col * mt];
  }

  size_t get_index(size_t row, size_t col)
  {
#ifdef TILED
    // Find which tile contains this element
    int tile_row = row / mb;
    int tile_col = col / nb;

    size_t tile_size = mb * nb;

    // Look for the index of the begining of the tile
    size_t tile_start = (tile_row + mt * tile_col) * tile_size;

    // Offset within the tile
    size_t offset = (row % mb) + (col % nb) * mb;

    return tile_start + offset;
#else
    return row + col * m;
#endif
  }

  T* get_block_h(int brow, int bcol)
  {
    size_t index = get_index(brow * mb, bcol * nb);
    return &h_array[index];
  }

  // Fill with func(Matrix*,row, col)
  template <typename Fun>
  void fill(Fun&& fun)
  {
    // Fill blocks by blocks
    for (int colb = 0; colb < nt; colb++)
    {
      int low_rowb = sym_matrix ? colb : 0;
      for (int rowb = low_rowb; rowb < mt; rowb++)
      {
        // Each task fills a block
        ctx.host_launch(handle(rowb, colb).write())->*[this, fun, rowb, colb](auto sA) {
          for (int lcol = 0; lcol < sA.extent(1); lcol++)
          {
            size_t col = lcol + colb * sA.extent(1);
            for (int lrow = 0; lrow < sA.extent(0); lrow++)
            {
              size_t row     = lrow + rowb * sA.extent(0);
              sA(lrow, lcol) = fun(*this, row, col);
            }
          }
        };
      }
    }
  }

  std::vector<T> h_array;
  size_t m; // nrows
  size_t n; // ncols

  // Is this a sym matrix ? (lower assumed)
  bool sym_matrix;

  size_t mb; // block size (rows)
  size_t nb; // block size (cols)

  size_t mt; // number of column blocks
  size_t nt; // number of row blocks

  // abstract data handles
  std::vector<logical_data<slice<double, 2>>> handles;

  const char* symbol;

  // for the mapping
  int ndevs;
  int grid_p, grid_q;
};

void DPOTRF(cublasFillMode_t uplo, class matrix<double>& A, int A_row, int A_col)
{
  auto& Akk    = A.handle(A_row, A_col);
  size_t m_akk = Akk.shape().extent(0);
  // Note that the handle may be different from the actual handle...
  int Lwork_expected;
  cuda_safe_call(cusolverDnDpotrf_bufferSize(get_cusolver_handle(), uplo, m_akk, nullptr, 0, &Lwork_expected));

  auto potrf_buffer = ctx.logical_data<double>(Lwork_expected);
  auto devInfo      = ctx.logical_data(shape_of<slice<int>>(1));

  auto t = ctx.task(Akk.rw(), potrf_buffer.write(), devInfo.write());
  t.set_symbol("DPOTRF");
  t->*[&](cudaStream_t s, auto sAkk, auto buffer, auto info) {
    auto& h = get_cusolver_handle();
    cuda_safe_call(cusolverDnSetStream(h, s));

    cuda_safe_call(cusolverDnDpotrf(
      h,
      uplo,
      sAkk.extent(0),
      sAkk.data_handle(),
      sAkk.stride(1),
      buffer.data_handle(),
      buffer.extent(0),
      info.data_handle()));
  };
}

void DGEMM(
  cublasOperation_t transa,
  cublasOperation_t transb,
  double alpha,
  class matrix<double>& A,
  int A_row,
  int A_col,
  class matrix<double>& B,
  int B_row,
  int B_col,
  double beta,
  class matrix<double>& C,
  int C_row,
  int C_col)
{
  auto t = ctx.task(A.handle(A_row, A_col).read(), B.handle(B_row, B_col).read(), C.handle(C_row, C_col).rw());
  t.set_symbol("DGEMM");
  t->*[&](cudaStream_t s, auto sA, auto sB, auto sC) {
    auto& h = get_cublas_handle();
    cuda_safe_call(cublasSetStream(h, s));

    auto k = (transa == CUBLAS_OP_N) ? sA.extent(1) : sA.extent(0);
    cuda_safe_call(cublasDgemm(
      h,
      transa,
      transb,
      sC.extent(0),
      sC.extent(1),
      k,
      &alpha,
      sA.data_handle(),
      sA.stride(1),
      sB.data_handle(),
      sB.stride(1),
      &beta,
      sC.data_handle(),
      sC.stride(1)));
  };
}

void DSYRK(
  cublasFillMode_t uplo,
  cublasOperation_t trans,
  double alpha,
  class matrix<double>& A,
  int A_row,
  int A_col,
  double beta,
  class matrix<double>& C,
  int C_row,
  int C_col)
{
  auto t = ctx.task(A.handle(A_row, A_col).read(), C.handle(C_row, C_col).rw());
  t.set_symbol("DSYRK");
  t->*[&](cudaStream_t s, auto sA, auto sC) {
    auto& h = get_cublas_handle();
    cuda_safe_call(cublasSetStream(h, s));

    // number of rows of matrix op(A) and C
    auto n = sC.extent(0);

    // number of columns of matrix op(A)
    auto k = (trans == CUBLAS_OP_N) ? sA.extent(1) : sA.extent(0);

    cuda_safe_call(
      cublasDsyrk(h, uplo, trans, n, k, &alpha, sA.data_handle(), sA.stride(1), &beta, sC.data_handle(), sC.stride(1)));
  };
}

void DTRSM(
  cublasSideMode_t side,
  cublasFillMode_t uplo,
  cublasOperation_t transa,
  cublasDiagType_t diag,
  double alpha,
  class matrix<double>& A,
  int A_row,
  int A_col,
  class matrix<double>& B,
  int B_row,
  int B_col)
{
  auto t = ctx.task(A.handle(A_row, A_col).read(), B.handle(B_row, B_col).rw());
  t.set_symbol("DTRSM");
  t->*[&](cudaStream_t s, auto sA, auto sB) {
    auto& h = get_cublas_handle();
    cuda_safe_call(cublasSetStream(h, s));

    cuda_safe_call(cublasDtrsm(
      h,
      side,
      uplo,
      transa,
      diag,
      sB.extent(0),
      sB.extent(1),
      &alpha,
      sA.data_handle(),
      sA.stride(1),
      sB.data_handle(),
      sB.stride(1)));
  };
}

void PDNRM2_HOST(matrix<double>* A, double* result)
{
#ifdef HAVE_DOT
  reserved::dot::set_current_color("red");
#endif

  for (int rowb = 0; rowb < A->mt; rowb++)
  {
    for (int colb = 0; colb < A->nt; colb++)
    {
      ctx.host_launch(A->handle(rowb, colb).read())->*[=](auto sA) {
        double res2 = 0.0;
        for (size_t col = 0; col < sA.extent(1); col++)
        {
          for (size_t row = 0; row < sA.extent(0); row++)
          {
            double v = sA(row, col);
            res2 += v * v;
          }
        }
        *result += res2;
      };
    }
  }
}

void PDPOTRF(matrix<double>& A)
{
#ifdef HAVE_DOT
  reserved::dot::set_current_color("yellow");
#endif

  assert(A.m == A.n);
  assert(A.mt == A.nt);

  int NBLOCKS = A.mt;
  assert(A.mb == A.nb);

  cuda_safe_call(cudaSetDevice(0));

  nvtxRangePushA("SUBMIT_PDPOTRF");
  for (int K = 0; K < NBLOCKS; K++)
  {
    cuda_safe_call(cudaSetDevice(A.get_preferred_devid(K, K)));
    DPOTRF(CUBLAS_FILL_MODE_LOWER, A, K, K);

    for (int row = K + 1; row < NBLOCKS; row++)
    {
      cuda_safe_call(cudaSetDevice(A.get_preferred_devid(row, K)));
      DTRSM(CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 1.0, A, K, K, A, row, K);

      for (int col = K + 1; col < row; col++)
      {
        cuda_safe_call(cudaSetDevice(A.get_preferred_devid(row, col)));
        DGEMM(CUBLAS_OP_N, CUBLAS_OP_T, -1.0, A, row, K, A, col, K, 1.0, A, row, col);
      }

      cuda_safe_call(cudaSetDevice(A.get_preferred_devid(row, row)));
      DSYRK(CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, -1.0, A, row, K, 1.0, A, row, row);
    }
  }
  cuda_safe_call(cudaSetDevice(0));

  nvtxRangePop();
}

// Algorithm from PLASMA
void PDTRSM(cublasSideMode_t side,
            cublasFillMode_t uplo,
            cublasOperation_t trans,
            cublasDiagType_t diag,
            double alpha,
            class matrix<double>& A,
            class matrix<double>& B)
{
  //    std::cout << "[PDTRSM] START B MT " << B.mt << " NT " << B.nt << std::endl;

  if (side == CUBLAS_SIDE_LEFT)
  {
    if (uplo == CUBLAS_FILL_MODE_UPPER)
    {
      // TODO
      assert(0);
      abort();
    }
    else
    {
      //===========================================
      // CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_LOWER / CUBLAS_OP_N
      //===========================================
      if (trans == CUBLAS_OP_N)
      {
        for (int k = 0; k < B.mt; k++)
        {
          double lalpha = k == 0 ? alpha : 1.0;
          for (int n = 0; n < B.nt; n++)
          {
            cuda_safe_call(cudaSetDevice(A.get_preferred_devid(k, k)));
            DTRSM(side, uplo, trans, diag, lalpha, A, k, k, B, k, n);
          }
          for (int m = k + 1; m < B.mt; m++)
          {
            for (int n = 0; n < B.nt; n++)
            {
              cuda_safe_call(cudaSetDevice(A.get_preferred_devid(m, k)));
              DGEMM(CUBLAS_OP_N, CUBLAS_OP_N, -1.0, A, m, k, B, k, n, lalpha, B, m, n);
            }
          }
        }
      }
      //================================================
      // CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_LOWER / CUBLAS_OP_[C|T]
      //================================================
      else
      {
        for (int k = 0; k < B.mt; k++)
        {
          double lalpha = k == 0 ? alpha : 1.0;
          for (int n = 0; n < B.nt; n++)
          {
            cuda_safe_call(cudaSetDevice(A.get_preferred_devid(B.mt - k - 1, B.mt - k - 1)));
            DTRSM(side, uplo, trans, diag, lalpha, A, B.mt - k - 1, B.mt - k - 1, B, B.mt - k - 1, n);
          }
          for (int m = k + 1; m < B.mt; m++)
          {
            for (int n = 0; n < B.nt; n++)
            {
              cuda_safe_call(cudaSetDevice(A.get_preferred_devid(B.mt - k - 1, B.mt - 1 - m)));
              DGEMM(
                trans, CUBLAS_OP_N, -1.0, A, B.mt - k - 1, B.mt - 1 - m, B, B.mt - k - 1, n, lalpha, B, B.mt - 1 - m, n);
            }
          }
        }
      }
    }
  }
  else
  {
    // TODO
    abort();
  }
  cuda_safe_call(cudaSetDevice(0));
  //    std::cout << "[PDTRSM] END" << std::endl;
}

void PDPOTRS(matrix<double>& A, class matrix<double>& B, cublasFillMode_t uplo)
{
#ifdef HAVE_DOT
  reserved::dot::set_current_color("green");
#endif

  //    std::cout << "[PDPOTRS] START" << std::endl;
  // Call the parallel functions.
  PDTRSM(
    CUBLAS_SIDE_LEFT, uplo, uplo == CUBLAS_FILL_MODE_UPPER ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 1.0, A, B);

#ifdef HAVE_DOT
  reserved::dot::set_current_color("darkgreen");
#endif

  PDTRSM(
    CUBLAS_SIDE_LEFT, uplo, uplo == CUBLAS_FILL_MODE_UPPER ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 1.0, A, B);
  //    std::cout << "[PDPOTRS] END" << std::endl;
}

/*****************************************************************************
 * Parallel tile matrix-matrix
 *multiplication.
 * @see plasma_omp_dgemm
 ******************************************************************************/
void PDGEMM(cublasOperation_t transa,
            cublasOperation_t transb,
            double alpha,
            class matrix<double>& A,
            class matrix<double>& B,
            double beta,
            class matrix<double>& C)
{
#ifdef HAVE_DOT
  reserved::dot::set_current_color("blue");
#endif

  for (int m = 0; m < C.mt; m++)
  {
    for (int n = 0; n < C.nt; n++)
    {
      //=========================================
      // alpha*A*B does not contribute; scale C
      //=========================================
      int inner_k = transa == CUBLAS_OP_N ? A.n : A.m;
      if (alpha == 0.0 || inner_k == 0)
      {
        DGEMM(transa, transb, alpha, A, 0, 0, B, 0, 0, beta, C, m, n);
      }
      else if (transa == CUBLAS_OP_N)
      {
        //================================
        // CUBLAS_OP_N / CUBLAS_OP_N
        //================================
        if (transb == CUBLAS_OP_N)
        {
          for (int k = 0; k < A.nt; k++)
          {
            double zbeta = k == 0 ? beta : 1.0;
            DGEMM(transa, transb, alpha, A, m, k, B, k, n, zbeta, C, m, n);
          }
        }
        //=====================================
        // CUBLAS_OP_N / CUBLAS_OP_T
        //=====================================
        else
        {
          for (int k = 0; k < A.nt; k++)
          {
            double zbeta = k == 0 ? beta : 1.0;
            DGEMM(transa, transb, alpha, A, m, k, B, n, k, zbeta, C, m, n);
          }
        }
      }
      else
      {
        //=====================================
        // CUBLAS_OP_T / CUBLAS_OP_N
        //=====================================
        if (transb == CUBLAS_OP_N)
        {
          for (int k = 0; k < A.mt; k++)
          {
            double zbeta = k == 0 ? beta : 1.0;
            DGEMM(transa, transb, alpha, A, k, m, B, k, n, zbeta, C, m, n);
          }
        }
        //==========================================
        // CUBLAS_OP_T / CUBLAS_OP_T
        //==========================================
        else
        {
          for (int k = 0; k < A.mt; k++)
          {
            double zbeta = k == 0 ? beta : 1.0;
            DGEMM(transa, transb, alpha, A, k, m, B, n, k, zbeta, C, m, n);
          }
        }
      }
    }
  }
}

int main(int argc, char** argv)
{
  int N  = 1024;
  int NB = 128;

  if (argc > 1)
  {
    N = atoi(argv[1]);
  }

  if (argc > 2)
  {
    NB = atoi(argv[2]);
  }

  int check_result = 1;
  if (getenv("CHECK_RESULT"))
  {
    check_result = atoi(getenv("CHECK_RESULT"));
  }

  assert(N % NB == 0);

  // Set up CUBLAS and CUSOLVER
  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  cuda_safe_call(cudaSetDevice(0));

  matrix<double> A(N, N, NB, NB, true, "A");
  matrix<double> Aref(N, N, NB, NB, false, "Aref");

  // (Hilbert matrix + 2*N*Id) to have a diagonal dominant matrix
  auto hilbert = [](matrix<double>& mat, int row, int col) {
    return 1.0 / (col + row + 1.0) + 2.0 * mat.n * (col == row);
  };

  if (check_result)
  {
    Aref.fill(hilbert);
  }

  A.fill(hilbert);

  /* Right-hand side */
  matrix<double> B_potrs(N, 1, NB, 1, false, "B");
  matrix<double> Bref_potrs(N, 1, NB, 1, false, "Bref");

  if (check_result)
  {
    auto rhs_vals = [](matrix<double>&, int row, int /*col*/) {
      return 1.0 * (row + 1);
    };
    B_potrs.fill(rhs_vals);
    Bref_potrs.fill(rhs_vals);
  }

  //    // Compute ||Bref||
  double Bref_nrm2 = 0.0;
  double res_nrm2  = 0.0;

  if (check_result)
  {
    PDNRM2_HOST(&Bref_potrs, &Bref_nrm2);
  }

  cudaEvent_t startEvent_pdpotrf, stopEvent_pdpotrf;
  float milliseconds_pdpotrf = 0;

  //    for (int row = 0; row < A.mt; row++)
  //    {
  //        for (int col = 0; col <= row; col++)
  //        {
  //            cuda_safe_call(cudaSetDevice(A.get_preferred_devid(row, col)));
  //            NOOP(A, row, col);
  //        }
  //    }

  cuda_safe_call(cudaEventCreate(&startEvent_pdpotrf));
  cuda_safe_call(cudaEventCreate(&stopEvent_pdpotrf));

  cuda_safe_call(cudaEventRecord(startEvent_pdpotrf, ctx.task_fence()));

  PDPOTRF(A);

  cuda_safe_call(cudaEventRecord(stopEvent_pdpotrf, ctx.task_fence()));

  /*
   *  POTRS
   */

  if (check_result)
  {
    // Solve AX = B and put the result in B
    PDPOTRS(A, B_potrs, CUBLAS_FILL_MODE_LOWER);

    // Compute (AX - B)
    // Bref = (Aref*B - Bref)
    PDGEMM(CUBLAS_OP_N, CUBLAS_OP_N, 1.0, Aref, B_potrs, -1.0, Bref_potrs);

    // Compute ||AX - B|| = ||Bref||
    PDNRM2_HOST(&Bref_potrs, &res_nrm2);
  }

  ctx.finalize();

  cuda_safe_call(cudaEventElapsedTime(&milliseconds_pdpotrf, startEvent_pdpotrf, stopEvent_pdpotrf));

  double gflops_pdpotrf = 1.0 / 3.0 * ((double) N * (double) N * (double) N) / (1000000000.0);
  std::cout << "[PDPOTRF] ELAPSED: " << milliseconds_pdpotrf
            << " ms, GFLOPS: " << gflops_pdpotrf / (milliseconds_pdpotrf / 1000.0) << std::endl;

  if (check_result)
  {
    if (const auto residual = sqrt(res_nrm2) / sqrt(Bref_nrm2); residual >= 0.01)
    {
      std::cerr << "[POTRS] ||AX - B|| : " << sqrt(res_nrm2) << '\n';
      std::cerr << "[POTRS] ||B|| : " << sqrt(Bref_nrm2) << '\n';
      std::cerr << "[POTRS] RESIDUAL (||AX - B||/||B||) : " << residual << '\n';
      assert(!"Algorithm did not converge.");
    }
  }
}
