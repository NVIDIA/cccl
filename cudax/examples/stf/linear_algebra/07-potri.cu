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
 * @brief This example implements the POTRI matrix inversion algorithm over multiple devices
 *
 *
 */
#include <cuda/experimental/stf.cuh>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <nvtx3/nvToolsExt.h>

#define TILED

using namespace cuda::experimental::stf;

stream_ctx ctx;

static std::unordered_map<int, cublasHandle_t> cublas_handles;
static std::unordered_map<int, cusolverDnHandle_t> cusolver_handles;

/* Get a CUBLAS handle valid on the current device, or initialize it lazily */
cublasHandle_t& get_cublas_handle()
{
  int dev = cuda_try<cudaGetDevice>();

  auto& result = cublas_handles[dev];
  if (result == cublasHandle_t())
  { // not found, default value inserted
    // Lazy initialization, and save the handle for future use
    cuda_try(cublasCreate(&result));
  }
  return result;
}

/* Get a CUSOLVER handle valid on the current device, or initialize it lazily */
cusolverDnHandle_t& get_cusolver_handle()
{
  int dev;
  cuda_try(cudaGetDevice(&dev));

  auto& result = cusolver_handles[dev];
  if (result == cusolverDnHandle_t())
  { // not found, default value inserted
    // Lazy initialization, and save the handle for future use
    cuda_try(cusolverDnCreate(&result));
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

    size_t s = m * n * sizeof(T);
    // cuda_try(cudaMallocHost(&h_array, m*n*sizeof(T)));
    // fprintf(stderr, "Allocating %ld x %ld x %ld = %ld bytes (%f GB) on host for %s\n", m, n, sizeof(T), s,
    //        s / (1024.0 * 1024.0 * 1024.0), _symbol);
    h_array = (T*) malloc(s);
    assert(h_array);
    cuda_try(cudaHostRegister(h_array, s, cudaHostRegisterPortable));
    // cuda_try(cudaMalloc(&d_array, m*n*sizeof(T)));

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
        auto& h   = get_handle(rowb, colb);

#ifdef TILED
        // tiles are stored contiguously
        size_t ld = mb;
#else
        size_t ld = m;
#endif
        std::ignore = ld; // work around compiler bug
        h           = ctx.logical_data(make_slice(addr_h, std::tuple{mb, nb}, ld));
        h.set_symbol(std::string(symbol) + "_" + std::to_string(rowb) + "_" + std::to_string(colb));
        h.set_write_back(false);
      }
    }

    cuda_try(cudaGetDeviceCount(&ndevs));
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

  auto& get_handle(int row, int col)
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
    nvtxRangePushA("FILL");
    // Fill blocks by blocks
    for (int colb = 0; colb < nt; colb++)
    {
      int low_rowb = sym_matrix ? colb : 0;
      for (int rowb = low_rowb; rowb < mt; rowb++)
      {
        // Each task fills a block
        auto& h   = get_handle(rowb, colb);
        int devid = get_preferred_devid(rowb, colb);

        ctx.parallel_for(exec_place::device(devid), h.shape(), h.write()).set_symbol("INIT")->*
          [=] _CCCL_DEVICE(size_t lrow, size_t lcol, auto sA) {
            size_t row     = lrow + rowb * sA.extent(0);
            size_t col     = lcol + colb * sA.extent(1);
            sA(lrow, lcol) = fun(row, col);
          };
      }
    }
    nvtxRangePop();
  }

  // Print blocks
  void print()
  {
    // print blocks by blocks
    for (int colb = 0; colb < nt; colb++)
    {
      int low_rowb = sym_matrix ? colb : 0;
      for (int rowb = low_rowb; rowb < mt; rowb++)
      {
        // Each task fills a block
        ctx.host_launch(get_handle(rowb, colb).read())->*[=](auto sA) {
          for (int lcol = 0; lcol < sA.extent(1); lcol++)
          {
            size_t col = lcol + colb * sA.extent(1);
            for (int lrow = 0; lrow < sA.extent(0); lrow++)
            {
              size_t row = lrow + rowb * sA.extent(0);

              fprintf(stderr, "%d,%d : %le\n", row, col, sA(lrow, lcol));
            }
          }
        };
      }
    }
  }

  T* h_array;
  T* d_array;
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

void DPOTRF(cublasFillMode_t uplo, matrix<double>& A, int A_row, int A_col)
{
  auto& Akk    = A.get_handle(A_row, A_col);
  size_t m_akk = Akk.shape().extent(0);
  // Note that the handle may be different from the actual handle...
  int Lwork_expected;
  cuda_safe_call(cusolverDnDpotrf_bufferSize(get_cusolver_handle(), uplo, m_akk, nullptr, 0, &Lwork_expected));

  auto potrf_buffer = ctx.logical_data(shape_of<slice<double>>(Lwork_expected));
  potrf_buffer.set_allocator(ctx.get_default_allocator());

  auto devInfo = ctx.logical_data(shape_of<slice<int>>(1));

  auto t = ctx.task(Akk.rw(), potrf_buffer.write(), devInfo.write());
  t.set_symbol("DPOTRF");
  t->*[&](auto s, auto sAkk, auto buffer, auto info) {
    auto& h = get_cusolver_handle();
    cuda_try(cusolverDnSetStream(h, s));

    cuda_try(cusolverDnDpotrf(
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

void DTRTRI(cublasFillMode_t uplo, cublasDiagType_t diag, matrix<double>& A, int A_row, int A_col)
{
  // Preallocate a buffer used by CUSOLVER
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  int64_t m_a00 = A.mb;
  assert(A.mb == A.nb);

  cuda_try(cusolverDnXtrtri_bufferSize(
    get_cusolver_handle(),
    uplo,
    diag,
    m_a00,
    CUDA_R_64F /* DTRTRI */,
    nullptr,
    m_a00,
    &workspaceInBytesOnDevice,
    &workspaceInBytesOnHost));

  // We don't support allocating buffers of 0 bytes ... XXX
  if (workspaceInBytesOnHost == 0)
  {
    workspaceInBytesOnHost = 8;
  }

  auto d_buffer = ctx.logical_data(shape_of<slice<char>>(workspaceInBytesOnDevice));
  auto h_buffer = ctx.logical_data(shape_of<slice<char>>(workspaceInBytesOnHost));
  d_buffer.set_allocator(ctx.get_default_allocator());
  h_buffer.set_allocator(ctx.get_default_allocator());

  auto devInfo = ctx.logical_data(shape_of<slice<int>>(1));

  auto t =
    ctx.task(A.get_handle(A_row, A_col).rw(), d_buffer.write(), h_buffer.write(data_place::managed), devInfo.write());
  t.set_symbol("DTRTRI");
  t->*[&](auto s, auto sA, auto dbuffer, auto hbuffer, auto info) {
    auto& h = get_cusolver_handle();
    cuda_try(cusolverDnSetStream(h, s));

    // DTRTRI(...)
    cuda_try(cusolverDnXtrtri(
      h,
      uplo,
      diag,
      sA.extent(0),
      CUDA_R_64F /* DTRTRI */,
      sA.data_handle(),
      sA.stride(1),
      (double*) dbuffer.data_handle(),
      workspaceInBytesOnDevice,
      (double*) hbuffer.data_handle(),
      workspaceInBytesOnHost,
      info.data_handle()));
  };
}

/*
 * Note: this code was taken from CUSOLVER
 *
 * SLACPY copies all or part of a two-dimensional matrix A to another matrix B.
 *
 *  up     up_and_lo
 *  1          0          upper triangle, including diagonal
 *  0          0          lower triangle, including diagonal
 *  ?          1          whole matrix
 *
 * configuration:
 *   dim3 grids( m/VEC, m/BY )
 *   dim3 threads(VEC,BY)
 */
template <typename T_ELEM_SRC, typename T_ELEM_DST, int VEC_LOG, int BY_LOG>
__global__ void __launch_bounds__(1 << (VEC_LOG + BY_LOG))
  lacpy_kernel(int m, int n, const T_ELEM_SRC* A, size_t lda, T_ELEM_DST* B, size_t ldb, int up, int up_and_lo)
{
  const int VEC = (1 << VEC_LOG);
  const int BY  = (1 << BY_LOG);

  const int inx = threadIdx.x;
  const int iny = threadIdx.y;

  const int ibx = blockIdx.x * VEC;
  const int iby = blockIdx.y * BY;

  const int i = ibx + inx;
  const int j = iby + iny;

  if (ibx >= m)
  {
    return;
  }
  if (iby >= n)
  {
    return;
  }

  T_ELEM_SRC Areg = T_ELEM_SRC(0);

  if (up_and_lo)
  {
    /*
     * copy whole matrix
             DO 60 J = 1, N
                DO 50 I = 1, M
                   B( I, J ) = A( I, J )
       50       CONTINUE
       60    CONTINUE
    */
    if ((i < m) && (j < n))
    {
      Areg           = A[i + j * lda];
      B[i + j * ldb] = T_ELEM_DST(Areg);
    }
    return;
  }

  // only lower or upper triangle is copied.
  if (up)
  {
    /*
     * copy upper triangle, including diagonal
             DO 20 J = 1, N
                DO 10 I = 1, MIN( J, M )
                   B( I, J ) = A( I, J )
       10       CONTINUE
       20    CONTINUE
     */
    if ((i <= min(j, m - 1)) && (j < n))
    {
      Areg           = A[i + j * lda];
      B[i + j * ldb] = T_ELEM_DST(Areg);
    }
  }
  else
  {
    /*
     * copy lower triangle, including diagonal
             DO 40 J = 1, N
                DO 30 I = J, M
                   B( I, J ) = A( I, J )
       30       CONTINUE
       40    CONTINUE

     */
    if (((j <= i) && (i < m)) && (j < n))
    {
      Areg           = A[i + j * lda];
      B[i + j * ldb] = T_ELEM_DST(Areg);
    }
  }
}

/*
 * SLACPY copies all or part of a two-dimensional matrix A to another
 * matrix B.
 *
 * Input
 * -------
 *          UPLO is CHARACTER*1
 *          Specifies the part of the matrix A to be copied to B.
 *          = 'U':      Upper triangular part
 *          = 'L':      Lower triangular part
 *          Otherwise:  All of the matrix A
 *
 *          M is INTEGER
 *          The number of rows of the matrix A.
 *          M >= 0.
 *
 *          N is INTEGER
 *          The number of columns of the matrix A.
 *          N >= 0.
 *
 *          A is REAL array, dimension (LDA,N)
 *          The m by n matrix A.  If UPLO = 'U', only the upper triangle
 *          or trapezoid is accessed; if UPLO = 'L', only the lower
 *          triangle or trapezoid is accessed.
 *
 *          LDA is INTEGER
 *          The first dimension of the array A. LDA >= max(1,M).
 *
 *          B is REAL array, dimension (LDB,N)
 *          On exit, B = A in the locations specified by UPLO.
 *
 *          LDB is INTEGER
 *          The leading dimension of the array B.  LDB >= max(1,M).
 *
 */
template <typename T_ELEM_SRC, typename T_ELEM_DST>
cusolverStatus_t cusolverDnXlacpy(
  cublasFillMode_t uplo, // "UPPER", B = upper(A)
                         // "LOWER", B = lower(A)
                         // otherwise, B = A
  int m,
  int n,
  const T_ELEM_SRC* A,
  int lda,
  T_ELEM_DST* B,
  int ldb,
  cudaStream_t stream)
{
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1   = cudaSuccess;

  int up        = 0;
  int up_and_lo = 0;

  //  Quick return if possible
  if ((0 >= m) || (0 >= n))
  {
    return status;
  }

  /*
   *  up     up_and_lo
   *  1          0          upper triangle, including diagonal
   *  0          0          lower triangle, including diagonal
   *  ?          1          whole matrix
   */
  if (CUBLAS_FILL_MODE_LOWER == uplo)
  {
    // Lower triangular part
    up = 0;
  }
  else if (CUBLAS_FILL_MODE_UPPER == uplo)
  {
    // upper triangular part
    up = 1;
  }
  else
  {
    up_and_lo = 1; // Otherwise:  All of the matrix A
  }

  const int VEC_LOG = 5;
  const int BY_LOG  = 3;
  const int VEC     = (1 << VEC_LOG);
  const int BY      = (1 << BY_LOG);
  dim3 grids((m + VEC - 1) / VEC, (n + BY - 1) / BY);
  dim3 threads(VEC, BY);

  lacpy_kernel<T_ELEM_SRC, T_ELEM_DST, VEC_LOG, BY_LOG>
    <<<grids, threads, 0, stream>>>(m, n, A, (size_t) lda, B, (size_t) ldb, up, up_and_lo);

  cudaStat1 = cudaGetLastError(); /* launch error */
  if (cudaSuccess != cudaStat1)
  {
    fprintf(stderr, "Error (lacpy): %d\n", cudaStat1);
    status = CUSOLVER_STATUS_EXECUTION_FAILED;
  }

  return status;
}

cusolverStatus_t cusolverDnDlacpy(
  cublasFillMode_t uplo, // "UPPER", B = upper(A)
                         // "LOWER", B = lower(A)
                         // otherwise, B = A
  int m,
  int n,
  const double* A,
  int lda,
  double* B,
  int ldb,
  cudaStream_t stream)
{
  return cusolverDnXlacpy<double, double>(uplo, m, n, A, lda, B, ldb, stream);
}

// Pretend there is a CUBLAS interface for DLAAUM
void cublasDnDlaaum_bufferSize(cublasHandle_t /*unused*/, int m, int n, size_t* Workspace_size)
{
  assert(Workspace_size);
  *Workspace_size = m * n * sizeof(double);
}

// Pretend there is a CUBLAS interface for DLAAUM
// A triangular
// Lower : A = A^T * A
// Upper : A = A A^T
void cublasDnDlaaum(
  cublasHandle_t cublas_handle,
  cublasFillMode_t uplo,
  int m,
  int n,
  double* A,
  int ldA,
  double* Workspace_d,
  size_t Workspace_size)
{
  cudaStream_t stream;
  cuda_safe_call(cublasGetStream(cublas_handle, &stream));

  // "Hand coded"
  // We use a full copy of A !
  // fprintf(stderr, "GOT Workspace_size %ld ... expected %d\n", Workspace_size, m * n * sizeof(double));
  std::ignore = Workspace_size;
  assert(Workspace_size >= m * n * sizeof(double));

  double* B = Workspace_d;
  int ldB   = m;

  // Blank the buffer
  cuda_safe_call(cudaMemsetAsync(B, 0, m * n * sizeof(double), stream));

  // Copy A (with upper or lower 0 untouched)
  cusolverDnDlacpy(uplo, m, n, A, ldA, B, ldB, stream);

  cublasDiagType_t diag = CUBLAS_DIAG_NON_UNIT;
  const double one      = 1.0;

  auto side = (uplo == CUBLAS_FILL_MODE_LOWER) ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

  // LOWER: TRMM(A,B) : B = op(A) * B = A^T * B with A triangular (B = C in CUBLAS), CUBLAS_OP_T, CUBLAS_SIDE_RIGHT
  // UPPER: TRMM(A,B) : B = B * op(A) = B A^T with A triangular (B = C in CUBLAS), CUBLAS_OP_T, CUBLAS_SIDE_RIGHT
  cuda_safe_call(cublasDtrmm(cublas_handle, side, uplo, CUBLAS_OP_T, diag, m, n, &one, A, ldA, B, ldB, B, ldB));

  // Copy B=AA^T back into A (with upper or lower 0 untouched)
  cusolverDnDlacpy(uplo, m, n, B, ldB, A, ldA, stream);
}

void DLAAUM(cublasFillMode_t uplo, matrix<double>& A, int A_row, int A_col)
{
  int NB = A.mb;
  size_t Lwork;
  cublasDnDlaaum_bufferSize(get_cublas_handle(), NB, NB, &Lwork);

  auto d_buffer = ctx.logical_data(shape_of<slice<char>>(Lwork));

  auto t = ctx.task(A.get_handle(A_row, A_col).rw(), d_buffer.write());
  t.set_symbol("DLAAUM");
  t->*[&](auto s, auto sA, auto buffer) {
    auto& h = get_cublas_handle();
    cuda_try(cublasSetStream(h, s));

    cublasDnDlaaum(
      h, uplo, sA.extent(0), sA.extent(1), sA.data_handle(), sA.stride(1), (double*) buffer.data_handle(), Lwork);
  };
}

void DGEMM(
  cublasOperation_t transa,
  cublasOperation_t transb,
  double alpha,
  matrix<double>& A,
  int A_row,
  int A_col,
  matrix<double>& B,
  int B_row,
  int B_col,
  double beta,
  matrix<double>& C,
  int C_row,
  int C_col)
{
  auto ignored = get_cublas_handle();
  auto t =
    ctx.task(A.get_handle(A_row, A_col).read(), B.get_handle(B_row, B_col).read(), C.get_handle(C_row, C_col).rw());
  t.set_symbol("DGEMM");
  t->*[&](auto s, auto sA, auto sB, auto sC) {
    auto& h = get_cublas_handle();
    cuda_try(cublasSetStream(h, s));

    int k = (transa == CUBLAS_OP_N) ? sA.extent(1) : sA.extent(0);
    cuda_try(cublasDgemm(
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

void DSYMM(
  cublasSideMode_t side,
  cublasFillMode_t uplo,
  double alpha,
  matrix<double>& A,
  int A_row,
  int A_col,
  matrix<double>& B,
  int B_row,
  int B_col,
  double beta,
  matrix<double>& C,
  int C_row,
  int C_col)
{
  auto ignored = get_cublas_handle();
  auto t =
    ctx.task(A.get_handle(A_row, A_col).read(), B.get_handle(B_row, B_col).read(), C.get_handle(C_row, C_col).rw());
  t.set_symbol("DSYMM");
  t->*[&](auto s, auto sA, auto sB, auto sC) {
    auto& h = get_cublas_handle();
    cuda_try(cublasSetStream(h, s));

    cuda_try(cublasDsymm(
      h,
      side,
      uplo,
      sC.extent(0),
      sC.extent(1),
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
  matrix<double>& A,
  int A_row,
  int A_col,
  double beta,
  matrix<double>& C,
  int C_row,
  int C_col)
{
  auto ignored = get_cublas_handle();
  auto t       = ctx.task(A.get_handle(A_row, A_col).read(), C.get_handle(C_row, C_col).rw());
  t.set_symbol("DSYRK");
  t->*[&](auto s, auto sA, auto sC) {
    auto& h = get_cublas_handle();
    cuda_try(cublasSetStream(h, s));

    // number of rows of matrix op(A) and C
    int n = sC.extent(0);

    // number of columns of matrix op(A)
    int k = (trans == CUBLAS_OP_N) ? sA.extent(1) : sA.extent(0);

    cuda_try(
      cublasDsyrk(h, uplo, trans, n, k, &alpha, sA.data_handle(), sA.stride(1), &beta, sC.data_handle(), sC.stride(1)));
  };
}

void DTRSM(
  cublasSideMode_t side,
  cublasFillMode_t uplo,
  cublasOperation_t transa,
  cublasDiagType_t diag,
  double alpha,
  matrix<double>& A,
  int A_row,
  int A_col,
  matrix<double>& B,
  int B_row,
  int B_col)
{
  auto ignored = get_cublas_handle();
  auto t       = ctx.task(A.get_handle(A_row, A_col).read(), B.get_handle(B_row, B_col).rw());
  t.set_symbol("DTRSM");
  t->*[&](auto s, auto sA, auto sB) {
    auto& h = get_cublas_handle();
    cuda_try(cublasSetStream(h, s));

    cuda_try(cublasDtrsm(
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

void DTRMM(
  cublasSideMode_t side,
  cublasFillMode_t uplo,
  cublasOperation_t transa,
  cublasDiagType_t diag,
  double alpha,
  matrix<double>& A,
  int A_row,
  int A_col,
  matrix<double>& B,
  int B_row,
  int B_col)
{
  auto ignored = get_cublas_handle();
  auto t       = ctx.task(A.get_handle(A_row, A_col).read(), B.get_handle(B_row, B_col).rw());
  t.set_symbol("DTRMM");
  t->*[&](auto s, auto sA, auto sB) {
    auto& h = get_cublas_handle();
    cuda_try(cublasSetStream(h, s));

    // Note : CUBLAS DTRMM implementation is out of place but supports in place by using the same buffer B and C
    cuda_try(cublasDtrmm(
      get_cublas_handle(),
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
      sB.stride(1),
      sB.data_handle(),
      sB.stride(1) /* same as B*/));
  };
}

void PDNRM2_HOST(matrix<double>* A, double* result)
{
#ifdef HAVE_DOT
  ctx.get_dot()->set_current_color("red");
#endif

  for (int rowb = 0; rowb < A->mt; rowb++)
  {
    for (int colb = 0; colb < A->nt; colb++)
    {
      ctx.host_launch(A->get_handle(rowb, colb).read())->*[=](auto sA) {
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
  ctx.get_dot()->set_current_color("yellow");
#endif

  assert(A.m == A.n);
  assert(A.mt == A.nt);

  int NBLOCKS = A.mt;
  assert(A.mb == A.nb);

  nvtxRangePushA("SUBMIT_PDPOTRF");
  for (int K = 0; K < NBLOCKS; K++)
  {
    cuda_try(cudaSetDevice(A.get_preferred_devid(K, K)));
    DPOTRF(CUBLAS_FILL_MODE_LOWER, A, K, K);

    for (int row = K + 1; row < NBLOCKS; row++)
    {
      cuda_try(cudaSetDevice(A.get_preferred_devid(row, K)));
      DTRSM(CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 1.0, A, K, K, A, row, K);

      for (int col = K + 1; col < row; col++)
      {
        cuda_try(cudaSetDevice(A.get_preferred_devid(row, col)));
        DGEMM(CUBLAS_OP_N, CUBLAS_OP_T, -1.0, A, row, K, A, col, K, 1.0, A, row, col);
      }

      cuda_try(cudaSetDevice(A.get_preferred_devid(row, row)));
      DSYRK(CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, -1.0, A, row, K, 1.0, A, row, row);
    }
  }

  nvtxRangePop();
}

// Algorithm from PLASMA
void PDTRSM(cublasSideMode_t side,
            cublasFillMode_t uplo,
            cublasOperation_t trans,
            cublasDiagType_t diag,
            double alpha,
            matrix<double>& A,
            matrix<double>& B)
{
  //    std::cout << "[PDTRSM] START B MT " << B.mt << " NT " << B.nt << std::endl;

  nvtxRangePushA("SUBMIT_PDTRSM");

  if (side == CUBLAS_SIDE_LEFT)
  {
    if (uplo == CUBLAS_FILL_MODE_UPPER)
    {
      // TODO
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
            cuda_try(cudaSetDevice(A.get_preferred_devid(k, k)));
            DTRSM(side, uplo, trans, diag, lalpha, A, k, k, B, k, n);
          }
          for (int m = k + 1; m < B.mt; m++)
          {
            for (int n = 0; n < B.nt; n++)
            {
              cuda_try(cudaSetDevice(A.get_preferred_devid(m, k)));
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
            cuda_try(cudaSetDevice(A.get_preferred_devid(B.mt - k - 1, B.mt - k - 1)));
            DTRSM(side, uplo, trans, diag, lalpha, A, B.mt - k - 1, B.mt - k - 1, B, B.mt - k - 1, n);
          }
          for (int m = k + 1; m < B.mt; m++)
          {
            for (int n = 0; n < B.nt; n++)
            {
              cuda_try(cudaSetDevice(A.get_preferred_devid(B.mt - k - 1, B.mt - 1 - m)));
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
  //    std::cout << "[PDTRSM] END" << std::endl;

  nvtxRangePop();
}

void PDPOTRS(matrix<double>& A, matrix<double>& B, cublasFillMode_t uplo)
{
  nvtxRangePushA("SUBMIT_PDPOTRS");

#ifdef HAVE_DOT
  ctx.get_dot()->set_current_color("green");
#endif

  //    std::cout << "[PDPOTRS] START" << std::endl;
  // Call the parallel functions.
  PDTRSM(
    CUBLAS_SIDE_LEFT, uplo, uplo == CUBLAS_FILL_MODE_UPPER ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 1.0, A, B);

#ifdef HAVE_DOT
  ctx.get_dot()->set_current_color("darkgreen");
#endif

  PDTRSM(
    CUBLAS_SIDE_LEFT, uplo, uplo == CUBLAS_FILL_MODE_UPPER ? CUBLAS_OP_N : CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 1.0, A, B);
  //    std::cout << "[PDPOTRS] END" << std::endl;

  nvtxRangePop();
}

/***************************************************************************/ /**
                                                                               * Parallel tile matrix-matrix
                                                                               *multiplication.
                                                                               * @see plasma_omp_dgemm
                                                                               ******************************************************************************/
void PDGEMM(cublasOperation_t transa,
            cublasOperation_t transb,
            double alpha,
            matrix<double>& A,
            matrix<double>& B,
            double beta,
            matrix<double>& C)
{
#ifdef HAVE_DOT
  reserved::dot::set_current_color("blue");
#endif

  for (int m = 0; m < C.mt; m++)
  {
    for (int n = 0; n < C.nt; n++)
    {
      cuda_try(cudaSetDevice(C.get_preferred_devid(m, n)));

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
          assert(A.nt == B.mt);
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

/*
 * Algorithm taken from the PLASMA library
 */
// We assume a lower triangular matrix (uplo == CUBLAS_FILL_MODE_LOWER)
void PDTRTRI(matrix<double>& A, cublasFillMode_t uplo, cublasDiagType_t diag)
{
  assert(uplo == CUBLAS_FILL_MODE_LOWER);

  nvtxRangePushA("SUBMIT_PDTRTRI");

  for (int k = 0; k < A.nt; k++)
  {
    for (int m = k + 1; m < A.mt; m++)
    {
      cuda_try(cudaSetDevice(A.get_preferred_devid(m, k)));
      DTRSM(CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, diag, -1.0, A, k, k, A, m, k);
    }
    for (int m = k + 1; m < A.mt; m++)
    {
      for (int n = 0; n < k; n++)
      {
        cuda_try(cudaSetDevice(A.get_preferred_devid(m, n)));
        DGEMM(CUBLAS_OP_N, CUBLAS_OP_N, 1.0, A, m, k, A, k, n, 1.0, A, m, n);
      }
    }
    for (int n = 0; n < k; n++)
    {
      cuda_try(cudaSetDevice(A.get_preferred_devid(k, n)));
      DTRSM(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, diag, 1.0, A, k, k, A, k, n);
    }

    // DTRTRI(...)
    cuda_try(cudaSetDevice(A.get_preferred_devid(k, k)));
    DTRTRI(uplo, diag, A, k, k);
  }

  nvtxRangePop();
}

/*
 * Algorithm taken from the PLASMA library
 */
// We assume a lower triangular matrix (uplo == CUBLAS_FILL_MODE_LOWER)
void PDLAUUM(matrix<double>& A, cublasFillMode_t uplo)
{
  assert(uplo == CUBLAS_FILL_MODE_LOWER);

  nvtxRangePushA("SUBMIT_PDLAUUM");

  for (int k = 0; k < A.mt; k++)
  {
    for (int n = 0; n < k; n++)
    {
      cuda_try(cudaSetDevice(A.get_preferred_devid(n, n)));
      DSYRK(CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, 1.0, A, k, n, 1.0, A, n, n);

      for (int m = n + 1; m < k; m++)
      {
        cuda_try(cudaSetDevice(A.get_preferred_devid(m, n)));
        DGEMM(CUBLAS_OP_T, CUBLAS_OP_N, 1.0, A, k, m, A, k, n, 1.0, A, m, n);
      }
    }
    for (int n = 0; n < k; n++)
    {
      cuda_try(cudaSetDevice(A.get_preferred_devid(k, n)));
      DTRMM(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, 1.0, A, k, k, A, k, n);
    }

    // LAAUM (Akk RW) (compute Akk^T * Akk)
    cuda_try(cudaSetDevice(A.get_preferred_devid(k, k)));
    DLAAUM(uplo, A, k, k);
  }

  nvtxRangePop();
}

void PDSYMM(cublasSideMode_t side,
            cublasFillMode_t uplo,
            double alpha,
            matrix<double>& A,
            matrix<double>& B,
            double beta,
            matrix<double>& C)
{
  int k, m, n;
  double zbeta;
  double zone = (double) 1.0;

  for (m = 0; m < C.mt; m++)
  {
    for (n = 0; n < C.nt; n++)
    {
      cuda_try(cudaSetDevice(C.get_preferred_devid(m, n)));
      /*
       *  CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_LOWER
       */
      if (side == CUBLAS_SIDE_LEFT)
      {
        if (uplo == CUBLAS_FILL_MODE_LOWER)
        {
          for (k = 0; k < C.mt; k++)
          {
            zbeta = k == 0 ? beta : zone;
            if (k < m)
            {
              DGEMM(CUBLAS_OP_N, CUBLAS_OP_N, alpha, A, m, k, B, k, n, zbeta, C, m, n);
            }
            else
            {
              if (k == m)
              {
                DSYMM(side, uplo, alpha, A, k, k, B, k, n, zbeta, C, m, n);
              }
              else
              {
                DGEMM(CUBLAS_OP_T, CUBLAS_OP_N, alpha, A, k, m, B, k, n, zbeta, C, m, n);
              }
            }
          }
        }
        /*
         *  CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_UPPER
         */
        else
        {
          for (k = 0; k < C.mt; k++)
          {
            zbeta = k == 0 ? beta : zone;
            if (k < m)
            {
              DGEMM(CUBLAS_OP_T, CUBLAS_OP_N, alpha, A, k, m, B, k, n, zbeta, C, m, n);
            }
            else
            {
              if (k == m)
              {
                DSYMM(side, uplo, alpha, A, k, k, B, k, n, zbeta, C, m, n);
              }
              else
              {
                DGEMM(CUBLAS_OP_N, CUBLAS_OP_N, alpha, A, m, k, B, k, n, zbeta, C, m, n);
              }
            }
          }
        }
      }
      /*
       *  CUBLAS_SIDE_RIGHT / CUBLAS_FILL_MODE_LOWER
       */
      else
      {
        if (uplo == CUBLAS_FILL_MODE_LOWER)
        {
          for (k = 0; k < C.nt; k++)
          {
            zbeta = k == 0 ? beta : zone;
            if (k < n)
            {
              DGEMM(CUBLAS_OP_N, CUBLAS_OP_T, alpha, B, m, k, A, n, k, zbeta, C, m, n);
            }
            else
            {
              if (k == n)
              {
                DSYMM(side, uplo, alpha, A, k, k, B, m, k, zbeta, C, m, n);
              }
              else
              {
                DGEMM(CUBLAS_OP_N, CUBLAS_OP_N, alpha, B, m, k, A, k, n, zbeta, C, m, n);
              }
            }
          }
        }
        /*
         *  CUBLAS_SIDE_RIGHT / CUBLAS_FILL_MODE_UPPER
         */
        else
        {
          for (k = 0; k < C.nt; k++)
          {
            zbeta = k == 0 ? beta : zone;
            if (k < n)
            {
              DGEMM(CUBLAS_OP_N, CUBLAS_OP_N, alpha, B, m, k, A, k, n, zbeta, C, m, n);
            }
            else
            {
              if (k == n)
              {
                DSYMM(side, uplo, alpha, A, k, k, B, m, k, zbeta, C, m, n);
              }
              else
              {
                DGEMM(CUBLAS_OP_N, CUBLAS_OP_T, alpha, B, m, k, A, n, k, zbeta, C, m, n);
              }
            }
          }
        }
      }
    }
  }
}

void PDTRMM(cublasSideMode_t side,
            cublasFillMode_t uplo,
            cublasOperation_t trans,
            cublasDiagType_t diag,
            double alpha,
            matrix<double>& A,
            matrix<double>& B)
{
  if (side == CUBLAS_SIDE_LEFT)
  {
    if (uplo == CUBLAS_FILL_MODE_UPPER)
    {
      //===========================================
      // CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_UPPER / CUBLAS_OP_N
      //===========================================
      if (trans == CUBLAS_OP_N)
      {
        for (int m = 0; m < B.mt; m++)
        {
          for (int n = 0; n < B.nt; n++)
          {
            cuda_try(cudaSetDevice(B.get_preferred_devid(m, n)));

            DTRMM(side, uplo, trans, diag, alpha, A, m, m, B, m, n);

            for (int k = m + 1; k < A.mt; k++)
            {
              DGEMM(trans, CUBLAS_OP_N, alpha, A, m, k, B, k, n, 1.0, B, m, n);
            }
          }
        }
      }
      //================================================
      // CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_UPPER / CUBLAS_OP_T
      //================================================
      else
      {
        for (int m = B.mt - 1; m > -1; m--)
        {
          for (int n = 0; n < B.nt; n++)
          {
            cuda_try(cudaSetDevice(B.get_preferred_devid(m, n)));

            DTRMM(side, uplo, trans, diag, alpha, A, m, m, B, m, n);

            for (int k = 0; k < m; k++)
            {
              DGEMM(trans, CUBLAS_OP_N, alpha, A, k, m, B, k, n, 1.0, B, m, n);
            }
          }
        }
      }
    }
    else
    {
      //===========================================
      // CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_LOWER / CUBLAS_OP_N
      //===========================================
      if (trans == CUBLAS_OP_N)
      {
        for (int m = B.mt - 1; m > -1; m--)
        {
          for (int n = 0; n < B.nt; n++)
          {
            cuda_try(cudaSetDevice(B.get_preferred_devid(m, n)));

            DTRMM(side, uplo, trans, diag, alpha, A, m, m, B, m, n);

            for (int k = 0; k < m; k++)
            {
              DGEMM(trans, CUBLAS_OP_N, alpha, A, m, k, B, k, n, 1.0, B, m, n);
            }
          }
        }
      }
      //================================================
      // CUBLAS_SIDE_LEFT / CUBLAS_FILL_MODE_LOWER / CUBLAS_OP_T
      //================================================
      else
      {
        for (int m = 0; m < B.mt; m++)
        {
          for (int n = 0; n < B.nt; n++)
          {
            DTRMM(side, uplo, trans, diag, alpha, A, m, m, B, m, n);

            for (int k = m + 1; k < A.mt; k++)
            {
              DGEMM(trans, CUBLAS_OP_N, alpha, A, k, m, B, k, n, 1.0, B, m, n);
            }
          }
        }
      }
    }
  }
  else
  {
    if (uplo == CUBLAS_FILL_MODE_UPPER)
    {
      //============================================
      // CUBLAS_SIDE_RIGHT / CUBLAS_FILL_MODE_UPPER / CUBLAS_OP_N
      //============================================
      if (trans == CUBLAS_OP_N)
      {
        for (int n = B.nt - 1; n > -1; n--)
        {
          for (int m = 0; m < B.mt; m++)
          {
            cuda_try(cudaSetDevice(B.get_preferred_devid(m, n)));

            DTRMM(side, uplo, trans, diag, alpha, A, n, n, B, m, n);

            for (int k = 0; k < n; k++)
            {
              DGEMM(CUBLAS_OP_N, trans, alpha, B, m, k, A, k, n, 1.0, B, m, n);
            }
          }
        }
      }
      //=================================================
      // CUBLAS_SIDE_RIGHT / CUBLAS_FILL_MODE_UPPER / Plasma[_Conj]Trans
      //=================================================
      else
      {
        for (int n = 0; n < B.nt; n++)
        {
          for (int m = 0; m < B.mt; m++)
          {
            cuda_try(cudaSetDevice(B.get_preferred_devid(m, n)));

            DTRMM(side, uplo, trans, diag, alpha, A, n, n, B, m, n);

            for (int k = n + 1; k < A.mt; k++)
            {
              DGEMM(CUBLAS_OP_N, trans, alpha, B, m, k, A, n, k, 1.0, B, m, n);
            }
          }
        }
      }
    }
    else
    {
      //============================================
      // CUBLAS_SIDE_RIGHT / CUBLAS_FILL_MODE_LOWER / CUBLAS_OP_N
      //============================================
      if (trans == CUBLAS_OP_N)
      {
        for (int n = 0; n < B.nt; n++)
        {
          for (int m = 0; m < B.mt; m++)
          {
            cuda_try(cudaSetDevice(B.get_preferred_devid(m, n)));

            DTRMM(side, uplo, trans, diag, alpha, A, n, n, B, m, n);

            for (int k = n + 1; k < A.mt; k++)
            {
              DGEMM(CUBLAS_OP_N, trans, alpha, B, m, k, A, k, n, 1.0, B, m, n);
            }
          }
        }
      }
      //=================================================
      // CUBLAS_SIDE_RIGHT / CUBLAS_FILL_MODE_LOWER / Plasma[_Conj]Trans
      //=================================================
      else
      {
        for (int n = B.nt - 1; n > -1; n--)
        {
          for (int m = 0; m < B.mt; m++)
          {
            cuda_try(cudaSetDevice(B.get_preferred_devid(m, n)));

            DTRMM(side, uplo, trans, diag, alpha, A, n, n, B, m, n);

            for (int k = 0; k < n; k++)
            {
              DGEMM(CUBLAS_OP_N, trans, alpha, B, m, k, A, n, k, 1.0, B, m, n);
            }
          }
        }
      }
    }
  }
}

// Taken from Chameleon (INRIA)
// All the formula are reported in the LAPACK Lawn 41:
//     http://www.netlib.org/lapack/lawns/lawn41.ps
#define FMULS_POTRI(__n) ((double) (__n) * ((2. / 3.) + (double) (__n) * ((1. / 3.) * (double) (__n) + 1.)))
#define FADDS_POTRI(__n) ((double) (__n) * ((1. / 6.) + (double) (__n) * ((1. / 3.) * (double) (__n) - 0.5)))
double flops_dpotri(double __n)
{
  double flops = (FMULS_POTRI((__n)) + FADDS_POTRI((__n)));
  return flops;
}

void run(int N, int NB)
{
  // Use pools of preallocated blocks
  auto fixed_alloc = block_allocator<fixed_size_allocator>(ctx, NB * NB * sizeof(double));
  ctx.set_allocator(fixed_alloc);

  // Set up CUBLAS and CUSOLVER
  int ndevs;
  cuda_try(cudaGetDeviceCount(&ndevs));

  for (size_t d = 0; d < ndevs; d++)
  {
    auto ldummy = ctx.logical_data(shape_of<slice<char>>(1));
    ctx.task(exec_place::device(d), ldummy.write())->*[](cudaStream_t, auto) {
      get_cublas_handle();
      get_cusolver_handle();
    };

    ctx.task(exec_place::host, ldummy.write(data_place::managed))->*[](cudaStream_t, auto) {};
  }

  cuda_try(cudaSetDevice(0));

  cudaStream_t timing_stream;
  cuda_try(cudaStreamCreate(&timing_stream));

  matrix<double> A(N, N, NB, NB, true, "A");
  matrix<double> Aref(N, N, NB, NB, false, "Aref");

  // (Hilbert matrix + 2*N*Id) to have a diagonal dominant matrix
  auto hilbert = [=] _CCCL_HOST_DEVICE(size_t row, size_t col) {
    return 1.0 / (col + row + 1.0) + 2.0 * N * (col == row);
  };

  Aref.fill(hilbert);
  A.fill(hilbert);

  /* Right-hand side */
  matrix<double> B_potrs(N, 1, NB, 1, false, "B");
  matrix<double> Bref_potrs(N, 1, NB, 1, false, "Bref");

  auto rhs_vals = [] _CCCL_HOST_DEVICE(size_t row, size_t /*unused*/) {
    return 1.0 * (row + 1);
  };

  B_potrs.fill(rhs_vals);
  Bref_potrs.fill(rhs_vals);

  int check_result = 1;
  if (getenv("CHECK_RESULT"))
  {
    check_result = atoi(getenv("CHECK_RESULT"));
  }

  int check_result_potrs = check_result;
  if (getenv("CHECK_RESULT_POTRS"))
  {
    check_result_potrs = atoi(getenv("CHECK_RESULT_POTRS"));
  }

  //    // Compute ||Bref||
  double Bref_nrm2 = 0.0;
  double res_nrm2  = 0.0;

  if (check_result_potrs)
  {
    PDNRM2_HOST(&Bref_potrs, &Bref_nrm2);
  }

  cudaEvent_t startEvent, stopEvent;

  cuda_safe_call(cudaSetDevice(0));
  cuda_safe_call(cudaStreamSynchronize(ctx.task_fence()));
  cuda_safe_call(cudaEventCreate(&startEvent));
  cuda_safe_call(cudaEventCreate(&stopEvent));
  cuda_safe_call(cudaEventRecord(startEvent, ctx.task_fence()));

  ctx.get_dot()->set_current_color("green");
  PDPOTRF(A);
  ctx.get_dot()->set_current_color("white");

  /*
   *  POTRS
   */

  if (check_result_potrs)
  {
    // Solve AX = B and put the result in B
    PDPOTRS(A, B_potrs, CUBLAS_FILL_MODE_LOWER);

    // Compute (AX - B)
    // Bref = (Aref*B - Bref)
    PDGEMM(CUBLAS_OP_N, CUBLAS_OP_N, 1.0, Aref, B_potrs, -1.0, Bref_potrs);

    // Compute ||AX - B|| = ||Bref||
    PDNRM2_HOST(&Bref_potrs, &res_nrm2);
  }

  /*
   *  POTRI
   */
  /* PDPOTRI = PDTRTRI + PDLAUUM */

  // PDTRTRI : La^-1 (invert A)
  //    fprintf(stderr, "A=La before POTRI\n");
  //    A.print();

  ctx.get_dot()->set_current_color("yellow");
  PDTRTRI(A, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_NON_UNIT);
  ctx.get_dot()->set_current_color("white");

  //    fprintf(stderr, "A=La^-1 after POTRI\n");
  //    A.print();

  // Computes the lower part of A^tA (La^-t La^-1)
  ctx.get_dot()->set_current_color("blue");
  PDLAUUM(A, CUBLAS_FILL_MODE_LOWER);
  ctx.get_dot()->set_current_color("white");

  double b_nrm2_potri   = 0.0;
  double res_nrm2_potri = 0.0;

  if (check_result)
  {
    /* Right-hand side */
    matrix<double> B_potri(N, 1, NB, 1, false, "B_potri");
    matrix<double> Bref_potri(N, 1, NB, 1, false, "Bref_potri");

    // auto rhs_vals = [](matrix<double>& mat, int row, int col) { return 1.0 * (row + 1); };
    B_potri.fill(rhs_vals);
    Bref_potri.fill(rhs_vals);

    // AX = B, X = A^-1 B
    // LLt X = B, X = (LLt)^-1 B = L^-t L^-1 B
    // Compute Bref_potri = (A^-1 B - B)
    PDNRM2_HOST(&Bref_potri, &b_nrm2_potri);

    // B = (A^-1)*B (A triangular lower, B_potri full)
    //    fprintf(stderr, "B_potri before PDTRMM\n");
    //    B_potri.print();
    //
    //    fprintf(stderr, "A before PDTRMM\n");
    //    A.print();

    // B_tmp = 0 (to avoid NaN*0.0)
    matrix<double> B_tmp(N, 1, NB, 1, false, "B_tmp");
    auto zero_vals = [] _CCCL_HOST_DEVICE(size_t /* unused */, size_t /*unused*/) {
      return 0.0;
    };
    B_tmp.fill(zero_vals);

    // B_tmp = A * B_potri + 0*B_tmp
    PDSYMM(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 1.0, A, B_potri, 0.0, B_tmp);

    //    fprintf(stderr, "B_potri after PDTRMM\n");
    //    B_potri.print();

    // res = A X - B
    PDGEMM(CUBLAS_OP_N, CUBLAS_OP_N, 1.0, Aref, B_tmp, -1.0, Bref_potri);

    //    fprintf(stderr, "Bref_potri after PDGEMM\n");
    //    Bref_potri.print();

    // Compute residual
    PDNRM2_HOST(&Bref_potri, &res_nrm2_potri);
  }

  cuda_safe_call(cudaSetDevice(0));
  cuda_safe_call(cudaEventRecord(stopEvent, ctx.task_fence()));

  ctx.finalize();

  if (check_result_potrs)
  {
    double residual = sqrt(res_nrm2) / sqrt(Bref_nrm2);
    // std::cout << "[POTRS] ||AX - B|| : " << sqrt(res_nrm2) << std::endl;
    // std::cout << "[POTRS] ||B|| : " << sqrt(Bref_nrm2) << std::endl;
    // std::cout << "[POTRS] RESIDUAL (||AX - B||/||B||) : " << residual << std::endl;
    EXPECT(residual < 0.01);
  }

  if (check_result)
  {
    double residual_potri = sqrt(res_nrm2_potri) / sqrt(b_nrm2_potri);
    // std::cout << "[POTRI] RESIDUAL ||A * ((A^-1)B) - B|| : " << sqrt(res_nrm2_potri) << std::endl;
    // std::cout << "[POTRI] RESIDUAL ||B|| : " << sqrt(b_nrm2_potri) << std::endl;
    // std::cout << "[POTRI] RESIDUAL (||A * ((A^-1)B) - B||/||B||) : " << residual_potri << std::endl;
    EXPECT(residual_potri < 0.0001);
  }

  //    // Compute Aref * A^-1 in Aref (A^-1 is lower triangular)
  //    PDTRMM(CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, 1.0, A, Aref);

  //    // This should be almost identity
  //    Aref.print();

#if 0

    std::cout << "Print A^-1 after PDLAUUM : " << std::endl;
    A.print();

    std::cout << "RES after AX - B POTRI : " << std::endl;
    Bref_potri.print();

    // This should be almost identity
    Aref.print();
#endif

  float milliseconds;
  cuda_safe_call(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

  double gflops = flops_dpotri((double) N) / (1000000000.0);
  std::cout
    << "[PDPOTRI] ELAPSED: " << milliseconds << " ms, GFLOPS: " << gflops / (milliseconds / 1000.0) << std::endl;
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

  assert(N % NB == 0);

  run(N, NB);
}
