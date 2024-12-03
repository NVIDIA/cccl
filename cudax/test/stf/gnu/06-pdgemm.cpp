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
 * @brief An example that implements a tiled matrix product over multiple devices using CUBLAS
 *
 * This also illustrates how the same code base can be used both with a
 * stream_ctx and a graph_ctx backend.
 */

#include <cuda/experimental/__stf/graph/graph_ctx.cuh>
#include <cuda/experimental/__stf/stream/stream_ctx.cuh>

#include <nvtx3/nvToolsExt.h>

#define TILED

using namespace cuda::experimental::stf;

static std::unordered_map<int, cublasHandle_t> cublas_handles;

/* Get a CUBLAS handle valid on the current device, or initialize it lazily */
cublasHandle_t get_cublas_handle()
{
  int dev;
  cuda_safe_call(cudaGetDevice(&dev));

  auto& result = cublas_handles[dev];
  if (result == cublasHandle_t())
  { // not found, default value inserted
    // Lazy initialization, and save the handle for future use
    cuda_safe_call(cublasCreate(&result));
  }
  return result;
}

template <typename T>
class matrix
{
public:
  template <typename Ctx>
  matrix(
    Ctx& ctx, size_t NROWS, size_t NCOLS, size_t BLOCKSIZE_ROWS, size_t BLOCKSIZE_COLS, const char* _symbol = "matrix")
  {
    symbol = _symbol;

    m  = NROWS;
    mb = BLOCKSIZE_ROWS;

    n  = NCOLS;
    nb = BLOCKSIZE_COLS;

    assert(m % mb == 0);
    assert(n % nb == 0);

    size_t s = ((size_t) m) * ((size_t) n) * sizeof(T);
    // cuda_safe_call(cudaMallocHost(&h_array, m*n*sizeof(T)));
    // fprintf(stderr, "Allocating %ld x %ld x %ld = %ld bytes (%f GB) on host for %s\n", m, n, sizeof(T), s,
    //        s / (1024.0 * 1024.0 * 1024.0), _symbol);
    h_array = (T*) malloc(s);
    assert(h_array);
    cuda_safe_call(cudaHostRegister(h_array, s, cudaHostRegisterPortable));

    // Compute the number of blocks
    mt = m / mb;
    nt = n / nb;

    handles.resize(mt * nt);

    for (size_t colb = 0; colb < nt; colb++)
    {
      for (size_t rowb = 0; rowb < mt; rowb++)
      {
        T* addr_h = get_block_h(rowb, colb);

#ifdef TILED
        // tiles are stored contiguously
        const size_t ld = mb;
#else
        const size_t ld = m;
#endif

        std::ignore = ld; // avoid warning #177-D: variable "ld" was declared but never referenced
        auto s      = make_slice(addr_h, std::tuple{mb, nb}, ld);
        auto tile   = ctx.logical_data(s);

        tile.set_symbol(std::string(symbol) + "_" + std::to_string(rowb) + "_" + std::to_string(colb));

        handles[rowb + colb * mt] = std::move(tile);
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
    //           << "p=" << grid_p << " q=" << grid_q << std::endl;
  }

  int get_preferred_devid(int row, int col)
  {
    return (row % grid_p) + (col % grid_q) * grid_p;
  }

  logical_data<slice<T, 2>>& get_handle(int row, int col)
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
  void fill(T (*func)(matrix<T>*, int, int))
  {
    // Fill blocks by blocks
    for (int colb = 0; colb < nt; colb++)
    {
      for (int rowb = 0; rowb < mt; rowb++)
      {
        T* addr_h = get_block_h(rowb, colb);
#ifdef TILED
        // tiles are stored contiguously
        int ld = mb;
#else
        int ld = m;
#endif

        for (int lrow = 0; lrow < mb; lrow++)
        {
          for (int lcol = 0; lcol < nb; lcol++)
          {
            size_t row = lrow + rowb * mb;
            size_t col = lcol + colb * nb;

            T val = func(this, row, col);

            addr_h[lrow + lcol * ld] = val;
          }
        }
      }
    }
  }

  T* h_array;
  size_t m; // nrows
  size_t n; // ncols

  size_t mb; // block size (rows)
  size_t nb; // block size (cols)

  size_t mt; // numter of column blocks
  size_t nt; // numter of row blocks

  // abstract data handles
  std::vector<logical_data<slice<T, 2>>> handles;

  const char* symbol;

  // for the mapping
  int ndevs;
  int grid_p, grid_q;
};

template <typename Ctx>
void DGEMM(
  Ctx& ctx,
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
  auto dev = exec_place::device(C.get_preferred_devid(C_row, C_col));

  auto t = ctx.task(
    dev, A.get_handle(A_row, A_col).read(), B.get_handle(B_row, B_col).read(), C.get_handle(C_row, C_col).rw());
  t.set_symbol("DGEMM");

  t->*[&](cudaStream_t stream, auto tA, auto tB, auto tC) {
    cuda_safe_call(cublasSetStream(get_cublas_handle(), stream));
    int k = tA.extent(transa == CUBLAS_OP_N ? 1 : 0);
    cuda_safe_call(cublasDgemm(
      get_cublas_handle(),
      transa,
      transb,
      tC.extent(0),
      tC.extent(1),
      k,
      &alpha,
      tA.data_handle(),
      tA.stride(1),
      tB.data_handle(),
      tB.stride(1),
      &beta,
      tC.data_handle(),
      tC.stride(1)));
  };
}

template <typename Ctx>
void PDGEMM(Ctx& ctx,
            cublasOperation_t transa,
            cublasOperation_t transb,
            double alpha,
            matrix<double>& A,
            matrix<double>& B,
            double beta,
            matrix<double>& C)
{
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
        DGEMM(ctx, transa, transb, alpha, A, 0, 0, B, 0, 0, beta, C, m, n);
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
            DGEMM(ctx, transa, transb, alpha, A, m, k, B, k, n, zbeta, C, m, n);
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
            DGEMM(ctx, transa, transb, alpha, A, m, k, B, n, k, zbeta, C, m, n);
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
            DGEMM(ctx, transa, transb, alpha, A, k, m, B, k, n, zbeta, C, m, n);
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
            DGEMM(ctx, transa, transb, alpha, A, k, m, B, n, k, zbeta, C, m, n);
          }
        }
      }
    }
  }
}

double hilbert(matrix<double>* mat, int row, int col)
{
  return 1.0 / (col + row + 1.0) + 2.0 * mat->n * (col == row);
}

template <typename Ctx>
void run(size_t N, size_t NB)
{
  /* This is the CUDASTF context */
  Ctx ctx;

  matrix<double> A(ctx, N, N, NB, NB, "A");
  matrix<double> B(ctx, N, N, NB, NB, "B");
  matrix<double> C(ctx, N, N, NB, NB, "C");

  // (Hilbert matrix + 2*N*Id) to have a diagonal dominant matrix
  A.fill(hilbert);
  B.fill(hilbert);
  C.fill(hilbert);

  PDGEMM(ctx, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, A, B, -2.0, C);

  ctx.finalize();
}

int main(int argc, char** argv)
{
  size_t N  = 1024;
  size_t NB = 128;

  if (argc > 1)
  {
    N = atoi(argv[1]);
  }

  if (argc > 2)
  {
    NB = atoi(argv[2]);
  }

  assert(N % NB == 0);

  run<stream_ctx>(N, NB);
  run<graph_ctx>(N, NB);
}
