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

#include <cuda/experimental/stf.cuh>

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
  matrix(stream_ctx& ctx,
         size_t NROWS,
         size_t NCOLS,
         size_t BLOCKSIZE_ROWS,
         size_t BLOCKSIZE_COLS,
         const char* _symbol = "matrix")
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
        tile.set_write_back(false);

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
  template <typename Fun>
  void fill(stream_ctx& ctx, Fun&& fun)
  {
    nvtxRangePushA("FILL");
    // Fill blocks by blocks
    for (int colb = 0; colb < nt; colb++)
    {
      for (int rowb = 0; rowb < mt; rowb++)
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

void DGEMM(
  stream_ctx& ctx,
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

void PDGEMM(stream_ctx& ctx,
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

void run(stream_ctx& ctx, size_t N, size_t NB)
{
  auto fixed_alloc = block_allocator<fixed_size_allocator>(ctx, NB * NB * sizeof(double));
  ctx.set_allocator(fixed_alloc);

  // Set up CUBLAS and CUSOLVER
  int ndevs;
  cuda_safe_call(cudaGetDeviceCount(&ndevs));

  /* Warm up allocators */
  for (size_t d = 0; d < ndevs; d++)
  {
    auto lX = ctx.logical_data(shape_of<slice<double>>(1));
    ctx.parallel_for(exec_place::device(d), lX.shape(), lX.write())->*[] _CCCL_DEVICE(size_t, auto) {};
  }

  /* Initializes CUBLAS on all devices */
  for (size_t d = 0; d < ndevs; d++)
  {
    cuda_safe_call(cudaSetDevice(d));
    get_cublas_handle();
  }

  matrix<double> A(ctx, N, N, NB, NB, "A");
  matrix<double> B(ctx, N, N, NB, NB, "B");
  matrix<double> C(ctx, N, N, NB, NB, "C");

  // (Hilbert matrix + 2*N*Id)
  auto hilbert = [=] _CCCL_HOST_DEVICE(size_t row, size_t col) {
    return 1.0 / (col + row + 1.0) + 2.0 * N * (col == row);
  };

  A.fill(ctx, hilbert);
  B.fill(ctx, hilbert);
  C.fill(ctx, hilbert);

  cudaEvent_t startEvent, stopEvent;

  cuda_safe_call(cudaEventCreate(&startEvent));
  cuda_safe_call(cudaEventCreate(&stopEvent));

  cuda_safe_call(cudaEventRecord(startEvent, ctx.task_fence()));

  PDGEMM(ctx, CUBLAS_OP_N, CUBLAS_OP_N, 1.0, A, B, -2.0, C);

  cuda_safe_call(cudaEventRecord(stopEvent, ctx.task_fence()));

  ctx.finalize();

  float milliseconds;
  cuda_safe_call(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

  double gflops_pdgemm = 2.0 * ((double) N * (double) N * (double) N) / (1000000000.0);
  std::cout
    << "[PDDGEMM] ELAPSED: " << milliseconds << " ms, GFLOPS: " << gflops_pdgemm / (milliseconds / 1000.0) << '\n';
}

int main(int argc, char** argv)
{
  size_t N  = 4096;
  size_t NB = 512;

  if (argc > 1)
  {
    N = atoi(argv[1]);
  }

  if (argc > 2)
  {
    NB = atoi(argv[2]);
  }

  assert(N % NB == 0);

  stream_ctx ctx;
  run(ctx, N, NB);

  //    // Also run using a graph context.
  //    ctx = graph_ctx();
  //    run(ctx, N, NB);
}
