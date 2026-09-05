//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief `sharded::spmv` / `sharded::spmm` correctness (FP64): against a host
 *        reference on synthetic matrices with mixed row-length distributions
 *        (empty rows included) and a skewed one; bitwise against ONE
 *        whole-matrix cuSPARSE call; alpha/beta accumulation; plan reuse
 *        across repeated calls; outputs into a contiguous (VMM-backed)
 *        sharded array read back through the single base pointer; shape
 *        validation.
 */

#include <cuda/experimental/__sharded/sparse.cuh> // opt-in vendor tier
#include <cuda/experimental/sharded.cuh>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct host_csr
{
  ::std::int64_t rows = 0, cols = 0;
  ::std::vector<int> offsets;
  ::std::vector<int> colinds;
  ::std::vector<double> values;

  ::std::int64_t nnz() const
  {
    return offsets.back();
  }
};

// Mixed row lengths: empty rows, short rows, a long-row band.
host_csr make_mixed_csr(::std::int64_t rows, ::std::int64_t cols, unsigned seed)
{
  host_csr m;
  m.rows = rows;
  m.cols = cols;
  m.offsets.push_back(0);
  ::std::mt19937 rng(seed);
  ::std::uniform_int_distribution<int> col_dist(0, static_cast<int>(cols) - 1);
  ::std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
  for (::std::int64_t r = 0; r < rows; r++)
  {
    int len = 0;
    switch (r % 11)
    {
      case 0:
        len = 0; // empty row
        break;
      case 1:
      case 2:
      case 3:
        len = 1 + static_cast<int>(r % 4);
        break;
      case 4:
        len = 48; // long row
        break;
      default:
        len = 6 + static_cast<int>(r % 9);
        break;
    }
    for (int k = 0; k < len; k++)
    {
      m.colinds.push_back(col_dist(rng));
      m.values.push_back(val_dist(rng));
    }
    m.offsets.push_back(m.offsets.back() + len);
  }
  return m;
}

// Skewed structure: the first fifth of the rows carries most of the
// nonzeros in short rows; the rest are long rows.
host_csr make_skewed_csr(::std::int64_t rows, ::std::int64_t cols, unsigned seed)
{
  host_csr m;
  m.rows = rows;
  m.cols = cols;
  m.offsets.push_back(0);
  ::std::mt19937 rng(seed);
  ::std::uniform_int_distribution<int> col_dist(0, static_cast<int>(cols) - 1);
  ::std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
  for (::std::int64_t r = 0; r < rows; r++)
  {
    const int len = (r < rows / 5) ? 2 : 40;
    for (int k = 0; k < len; k++)
    {
      m.colinds.push_back(col_dist(rng));
      m.values.push_back(val_dist(rng));
    }
    m.offsets.push_back(m.offsets.back() + len);
  }
  return m;
}

::std::vector<double> random_vector(size_t n, unsigned seed)
{
  ::std::vector<double> v(n);
  ::std::mt19937 rng(seed);
  ::std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& x : v)
  {
    x = dist(rng);
  }
  return v;
}

// y = alpha * A * x + beta * y0
::std::vector<double>
host_spmv(const host_csr& m, const ::std::vector<double>& x, double alpha, double beta, const ::std::vector<double>& y0)
{
  ::std::vector<double> y(static_cast<size_t>(m.rows));
  for (::std::int64_t r = 0; r < m.rows; r++)
  {
    double acc = 0.0;
    for (int k = m.offsets[r]; k < m.offsets[r + 1]; k++)
    {
      acc += m.values[static_cast<size_t>(k)] * x[static_cast<size_t>(m.colinds[static_cast<size_t>(k)])];
    }
    y[static_cast<size_t>(r)] = alpha * acc + beta * y0[static_cast<size_t>(r)];
  }
  return y;
}

// C = alpha * A * B + beta * C0 (B, C row-major, ld = n_cols)
::std::vector<double> host_spmm(
  const host_csr& m,
  const ::std::vector<double>& B,
  ::std::int64_t n_cols,
  double alpha,
  double beta,
  const ::std::vector<double>& C0)
{
  ::std::vector<double> C(static_cast<size_t>(m.rows * n_cols));
  for (::std::int64_t r = 0; r < m.rows; r++)
  {
    for (::std::int64_t j = 0; j < n_cols; j++)
    {
      double acc = 0.0;
      for (int k = m.offsets[r]; k < m.offsets[r + 1]; k++)
      {
        acc += m.values[static_cast<size_t>(k)]
             * B[static_cast<size_t>(m.colinds[static_cast<size_t>(k)]) * static_cast<size_t>(n_cols)
                 + static_cast<size_t>(j)];
      }
      C[static_cast<size_t>(r * n_cols + j)] = alpha * acc + beta * C0[static_cast<size_t>(r * n_cols + j)];
    }
  }
  return C;
}

void expect_close(const ::std::vector<double>& got, const ::std::vector<double>& ref, double tol = 1e-13)
{
  EXPECT(got.size() == ref.size());
  for (size_t i = 0; i < got.size(); i++)
  {
    const double err = ::std::abs(got[i] - ref[i]) / ::std::max(1.0, ::std::abs(ref[i]));
    EXPECT(err <= tol);
  }
}

// Simple whole-device buffer helpers for the single-call reference arm.
template <typename T>
T* device_upload(const ::std::vector<T>& host)
{
  T* ptr = nullptr;
  cuda_safe_call(cudaMalloc(&ptr, host.size() * sizeof(T)));
  cuda_safe_call(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyDefault));
  return ptr;
}

// ONE whole-matrix cuSPARSE SpMV on the whole device, same algorithm as the
// sharded path (CSR_ALG2), for the bitwise comparison arm.
::std::vector<double> whole_matrix_spmv(
  const host_csr& m, const ::std::vector<double>& x, double alpha, double beta, const ::std::vector<double>& y0)
{
  int* d_off    = device_upload(m.offsets);
  int* d_col    = device_upload(m.colinds);
  double* d_val = device_upload(m.values);
  double* d_x   = device_upload(x);
  double* d_y   = device_upload(y0);

  cusparseHandle_t handle{};
  cusparseSpMatDescr_t mat{};
  cusparseDnVecDescr_t vx{}, vy{};
  cusparse_safe_call(cusparseCreate(&handle));
  cusparse_safe_call(cusparseCreateCsr(
    &mat,
    m.rows,
    m.cols,
    m.nnz(),
    d_off,
    d_col,
    d_val,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F));
  cusparse_safe_call(cusparseCreateDnVec(&vx, m.cols, d_x, CUDA_R_64F));
  cusparse_safe_call(cusparseCreateDnVec(&vy, m.rows, d_y, CUDA_R_64F));
  size_t wbytes = 0;
  cusparse_safe_call(cusparseSpMV_bufferSize(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, &wbytes));
  void* work = nullptr;
  cuda_safe_call(cudaMalloc(&work, wbytes == 0 ? 16 : wbytes));
  cusparse_safe_call(cusparseSpMV_preprocess(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, work));
  cusparse_safe_call(cusparseSpMV(
    handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy, CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2, work));
  cuda_safe_call(cudaDeviceSynchronize());

  ::std::vector<double> y(static_cast<size_t>(m.rows));
  cuda_safe_call(cudaMemcpy(y.data(), d_y, y.size() * sizeof(double), cudaMemcpyDefault));

  cusparse_safe_call(cusparseDestroyDnVec(vx));
  cusparse_safe_call(cusparseDestroyDnVec(vy));
  cusparse_safe_call(cusparseDestroySpMat(mat));
  cusparse_safe_call(cusparseDestroy(handle));
  cuda_safe_call(cudaFree(work));
  cuda_safe_call(cudaFree(d_off));
  cuda_safe_call(cudaFree(d_col));
  cuda_safe_call(cudaFree(d_val));
  cuda_safe_call(cudaFree(d_x));
  cuda_safe_call(cudaFree(d_y));
  return y;
}

// ONE whole-matrix cuSPARSE SpMM (CSR_ALG3, row-major), the sharded path's
// exact configuration, for the bitwise comparison arm.
::std::vector<double> whole_matrix_spmm(
  const host_csr& m,
  const ::std::vector<double>& B,
  ::std::int64_t n_cols,
  double alpha,
  double beta,
  const ::std::vector<double>& C0)
{
  int* d_off    = device_upload(m.offsets);
  int* d_col    = device_upload(m.colinds);
  double* d_val = device_upload(m.values);
  double* d_B   = device_upload(B);
  double* d_C   = device_upload(C0);

  cusparseHandle_t handle{};
  cusparseSpMatDescr_t mat{};
  cusparseDnMatDescr_t mB{}, mC{};
  cusparse_safe_call(cusparseCreate(&handle));
  cusparse_safe_call(cusparseCreateCsr(
    &mat,
    m.rows,
    m.cols,
    m.nnz(),
    d_off,
    d_col,
    d_val,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_32I,
    CUSPARSE_INDEX_BASE_ZERO,
    CUDA_R_64F));
  cusparse_safe_call(cusparseCreateDnMat(&mB, m.cols, n_cols, n_cols, d_B, CUDA_R_64F, CUSPARSE_ORDER_ROW));
  cusparse_safe_call(cusparseCreateDnMat(&mC, m.rows, n_cols, n_cols, d_C, CUDA_R_64F, CUSPARSE_ORDER_ROW));
  size_t wbytes = 0;
  cusparse_safe_call(cusparseSpMM_bufferSize(
    handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    mat,
    mB,
    &beta,
    mC,
    CUDA_R_64F,
    CUSPARSE_SPMM_CSR_ALG3,
    &wbytes));
  void* work = nullptr;
  cuda_safe_call(cudaMalloc(&work, wbytes == 0 ? 16 : wbytes));
  cusparse_safe_call(cusparseSpMM_preprocess(
    handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    mat,
    mB,
    &beta,
    mC,
    CUDA_R_64F,
    CUSPARSE_SPMM_CSR_ALG3,
    work));
  cusparse_safe_call(cusparseSpMM(
    handle,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    &alpha,
    mat,
    mB,
    &beta,
    mC,
    CUDA_R_64F,
    CUSPARSE_SPMM_CSR_ALG3,
    work));
  cuda_safe_call(cudaDeviceSynchronize());

  ::std::vector<double> C(static_cast<size_t>(m.rows * n_cols));
  cuda_safe_call(cudaMemcpy(C.data(), d_C, C.size() * sizeof(double), cudaMemcpyDefault));

  cusparse_safe_call(cusparseDestroyDnMat(mB));
  cusparse_safe_call(cusparseDestroyDnMat(mC));
  cusparse_safe_call(cusparseDestroySpMat(mat));
  cusparse_safe_call(cusparseDestroy(handle));
  cuda_safe_call(cudaFree(work));
  cuda_safe_call(cudaFree(d_off));
  cuda_safe_call(cudaFree(d_col));
  cuda_safe_call(cudaFree(d_val));
  cuda_safe_call(cudaFree(d_B));
  cuda_safe_call(cudaFree(d_C));
  return C;
}

void test_spmv(place_group& group, const host_csr& m)
{
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  cusparse_handles handles(group);
  spmv_plan<double> plan(handles, A);

  const auto x_h  = random_vector(static_cast<size_t>(m.cols), 101);
  const auto y0_h = random_vector(static_cast<size_t>(m.rows), 102);
  double* d_x     = device_upload(x_h);

  auto y = A.make_row_partitioned();

  // alpha/beta accumulation: y = 1.5*A*x + 2.5*y0
  const double alpha = 1.5, beta = 2.5;
  y.copy_from_host(y0_h.data());
  spmv(plan, d_x, y, alpha, beta);
  y.sync();
  ::std::vector<double> got(static_cast<size_t>(m.rows));
  y.copy_to_host(got.data());
  expect_close(got, host_spmv(m, x_h, alpha, beta, y0_h));

  // Plan-reuse determinism: repeated beta=0 calls are byte-identical.
  ::std::vector<double> first(got.size()), again(got.size());
  spmv(plan, d_x, y);
  y.sync();
  y.copy_to_host(first.data());
  for (int rep = 0; rep < 2; rep++)
  {
    spmv(plan, d_x, y);
    y.sync();
    y.copy_to_host(again.data());
    EXPECT(::std::memcmp(again.data(), first.data(), first.size() * sizeof(double)) == 0);
  }
  expect_close(first, host_spmv(m, x_h, 1.0, 0.0, y0_h));

  // Against ONE whole-matrix call, same algorithm. NOT asserted bitwise:
  // CSR_ALG2 is deterministic per matrix shape, but its reduction shape
  // depends on the call's row count, so a row split can differ from the
  // whole-matrix call in the last bits (observed here, as with the SpMM/SpMV
  // asymmetry below: SpMM CSR_ALG3 IS bitwise across the split and is
  // asserted so).
  const auto whole = whole_matrix_spmv(m, x_h, 1.0, 0.0, y0_h);
  expect_close(first, whole);

  // Contiguous (VMM-backed) output: same bytes, readable through ONE pointer.
  {
    auto yc = A.make_row_partitioned(1, /* contiguous */ true);
    spmv(plan, d_x, yc);
    yc.sync();
    ::std::vector<double> contig(static_cast<size_t>(m.rows));
    cuda_safe_call(cudaMemcpy(contig.data(), yc.contiguous_data(), contig.size() * sizeof(double), cudaMemcpyDefault));
    EXPECT(::std::memcmp(contig.data(), first.data(), first.size() * sizeof(double)) == 0);
  }

  // A mis-shaped output is refused before any engine work.
  {
    auto bad   = sharded_array<double>::allocate(group, 2 * static_cast<size_t>(m.rows));
    bool threw = false;
    try
    {
      spmv(plan, d_x, bad);
    }
    catch (const ::std::invalid_argument&)
    {
      threw = true;
    }
    EXPECT(threw);
  }

  cuda_safe_call(cudaFree(d_x));
}

void test_spmm(place_group& group, const host_csr& m, ::std::int64_t n_cols)
{
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  cusparse_handles handles(group);
  spmm_plan<double> plan(handles, A, n_cols);

  const auto B_h  = random_vector(static_cast<size_t>(m.cols * n_cols), 201);
  const auto C0_h = random_vector(static_cast<size_t>(m.rows * n_cols), 202);
  double* d_B     = device_upload(B_h);

  auto C = A.make_row_partitioned(n_cols);

  // alpha/beta accumulation
  const double alpha = 2.0, beta = 0.5;
  C.copy_from_host(C0_h.data());
  spmm(plan, d_B, C, alpha, beta);
  C.sync();
  ::std::vector<double> got(static_cast<size_t>(m.rows * n_cols));
  C.copy_to_host(got.data());
  expect_close(got, host_spmm(m, B_h, n_cols, alpha, beta, C0_h));

  // Plan-reuse determinism + whole-matrix bitwise arm (beta=0).
  ::std::vector<double> first(got.size()), again(got.size());
  spmm(plan, d_B, C);
  C.sync();
  C.copy_to_host(first.data());
  spmm(plan, d_B, C);
  C.sync();
  C.copy_to_host(again.data());
  EXPECT(::std::memcmp(again.data(), first.data(), first.size() * sizeof(double)) == 0);
  expect_close(first, host_spmm(m, B_h, n_cols, 1.0, 0.0, C0_h));

  const auto whole = whole_matrix_spmm(m, B_h, n_cols, 1.0, 0.0, C0_h);
  expect_close(first, whole);
  EXPECT(::std::memcmp(first.data(), whole.data(), whole.size() * sizeof(double)) == 0);

  // Contiguous (VMM-backed) output read through the single base pointer.
  {
    auto Cc = A.make_row_partitioned(n_cols, /* contiguous */ true);
    spmm(plan, d_B, Cc);
    Cc.sync();
    ::std::vector<double> contig(first.size());
    cuda_safe_call(cudaMemcpy(contig.data(), Cc.contiguous_data(), contig.size() * sizeof(double), cudaMemcpyDefault));
    EXPECT(::std::memcmp(contig.data(), first.data(), first.size() * sizeof(double)) == 0);
  }

  cuda_safe_call(cudaFree(d_B));
}

// ---------------------------------------------------------------------------
// In-place value mutation: a transform kernel rescales the operator's values
// between calls (the pattern solvers use for problem scaling). The structure
// is untouched, so the per-shard plans must remain valid with no rebuild or
// re-preprocess, and subsequent calls must track the new values exactly.
// ---------------------------------------------------------------------------
__global__ void scale_values_kernel(double* v, ::std::int64_t n, double m)
{
  ::std::int64_t i            = blockIdx.x * static_cast<::std::int64_t>(blockDim.x) + threadIdx.x;
  const ::std::int64_t stride = gridDim.x * static_cast<::std::int64_t>(blockDim.x);
  for (; i < n; i += stride)
  {
    v[i] *= m;
  }
}

void scale_shard_values(sharded_csr<double>& A, double m)
{
  for (size_t i = 0; i < A.num_shards(); i++)
  {
    auto& sh = A.shard(i);
    if (sh.nnz > 0)
    {
      scale_values_kernel<<<256, 256, 0, sh.stream>>>(sh.values, sh.nnz, m);
      cuda_safe_call(cudaGetLastError());
      cuda_safe_call(cudaStreamSynchronize(sh.stream));
    }
  }
}

void test_value_mutation(place_group& group, const host_csr& m, ::std::int64_t n_cols)
{
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  cusparse_handles handles(group);
  spmv_plan<double> plan(handles, A);
  spmm_plan<double> mplan(handles, A, n_cols);

  const auto x_h  = random_vector(static_cast<size_t>(m.cols), 601);
  const auto B_h  = random_vector(static_cast<size_t>(m.cols * n_cols), 602);
  const auto y0_h = ::std::vector<double>(static_cast<size_t>(m.rows), 0.0);
  const auto C0_h = ::std::vector<double>(static_cast<size_t>(m.rows * n_cols), 0.0);
  double* d_x     = device_upload(x_h);
  double* d_B     = device_upload(B_h);

  auto y = A.make_row_partitioned();
  auto C = A.make_row_partitioned(n_cols);

  // First calls build the plans.
  ::std::vector<double> first_y(static_cast<size_t>(m.rows));
  ::std::vector<double> first_C(static_cast<size_t>(m.rows * n_cols));
  spmv(plan, d_x, y);
  y.sync();
  y.copy_to_host(first_y.data());
  spmm(mplan, d_B, C);
  C.sync();
  C.copy_to_host(first_C.data());

  // Mutate the values in place (exact x2.0), matrix and reference alike.
  scale_shard_values(A, 2.0);
  host_csr m2 = m;
  for (auto& v : m2.values)
  {
    v *= 2.0;
  }

  // Same plans, new values: correct against the scaled host reference and
  // against ONE whole-matrix call on the scaled operator.
  ::std::vector<double> got_y(first_y.size());
  ::std::vector<double> got_C(first_C.size());
  spmv(plan, d_x, y);
  y.sync();
  y.copy_to_host(got_y.data());
  expect_close(got_y, host_spmv(m2, x_h, 1.0, 0.0, y0_h));
  expect_close(got_y, whole_matrix_spmv(m2, x_h, 1.0, 0.0, y0_h));
  spmm(mplan, d_B, C);
  C.sync();
  C.copy_to_host(got_C.data());
  expect_close(got_C, host_spmm(m2, B_h, n_cols, 1.0, 0.0, C0_h));
  expect_close(got_C, whole_matrix_spmm(m2, B_h, n_cols, 1.0, 0.0, C0_h));

  // Exact restore (x0.5): results must come back byte-identical to the
  // pre-mutation calls -- no plan or engine state drifted across mutations.
  scale_shard_values(A, 0.5);
  spmv(plan, d_x, y);
  y.sync();
  y.copy_to_host(got_y.data());
  EXPECT(::std::memcmp(got_y.data(), first_y.data(), first_y.size() * sizeof(double)) == 0);
  spmm(mplan, d_B, C);
  C.sync();
  C.copy_to_host(got_C.data());
  EXPECT(::std::memcmp(got_C.data(), first_C.data(), first_C.size() * sizeof(double)) == 0);

  cuda_safe_call(cudaFree(d_x));
  cuda_safe_call(cudaFree(d_B));
}

// Same contract with contiguous internals and a matrix adopted from device
// arrays: ONE kernel mutates the whole values array through
// contiguous_values() -- the unmodified-writer pattern (e.g. an existing
// problem-scaling kernel) -- and the per-shard plans stay valid.
void test_value_mutation_contiguous(place_group& group, const host_csr& m, ::std::int64_t n_cols)
{
  int* d_off    = device_upload(m.offsets);
  int* d_col    = device_upload(m.colinds);
  double* d_val = device_upload(m.values);
  auto A        = sharded_csr<double>::from_device(
    group, m.rows, m.cols, d_off, d_col, d_val, /*row_boundaries=*/{}, /*contiguous=*/true);
  EXPECT(A.values_contiguous());
  cusparse_handles handles(group);
  spmm_plan<double> plan(handles, A, n_cols);

  const auto B_h  = random_vector(static_cast<size_t>(m.cols * n_cols), 701);
  const auto C0_h = ::std::vector<double>(static_cast<size_t>(m.rows * n_cols), 0.0);
  double* d_B     = device_upload(B_h);
  auto C          = A.make_row_partitioned(n_cols);

  ::std::vector<double> first_C(static_cast<size_t>(m.rows * n_cols));
  spmm(plan, d_B, C);
  C.sync();
  C.copy_to_host(first_C.data());

  // One whole-array kernel through the base pointer (writer knows nothing
  // about shards or placement).
  scale_values_kernel<<<256, 256, 0, nullptr>>>(A.contiguous_values(), m.nnz(), 2.0);
  cuda_safe_call(cudaGetLastError());
  cuda_safe_call(cudaStreamSynchronize(nullptr));
  host_csr m2 = m;
  for (auto& v : m2.values)
  {
    v *= 2.0;
  }

  ::std::vector<double> got_C(first_C.size());
  spmm(plan, d_B, C);
  C.sync();
  C.copy_to_host(got_C.data());
  expect_close(got_C, host_spmm(m2, B_h, n_cols, 1.0, 0.0, C0_h));
  expect_close(got_C, whole_matrix_spmm(m2, B_h, n_cols, 1.0, 0.0, C0_h));

  // Exact restore => byte-identical to the pre-mutation call.
  scale_values_kernel<<<256, 256, 0, nullptr>>>(A.contiguous_values(), m.nnz(), 0.5);
  cuda_safe_call(cudaGetLastError());
  cuda_safe_call(cudaStreamSynchronize(nullptr));
  spmm(plan, d_B, C);
  C.sync();
  C.copy_to_host(got_C.data());
  EXPECT(::std::memcmp(got_C.data(), first_C.data(), first_C.size() * sizeof(double)) == 0);

  cuda_safe_call(cudaFree(d_off));
  cuda_safe_call(cudaFree(d_col));
  cuda_safe_call(cudaFree(d_val));
  cuda_safe_call(cudaFree(d_B));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  // Mixed row-length distribution, empty rows included; sized to span the
  // places while staying CI-friendly.
  const auto mixed = make_mixed_csr(60013, 4096, 1);
  test_spmv(group, mixed);
  test_spmm(group, mixed, 32);

  // Skewed distribution: short-row region + long-row region.
  const auto skewed = make_skewed_csr(30011, 2048, 2);
  test_spmv(group, skewed);
  test_spmm(group, skewed, 16);

  // In-place value mutation between calls (transform-rescaled operator).
  test_value_mutation(group, mixed, 32);
  test_value_mutation(group, skewed, 16);

  // Same through contiguous internals + device-adopted arrays (one
  // unmodified whole-array writer).
  test_value_mutation_contiguous(group, mixed, 32);
  test_value_mutation_contiguous(group, skewed, 16);

  return 0;
}
