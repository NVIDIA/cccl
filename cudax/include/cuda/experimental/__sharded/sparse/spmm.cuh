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
 * @brief Localized cuSPARSE SpMM over a `sharded_csr` (row-major dense
 *        operands): the per-shard plan (`reserved::spmm_shard_plan`), the
 *        matrix-bound `spmm_plan` and the `spmm` overloads (host scalars, raw
 *        output, device-pointer alpha / beta).
 *
 * Opt-in vendor tier: requires the cuSPARSE headers; not part of the
 * `cuda/experimental/sharded.cuh` umbrella.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !__has_include(<cusparse.h>)
#  error "<cuda/experimental/__sharded/sparse/spmm.cuh> requires the cuSPARSE headers (cusparse.h) to be installed"
#endif // !__has_include(<cusparse.h>)

#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/container/csr.cuh>
#include <cuda/experimental/__sharded/container/sharded_array.cuh>
#include <cuda/experimental/__sharded/sparse/cusparse.cuh>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
/// @brief One shard's cuSPARSE SpMM pipeline (row-major B and C, ld = n_cols);
/// same lazy build-and-reuse model as `spmv_shard_plan` (handle owned by the
/// group cache, passed in per call).
template <typename _Tp>
struct spmm_shard_plan
{
  cusparseSpMatDescr_t mat{};
  cusparseDnMatDescr_t mB{}, mC{};
  void* workspace        = nullptr;
  size_t workspace_bytes = 0;
  data_place wplace;
  const _Tp* bound_B = nullptr;
  _Tp* bound_C       = nullptr;
  bool built         = false;

  void build(cusparseHandle_t handle,
             const csr_shard<_Tp>& sh,
             ::std::int64_t cols,
             ::std::int64_t n_cols,
             const _Tp* B,
             _Tp* C,
             cudaStream_t stream)
  {
    const cudaDataType dt = cusparse_data_type<_Tp>::value;
    _Tp alpha = 1, beta = 0; // plan sizing only; real values passed per call
    cusparse_safe_call(cusparseSetStream(handle, stream));
    cusparse_safe_call(cusparseCreateCsr(
      &mat,
      sh.rows,
      cols,
      sh.nnz,
      sh.offsets,
      sh.colinds,
      sh.values,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_BASE_ZERO,
      dt));
    cusparse_safe_call(cusparseCreateDnMat(&mB, cols, n_cols, n_cols, const_cast<_Tp*>(B), dt, CUSPARSE_ORDER_ROW));
    cusparse_safe_call(cusparseCreateDnMat(&mC, sh.rows, n_cols, n_cols, C, dt, CUSPARSE_ORDER_ROW));
    cusparse_safe_call(cusparseSpMM_bufferSize(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      mat,
      mB,
      &beta,
      mC,
      dt,
      CUSPARSE_SPMM_CSR_ALG3,
      &workspace_bytes));
    wplace              = sh.place;
    const size_t wbytes = workspace_bytes == 0 ? 16 : workspace_bytes;
    workspace           = wplace.allocate(static_cast<::std::ptrdiff_t>(wbytes), stream);
    cuda_safe_call(cudaStreamSynchronize(stream));
    cusparse_safe_call(cusparseSpMM_preprocess(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      mat,
      mB,
      &beta,
      mC,
      dt,
      CUSPARSE_SPMM_CSR_ALG3,
      workspace));
    // Warm-up launch + drain: see spmv_shard_plan::build (first product
    // through a fresh handle is not reliably stream-ordered). The warm-up
    // writes into a SCRATCH output, not the user's C: the visible first call
    // may carry beta != 0, whose C contents must survive for run()'s real
    // launch.
    {
      const size_t scratch_elems = static_cast<size_t>(sh.rows) * static_cast<size_t>(n_cols);
      _Tp* C_scratch =
        static_cast<_Tp*>(wplace.allocate(static_cast<::std::ptrdiff_t>(scratch_elems * sizeof(_Tp)), stream));
      cusparseDnMatDescr_t mC_scratch{};
      cusparse_safe_call(cusparseCreateDnMat(&mC_scratch, sh.rows, n_cols, n_cols, C_scratch, dt, CUSPARSE_ORDER_ROW));
      cusparse_safe_call(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat,
        mB,
        &beta,
        mC_scratch,
        dt,
        CUSPARSE_SPMM_CSR_ALG3,
        workspace));
      cuda_safe_call(cudaStreamSynchronize(stream));
      cusparse_safe_call(cusparseDestroyDnMat(mC_scratch));
      wplace.deallocate(C_scratch, scratch_elems * sizeof(_Tp), stream);
    }
    bound_B = B;
    bound_C = C;
    built   = true;
  }

  void run(cusparseHandle_t handle,
           const csr_shard<_Tp>& sh,
           ::std::int64_t cols,
           ::std::int64_t n_cols,
           const _Tp* B,
           _Tp* C,
           _Tp alpha,
           _Tp beta,
           cudaStream_t stream)
  {
    if (!built)
    {
      build(handle, sh, cols, n_cols, B, C, stream);
    }
    else
    {
      // The handle is shared per place across matrices: rebind its stream on
      // every call.
      cusparse_safe_call(cusparseSetStream(handle, stream));
      if (B != bound_B)
      {
        cusparse_safe_call(cusparseDnMatSetValues(mB, const_cast<_Tp*>(B)));
        bound_B = B;
      }
      if (C != bound_C)
      {
        cusparse_safe_call(cusparseDnMatSetValues(mC, C));
        bound_C = C;
      }
    }
    const cudaDataType dt = cusparse_data_type<_Tp>::value;
    cusparse_safe_call(cusparseSpMM(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      &alpha,
      mat,
      mB,
      &beta,
      mC,
      dt,
      CUSPARSE_SPMM_CSR_ALG3,
      workspace));
  }

  //! As `run`, with DEVICE-RESIDENT alpha/beta (see spmv_shard_plan).
  void run_device_scalars(
    cusparseHandle_t handle,
    const csr_shard<_Tp>& sh,
    ::std::int64_t cols,
    ::std::int64_t n_cols,
    const _Tp* B,
    _Tp* C,
    const _Tp* d_alpha,
    const _Tp* d_beta,
    cudaStream_t stream)
  {
    if (!built)
    {
      build(handle, sh, cols, n_cols, B, C, stream);
    }
    else
    {
      cusparse_safe_call(cusparseSetStream(handle, stream));
      if (B != bound_B)
      {
        cusparse_safe_call(cusparseDnMatSetValues(mB, const_cast<_Tp*>(B)));
        bound_B = B;
      }
      if (C != bound_C)
      {
        cusparse_safe_call(cusparseDnMatSetValues(mC, C));
        bound_C = C;
      }
    }
    const cudaDataType dt = cusparse_data_type<_Tp>::value;
    cusparse_safe_call(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    const cusparseStatus_t st = cusparseSpMM(
      handle,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      CUSPARSE_OPERATION_NON_TRANSPOSE,
      d_alpha,
      mat,
      mB,
      d_beta,
      mC,
      dt,
      CUSPARSE_SPMM_CSR_ALG3,
      workspace);
    cusparse_safe_call(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
    cusparse_safe_call(st);
  }

  ~spmm_shard_plan()
  {
    if (mB)
    {
      cusparseDestroyDnMat(mB);
    }
    if (mC)
    {
      cusparseDestroyDnMat(mC);
    }
    if (mat)
    {
      cusparseDestroySpMat(mat);
    }
    if (workspace)
    {
      const size_t wbytes = workspace_bytes == 0 ? 16 : workspace_bytes;
      _CCCL_TRY
      {
        wplace.deallocate(workspace, wbytes, nullptr);
      }
      _CCCL_CATCH_ALL {}
    }
  }
};
} // namespace reserved

/**
 * @brief MATRIX-BOUND SpMM state (row-major B and C, ld = n_cols locked at
 * construction): per-shard CSR_ALG3 pipelines, same reuse model as
 * `spmv_plan`. One plan per (matrix, n_cols).
 */
template <typename _Tp>
class spmm_plan
{
public:
  spmm_plan(cusparse_handles& handles, sharded_csr<_Tp>& A, ::std::int64_t n_cols)
      : handles_(&handles)
      , A_(&A)
      , n_cols_(n_cols)
      , plans_(A.num_shards())
  {
    if (A.num_shards() != handles.group().size())
    {
      _CCCL_THROW(::std::invalid_argument, "spmm_plan: matrix was not partitioned over this group's places");
    }
  }

  spmm_plan(spmm_plan&&)                 = default;
  spmm_plan(const spmm_plan&)            = delete;
  spmm_plan& operator=(const spmm_plan&) = delete;

  sharded_csr<_Tp>& matrix() const
  {
    return *A_;
  }
  cusparse_handles& handles() const
  {
    return *handles_;
  }
  ::std::int64_t n_cols() const
  {
    return n_cols_;
  }
  reserved::spmm_shard_plan<_Tp>& shard_plan(size_t i)
  {
    return plans_[i];
  }

private:
  cusparse_handles* handles_;
  sharded_csr<_Tp>* A_;
  ::std::int64_t n_cols_;
  ::std::vector<reserved::spmm_shard_plan<_Tp>> plans_;
};

/**
 * @brief Localized cuSPARSE SpMM: C = alpha * A * B + beta * C (row-major B
 * and C, ld = plan.n_cols()), one confined deterministic (CSR_ALG3) call per
 * shard. Same contracts as `spmv`.
 */
template <typename _Tp>
void spmm(spmm_plan<_Tp>& plan, const _Tp* B, sharded_array<_Tp>& C, _Tp alpha = _Tp{1}, _Tp beta = _Tp{0})
{
  auto& A           = plan.matrix();
  const auto shards = reserved::matched_output_shards(A, C, plan.n_cols(), "sharded::spmm");
  for (const auto& [i, C_ptr] : shards)
  {
    auto& sh = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    plan.shard_plan(i).run(plan.handles().get(i), sh, A.num_cols(), plan.n_cols(), B, C_ptr, alpha, beta, sh.stream);
  }
}

/**
 * @brief Localized SpMM into a CONTIGUOUS row-major output: C is one device
 * pointer (ld = plan.n_cols()); each shard writes rows [row_begin,
 * row_begin + rows) at C + row_begin * n_cols. See the spmv overload.
 */
template <typename _Tp>
void spmm(spmm_plan<_Tp>& plan, const _Tp* B, _Tp* C, _Tp alpha = _Tp{1}, _Tp beta = _Tp{0})
{
  auto& A = plan.matrix();
  for (size_t i = 0; i < A.num_shards(); i++)
  {
    auto& sh = A.shard(i);
    if (sh.rows == 0)
    {
      continue;
    }
    if (sh.nnz == 0)
    {
      _CCCL_THROW(
        ::std::invalid_argument,
        "sharded::spmm: shard " + ::std::to_string(i) + " has rows but no nonzeros; adjust the row boundaries");
    }
    places::exec_place_scope scope(sh.exec);
    plan.shard_plan(i).run(
      plan.handles().get(i),
      sh,
      A.num_cols(),
      plan.n_cols(),
      B,
      C + sh.row_begin * plan.n_cols(),
      alpha,
      beta,
      sh.stream);
  }
}

/**
 * @brief As the contiguous-output `spmm`, with DEVICE-RESIDENT alpha/beta.
 */
template <typename _Tp>
void spmm(spmm_plan<_Tp>& plan, const _Tp* B, _Tp* C, const _Tp* d_alpha, const _Tp* d_beta)
{
  auto& A = plan.matrix();
  for (size_t i = 0; i < A.num_shards(); i++)
  {
    auto& sh = A.shard(i);
    if (sh.rows == 0)
    {
      continue;
    }
    if (sh.nnz == 0)
    {
      _CCCL_THROW(
        ::std::invalid_argument,
        "sharded::spmm: shard " + ::std::to_string(i) + " has rows but no nonzeros; adjust the row boundaries");
    }
    places::exec_place_scope scope(sh.exec);
    plan.shard_plan(i).run_device_scalars(
      plan.handles().get(i),
      sh,
      A.num_cols(),
      plan.n_cols(),
      B,
      C + sh.row_begin * plan.n_cols(),
      d_alpha,
      d_beta,
      sh.stream);
  }
}
} // namespace cuda::experimental::sharded
