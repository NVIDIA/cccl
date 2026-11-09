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
 * @brief Localized cuSPARSE SpMV over a `sharded_csr`: the per-shard plan
 *        (`reserved::spmv_shard_plan`), the matrix-bound `spmv_plan` and the
 *        `spmv` overloads (host scalars, raw output, device-pointer alpha /
 *        beta).
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
#  error "<cuda/experimental/__sharded/sparse/spmv.cuh> requires the cuSPARSE headers (cusparse.h) to be installed"
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
/**
 * @brief One shard's cuSPARSE SpMV pipeline: descriptors + workspace +
 * preprocessed plan, built lazily on first use against the shard's fixed
 * arrays and the current dense pointers, then reused (pointer rebinds only).
 * The `cusparseHandle_t` is NOT owned here: it is the place's handle from the
 * group cache, passed in per call (with the stream rebound per call).
 *
 * Build and run are expected to happen with the shard's exec place active
 * (`exec_place_scope`), so the plan's internal state is created in the
 * confined context that runs the call.
 */
template <typename _Tp>
struct spmv_shard_plan
{
  cusparseSpMatDescr_t mat{};
  cusparseDnVecDescr_t vx{}, vy{};
  void* workspace        = nullptr;
  size_t workspace_bytes = 0;
  data_place wplace; //!< place the workspace was drawn from
  const _Tp* bound_x = nullptr;
  _Tp* bound_y       = nullptr;
  bool built         = false;

  void build(
    cusparseHandle_t handle, const csr_shard<_Tp>& sh, ::std::int64_t cols, const _Tp* x, _Tp* y, cudaStream_t stream)
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
    cusparse_safe_call(cusparseCreateDnVec(&vx, cols, const_cast<_Tp*>(x), dt));
    cusparse_safe_call(cusparseCreateDnVec(&vy, sh.rows, y, dt));
    cusparse_safe_call(cusparseSpMV_bufferSize(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy, dt, CUSPARSE_SPMV_CSR_ALG2, &workspace_bytes));
    // Workspace from the shard's place, so engine scratch lands where the
    // shard's work runs (a minimal allocation keeps teardown uniform).
    wplace              = sh.place;
    const size_t wbytes = workspace_bytes == 0 ? 16 : workspace_bytes;
    workspace           = wplace.allocate(static_cast<::std::ptrdiff_t>(wbytes), stream);
    cuda_safe_call(cudaStreamSynchronize(stream));
    cusparse_safe_call(cusparseSpMV_preprocess(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy, dt, CUSPARSE_SPMV_CSR_ALG2, workspace));
    // Warm-up launch + drain: the very first product through a fresh handle
    // (lazy module/context init inside an exec-place scope) is not reliably
    // stream-ordered -- observed as an intermittent wrong FIRST result
    // in-solver (all later calls bitwise-correct). One throwaway launch here
    // makes the first visible result go through a fully warmed handle. It
    // writes into a SCRATCH output, not the user's y: the visible first call
    // may carry beta != 0, and run()'s real launch must still read the
    // caller's y contents. With the handle now shared per place, later
    // matrices' builds go through an already-warm handle and this launch is
    // merely a cheap plan shakedown. Same idiom as the warm-up run in
    // consumers' own SpMM benchmark contexts.
    {
      _Tp* y_scratch = static_cast<_Tp*>(
        wplace.allocate(static_cast<::std::ptrdiff_t>(static_cast<size_t>(sh.rows) * sizeof(_Tp)), stream));
      cusparseDnVecDescr_t vy_scratch{};
      cusparse_safe_call(cusparseCreateDnVec(&vy_scratch, sh.rows, y_scratch, dt));
      cusparse_safe_call(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        mat,
        vx,
        &beta,
        vy_scratch,
        dt,
        CUSPARSE_SPMV_CSR_ALG2,
        workspace));
      cuda_safe_call(cudaStreamSynchronize(stream));
      cusparse_safe_call(cusparseDestroyDnVec(vy_scratch));
      wplace.deallocate(y_scratch, static_cast<size_t>(sh.rows) * sizeof(_Tp), stream);
    }
    bound_x = x;
    bound_y = y;
    built   = true;
  }

  void run(cusparseHandle_t handle,
           const csr_shard<_Tp>& sh,
           ::std::int64_t cols,
           const _Tp* x,
           _Tp* y,
           _Tp alpha,
           _Tp beta,
           cudaStream_t stream)
  {
    if (!built)
    {
      build(handle, sh, cols, x, y, stream);
    }
    else
    {
      // The handle is shared per place across matrices: rebind its stream on
      // every call.
      cusparse_safe_call(cusparseSetStream(handle, stream));
      if (x != bound_x)
      {
        cusparse_safe_call(cusparseDnVecSetValues(vx, const_cast<_Tp*>(x)));
        bound_x = x;
      }
      if (y != bound_y)
      {
        cusparse_safe_call(cusparseDnVecSetValues(vy, y));
        bound_y = y;
      }
    }
    const cudaDataType dt = cusparse_data_type<_Tp>::value;
    cusparse_safe_call(cusparseSpMV(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy, dt, CUSPARSE_SPMV_CSR_ALG2, workspace));
  }

  //! As `run`, with DEVICE-RESIDENT alpha/beta (the consumer-library pointer
  //! mode): binds CUSPARSE_POINTER_MODE_DEVICE for the launch and restores
  //! host mode after (build/warm-up always run in host mode).
  void run_device_scalars(
    cusparseHandle_t handle,
    const csr_shard<_Tp>& sh,
    ::std::int64_t cols,
    const _Tp* x,
    _Tp* y,
    const _Tp* d_alpha,
    const _Tp* d_beta,
    cudaStream_t stream)
  {
    if (!built)
    {
      build(handle, sh, cols, x, y, stream);
    }
    else
    {
      cusparse_safe_call(cusparseSetStream(handle, stream));
      if (x != bound_x)
      {
        cusparse_safe_call(cusparseDnVecSetValues(vx, const_cast<_Tp*>(x)));
        bound_x = x;
      }
      if (y != bound_y)
      {
        cusparse_safe_call(cusparseDnVecSetValues(vy, y));
        bound_y = y;
      }
    }
    const cudaDataType dt = cusparse_data_type<_Tp>::value;
    cusparse_safe_call(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE));
    const cusparseStatus_t st = cusparseSpMV(
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE, d_alpha, mat, vx, d_beta, vy, dt, CUSPARSE_SPMV_CSR_ALG2, workspace);
    cusparse_safe_call(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
    cusparse_safe_call(st);
  }

  ~spmv_shard_plan()
  {
    // Best-effort teardown (no throwing from destructors)
    if (vx)
    {
      cusparseDestroyDnVec(vx);
    }
    if (vy)
    {
      cusparseDestroyDnVec(vy);
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
 * @brief MATRIX-BOUND SpMV state: one lazily built per-shard pipeline
 * (descriptors + workspace + preprocessed CSR_ALG2 plan) against the shard's
 * fixed addresses. The matrix and the handles must outlive the plan.
 */
template <typename _Tp>
class spmv_plan
{
public:
  spmv_plan(cusparse_handles& handles, sharded_csr<_Tp>& A)
      : handles_(&handles)
      , A_(&A)
      , plans_(A.num_shards())
  {
    if (A.num_shards() != handles.group().size())
    {
      _CCCL_THROW(::std::invalid_argument, "spmv_plan: matrix was not partitioned over this group's places");
    }
  }

  spmv_plan(spmv_plan&&)                 = default;
  spmv_plan(const spmv_plan&)            = delete;
  spmv_plan& operator=(const spmv_plan&) = delete;

  sharded_csr<_Tp>& matrix() const
  {
    return *A_;
  }
  cusparse_handles& handles() const
  {
    return *handles_;
  }
  reserved::spmv_shard_plan<_Tp>& shard_plan(size_t i)
  {
    return plans_[i];
  }

private:
  cusparse_handles* handles_;
  sharded_csr<_Tp>* A_;
  ::std::vector<reserved::spmv_shard_plan<_Tp>> plans_;
};

/**
 * @brief Localized cuSPARSE SpMV: y = alpha * A * x + beta * y, one confined
 * deterministic (CSR_ALG2) call per shard of the plan's matrix, on the
 * shard's stream. Row partition => disjoint y blocks, no combine.
 *
 * @param plan  Caller-held matrix-bound state (see `spmv_plan`)
 * @param x     Dense operand: device pointer to A.num_cols() values readable
 *              from every shard's place
 * @param y     Row-partitioned output matching the matrix
 *              (`A.make_row_partitioned()`)
 */
template <typename _Tp>
void spmv(spmv_plan<_Tp>& plan, const _Tp* x, sharded_array<_Tp>& y, _Tp alpha = _Tp{1}, _Tp beta = _Tp{0})
{
  auto& A           = plan.matrix();
  const auto shards = reserved::matched_output_shards(A, y, 1, "sharded::spmv");
  for (const auto& [i, y_ptr] : shards)
  {
    auto& sh = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    plan.shard_plan(i).run(plan.handles().get(i), sh, A.num_cols(), x, y_ptr, alpha, beta, sh.stream);
  }
}

/**
 * @brief Localized SpMV into a CONTIGUOUS output: y is one device pointer to
 * A.num_rows() values (e.g. a caller library's own whole-device buffer);
 * each shard writes its disjoint row block at y + row_begin. The
 * integration-facing overload: no sharded_array required on the output side.
 */
template <typename _Tp>
void spmv(spmv_plan<_Tp>& plan, const _Tp* x, _Tp* y, _Tp alpha = _Tp{1}, _Tp beta = _Tp{0})
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
        "sharded::spmv: shard " + ::std::to_string(i) + " has rows but no nonzeros; adjust the row boundaries");
    }
    places::exec_place_scope scope(sh.exec);
    plan.shard_plan(i).run(plan.handles().get(i), sh, A.num_cols(), x, y + sh.row_begin, alpha, beta, sh.stream);
  }
}

/**
 * @brief As the contiguous-output `spmv`, with DEVICE-RESIDENT alpha/beta
 * (consumer libraries running their handles in device pointer mode).
 */
template <typename _Tp>
void spmv(spmv_plan<_Tp>& plan, const _Tp* x, _Tp* y, const _Tp* d_alpha, const _Tp* d_beta)
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
        "sharded::spmv: shard " + ::std::to_string(i) + " has rows but no nonzeros; adjust the row boundaries");
    }
    places::exec_place_scope scope(sh.exec);
    plan.shard_plan(i).run_device_scalars(
      plan.handles().get(i), sh, A.num_cols(), x, y + sh.row_begin, d_alpha, d_beta, sh.stream);
  }
}
} // namespace cuda::experimental::sharded
