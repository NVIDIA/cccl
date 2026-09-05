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
 * @brief Placement-localized sparse products over `sharded_csr` with cuSPARSE
 *        as the per-place engine: `sharded::spmv` / `sharded::spmm` over
 *        EXPLICIT caller-held state (`cusparse_handles`, `spmv_plan` /
 *        `spmm_plan`), plus the measured-rebalance utilities
 *        `spmv_shard_times` / `spmm_shard_times`.
 *
 * OPT-IN VENDOR HEADER — not included by `<cuda/experimental/sharded.cuh>`
 * (same model as the cuRAND tier in `random.cuh`); consumers link cuSPARSE.
 * The `sharded_csr` container never depends on this header.
 *
 * The structure is the ladder's usual two tiers with a closed library in the
 * engine slot: the container owns the row partition and placement; the engine
 * tier is ONE confined cuSPARSE call per shard, on the shard's place stream.
 * A row partition makes the output row blocks disjoint, so there is never a
 * combine step.
 *
 * LIBRARY STATE IS EXPLICIT, split by its natural scope, and the caller can
 * read every lifetime off the page:
 *  - `cusparse_handles` — PLACE-BOUND: one `cusparseHandle_t` per place of a
 *    group, created lazily under the place's exec scope, shared by every
 *    matrix and plan built over it. Create it once next to the group; it must
 *    outlive the plans. (Creating handles per call measurably serializes on
 *    the host — the cuRAND tier's generator-lifecycle lesson.)
 *  - `spmv_plan` / `spmm_plan` — MATRIX-BOUND: per-shard descriptors,
 *    workspace and preprocessed plan, built lazily on first run against the
 *    shard's fixed addresses, reused across calls (later calls only rebind
 *    the dense pointers and the handle's stream). The plan references its
 *    matrix and handles: both must outlive it.
 *
 * Dense operands are plain device pointers readable from every place (for
 * example one whole-device allocation): which COPIES of a re-read operand
 * should exist, and when they go stale, is a coherence question that belongs
 * to the binding tier. Outputs are row-partitioned `sharded_array`s
 * (contiguous backing included), typically `sharded_csr::make_row_partitioned`.
 *
 * ASYNCHRONOUS with respect to the host: work is enqueued on the shards'
 * streams; join with `barrier(...)`, the output's `join_into`, or the group.
 * Not thread-safe per plan (mutable per-matrix state): externally serialize
 * concurrent calls on the same plan.
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
#  error "<cuda/experimental/__sharded/sparse.cuh> requires the cuSPARSE headers (cusparse.h) to be installed"
#endif // !__has_include(<cusparse.h>)

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/source_location>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/sharded_csr.cuh>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Throw `std::runtime_error` when a cuSPARSE call does not return
 * `CUSPARSE_STATUS_SUCCESS` (the sharded-scope counterpart of
 * `cuda_safe_call`, following the same optional-vendor-status precedent).
 */
inline void cusparse_safe_call(cusparseStatus_t status,
                               const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    _CCCL_THROW(::std::runtime_error,
                ::std::string(loc.file_name()) + "(" + ::std::to_string(loc.line())
                  + "): cuSPARSE error: " + cusparseGetErrorString(status));
  }
}

namespace reserved
{
/// @brief Maps element types to cuSPARSE data types (FP64 is the primary
/// target; FP32 is provided for completeness).
template <typename _Tp>
struct cusparse_data_type;

template <>
struct cusparse_data_type<double>
{
  static constexpr cudaDataType value = CUDA_R_64F;
};

template <>
struct cusparse_data_type<float>
{
  static constexpr cudaDataType value = CUDA_R_32F;
};

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

/**
 * @brief Match an output array against a matrix's row partition.
 *
 * `sharded_array` allocation skips zero-size shards, so an output for a
 * matrix with row-less shards has fewer shards than the matrix. Walk both
 * sides, pair every rows>0 matrix shard with the next output shard, and
 * validate sizes; also refuse nnz==0 shards, which cuSPARSE descriptors do
 * not support.
 *
 * @return (matrix shard index, output pointer) for every participating shard
 */
template <typename _Tp>
::std::vector<::std::pair<size_t, _Tp*>>
matched_output_shards(const sharded_csr<_Tp>& A, sharded_array<_Tp>& out, ::std::int64_t n_cols, const char* what)
{
  ::std::vector<::std::pair<size_t, _Tp*>> pairs;
  size_t out_idx = 0;
  for (size_t i = 0; i < A.num_shards(); i++)
  {
    const auto& sh = A.shard(i);
    if (sh.rows == 0)
    {
      continue;
    }
    if (sh.nnz == 0)
    {
      _CCCL_THROW(::std::invalid_argument,
                  ::std::string(what) + ": shard " + ::std::to_string(i)
                    + " has rows but no nonzeros; adjust the row boundaries");
    }
    if (out_idx >= out.num_shards()
        || out.shard(out_idx).size != static_cast<size_t>(sh.rows) * static_cast<size_t>(n_cols))
    {
      _CCCL_THROW(::std::invalid_argument,
                  ::std::string(what) + ": output is not row-partitioned like the matrix "
                    + "(use sharded_csr::make_row_partitioned)");
    }
    pairs.emplace_back(i, out.shard(out_idx).data);
    out_idx++;
  }
  if (out_idx != out.num_shards())
  {
    _CCCL_THROW(::std::invalid_argument,
                ::std::string(what) + ": output has more shards than the matrix has row ranges");
  }
  return pairs;
}

/// @brief Time `iters` runs of `body` on @p stream (host-submitted, event
/// bracketed), returning mean milliseconds per run.
template <typename _Body>
double time_on_stream(cudaStream_t stream, int warmup, int iters, _Body&& body)
{
  using ::cuda::experimental::places::cuda_safe_call;
  for (int w = 0; w < warmup; w++)
  {
    body();
  }
  cuda_safe_call(cudaStreamSynchronize(stream));
  cudaEvent_t e0{}, e1{};
  cuda_safe_call(cudaEventCreate(&e0));
  cuda_safe_call(cudaEventCreate(&e1));
  cuda_safe_call(cudaEventRecord(e0, stream));
  for (int it = 0; it < iters; it++)
  {
    body();
  }
  cuda_safe_call(cudaEventRecord(e1, stream));
  cuda_safe_call(cudaEventSynchronize(e1));
  float t = 0;
  cuda_safe_call(cudaEventElapsedTime(&t, e0, e1));
  cuda_safe_call(cudaEventDestroy(e0));
  cuda_safe_call(cudaEventDestroy(e1));
  return static_cast<double>(t) / iters;
}
} // namespace reserved

/**
 * @brief PLACE-BOUND cuSPARSE state: one handle per place of a group, created
 * lazily under the place's exec scope on first use, shared by every matrix
 * and plan built over the group.
 *
 * Create ONE of these next to the group and pass it to the plans; it must
 * outlive them. The stream is never bound here — plans rebind it per call,
 * since one handle serves every matrix at its place. Lazy creation is
 * mutex-guarded (thread-safe); teardown is best-effort (no throwing).
 */
class cusparse_handles
{
public:
  explicit cusparse_handles(places::place_group& group)
      : group_(&group)
      , handles_(group.size(), nullptr)
  {}

  cusparse_handles(const cusparse_handles&)            = delete;
  cusparse_handles& operator=(const cusparse_handles&) = delete;

  /// @brief The handle of the idx-th place, created on first use under the
  /// place's exec scope (so lazy library state lands in the confined context
  /// that runs the calls).
  cusparseHandle_t get(size_t place_idx)
  {
    _CCCL_ASSERT(place_idx < handles_.size(), "cusparse_handles: place index out of range");
    ::std::lock_guard<::std::mutex> lock(mutex_);
    if (!handles_[place_idx])
    {
      places::exec_place_scope scope(group_->place(place_idx));
      cusparse_safe_call(cusparseCreate(&handles_[place_idx]));
    }
    return handles_[place_idx];
  }

  places::place_group& group() const
  {
    return *group_;
  }

  ~cusparse_handles()
  {
    for (cusparseHandle_t h : handles_)
    {
      if (h)
      {
        cusparseDestroy(h); // best-effort
      }
    }
  }

private:
  places::place_group* group_;
  ::std::mutex mutex_;
  ::std::vector<cusparseHandle_t> handles_;
};

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
 * @brief Measure each shard's solo (confined) SpMV time through the exact
 * call path `spmv` uses (same plans, streams, places). Feed the result to
 * `sharded_csr::time_balanced_boundaries` to rebalance a time-skewed split.
 * Row-less shards report 0.
 */
template <typename _Tp>
::std::vector<double> spmv_shard_times(
  spmv_plan<_Tp>& plan,
  const _Tp* x,
  sharded_array<_Tp>& y,
  _Tp alpha  = _Tp{1},
  _Tp beta   = _Tp{0},
  int warmup = 3,
  int iters  = 10)
{
  auto& A           = plan.matrix();
  const auto shards = reserved::matched_output_shards(A, y, 1, "sharded::spmv_shard_times");
  ::std::vector<double> ms(A.num_shards(), 0.0);
  for (const auto& pair : shards)
  {
    const size_t i = pair.first;
    _Tp* y_ptr     = pair.second;
    auto& sh       = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    cusparseHandle_t handle = plan.handles().get(i);
    ms[i]                   = reserved::time_on_stream(sh.stream, warmup, iters, [&, i, y_ptr, handle] {
      plan.shard_plan(i).run(handle, sh, A.num_cols(), x, y_ptr, alpha, beta, sh.stream);
    });
  }
  return ms;
}

/**
 * @brief Measure each shard's solo (confined) SpMM time through the exact
 * call path `spmm` uses. See `spmv_shard_times`.
 */
template <typename _Tp>
::std::vector<double> spmm_shard_times(
  spmm_plan<_Tp>& plan,
  const _Tp* B,
  sharded_array<_Tp>& C,
  _Tp alpha  = _Tp{1},
  _Tp beta   = _Tp{0},
  int warmup = 3,
  int iters  = 10)
{
  auto& A           = plan.matrix();
  const auto shards = reserved::matched_output_shards(A, C, plan.n_cols(), "sharded::spmm_shard_times");
  ::std::vector<double> ms(A.num_shards(), 0.0);
  for (const auto& pair : shards)
  {
    const size_t i = pair.first;
    _Tp* C_ptr     = pair.second;
    auto& sh       = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    cusparseHandle_t handle = plan.handles().get(i);
    ms[i]                   = reserved::time_on_stream(sh.stream, warmup, iters, [&, i, C_ptr, handle] {
      plan.shard_plan(i).run(handle, sh, A.num_cols(), plan.n_cols(), B, C_ptr, alpha, beta, sh.stream);
    });
  }
  return ms;
}
} // namespace cuda::experimental::sharded
