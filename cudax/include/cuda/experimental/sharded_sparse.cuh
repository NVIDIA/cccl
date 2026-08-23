//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Placement-localized sparse products over `sharded_csr` with cuSPARSE
 *        as the per-place engine: `sharded::spmv`, `sharded::spmm`, and the
 *        measured-rebalance utilities `spmv_shard_times` / `spmm_shard_times`.
 *
 * OPT-IN VENDOR HEADER — not included by `<cuda/experimental/sharded.cuh>`.
 * Including it requires the cuSPARSE development headers and linking against
 * cuSPARSE (same model as `<cuda/experimental/cufile.cuh>`); the
 * `sharded_csr` container itself lives in the umbrella and never depends on
 * cuSPARSE.
 *
 * The structure is the ladder's usual two tiers with a closed library in the
 * engine slot: the container tier (`sharded_csr`) owns the row partition and
 * placement; the engine tier is ONE confined cuSPARSE call per shard, on the
 * shard's place stream. A row partition makes the output row blocks disjoint,
 * so there is never a combine step. Library state is split by natural scope:
 * the `cusparseHandle_t` is PER PLACE and lives in the `place_group`'s
 * library-state cache (created lazily on first use, shared by every matrix
 * built over the group, destroyed with the group — the `raft::handle_t`
 * precedent); per-(shard, op) state — descriptors, workspace, preprocessed
 * plan — is matrix-bound, created lazily on first call into the container's
 * `lib_state()` slot, built against the shard's fixed addresses, and reused
 * for the matrix's lifetime (later calls only rebind the dense pointers when
 * they change, and rebind the handle's stream per call). The group must
 * outlive the matrices built over it (the existing group contract).
 *
 * Dense operands are plain device pointers readable from every place (for
 * example one whole-device allocation): which COPIES of a re-read operand
 * should exist, and when they go stale, is a coherence question that belongs
 * to the binding tier — pass an STF-managed per-place instance pointer here
 * when composing with `logical_data` replication. Outputs are row-partitioned
 * `sharded_array`s (contiguous backing included), typically from
 * `sharded_csr::make_row_partitioned`.
 */

#ifndef __CUDAX_SHARDED_SPARSE_CUH
#define __CUDAX_SHARDED_SPARSE_CUH

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !__has_include(<cusparse.h>)
#  error "<cuda/experimental/sharded_sparse.cuh> requires the cuSPARSE headers (cusparse.h) to be installed"
#endif // !__has_include(<cusparse.h>)

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/source_location>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/sharded_csr.cuh>

#include <cstdint>
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
 * @brief Owner of one place's `cusparseHandle_t`, stored in the
 * `place_group`'s per-place library-state cache.
 *
 * The handle is a place-bound resource (not a matrix-bound one), so its
 * scope is the group's: created lazily on first sparse product at that
 * place, shared by every `sharded_csr` built over the group, destroyed at
 * group teardown. Destroying a container never destroys the handle.
 */
struct cusparse_place_handle
{
  cusparseHandle_t handle{};

  cusparse_place_handle()                                        = default;
  cusparse_place_handle(const cusparse_place_handle&)            = delete;
  cusparse_place_handle& operator=(const cusparse_place_handle&) = delete;

  ~cusparse_place_handle()
  {
    // Best-effort teardown (no throwing from destructors)
    if (handle)
    {
      cusparseDestroy(handle);
    }
  }
};

/**
 * @brief The per-place cuSPARSE handle of the idx-th place of @p group,
 * created on first use into the group's library-state cache.
 *
 * Expected to be called with the place's exec context active
 * (`exec_place_scope`), so a first-use creation happens in the confined
 * context that runs the calls. The stream is NOT bound here: callers rebind
 * it per call (`cusparseSetStream`), since the same handle serves every
 * matrix built over the group.
 */
inline cusparseHandle_t get_place_cusparse_handle(place_group& group, size_t place_idx)
{
  auto& holder = group.lib_state<cusparse_place_handle>(place_idx, [] {
    auto* s = new cusparse_place_handle();
    _CCCL_TRY
    {
      cusparse_safe_call(cusparseCreate(&s->handle));
    }
    _CCCL_CATCH_ALL
    {
      delete s;
      _CCCL_RETHROW;
    }
    return s;
  });
  return holder.handle;
}

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

template <typename _Tp>
struct spmv_state
{
  ::std::vector<spmv_shard_plan<_Tp>> plans;
};

template <typename _Tp>
struct spmm_state
{
  ::std::vector<spmm_shard_plan<_Tp>> plans;
};

template <typename _Tp>
spmv_state<_Tp>& get_spmv_state(sharded_csr<_Tp>& A)
{
  return A.template lib_state<spmv_state<_Tp>>("cusparse_spmv", [&] {
    auto* s = new spmv_state<_Tp>();
    s->plans.resize(A.num_shards());
    return s;
  });
}

template <typename _Tp>
spmm_state<_Tp>& get_spmm_state(sharded_csr<_Tp>& A, ::std::int64_t n_cols)
{
  return A.template lib_state<spmm_state<_Tp>>("cusparse_spmm:" + ::std::to_string(n_cols), [&] {
    auto* s = new spmm_state<_Tp>();
    s->plans.resize(A.num_shards());
    return s;
  });
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
 * @brief Localized cuSPARSE SpMV: y = alpha * A * x + beta * y.
 *
 * One confined cuSPARSE call per shard of A, at the shard's exec place on the
 * shard's stream (CSR_ALG2, deterministic); the y row range of each shard is
 * disjoint, so there is no combine. The per-shard plans are built lazily on
 * first call and reused across calls (see the file-level comment).
 *
 * ASYNCHRONOUS with respect to the host: work is enqueued on the shards'
 * streams; synchronize with `y.sync()` (or the group's `sync()`).
 *
 * Not thread-safe per matrix: the plans are mutable per-matrix state, so
 * concurrent calls on the same `sharded_csr` must be externally serialized
 * (calls on the same shard stream already serialize on the device).
 *
 * @param group Place group the matrix was built over (validation/resources)
 * @param A     Row-partitioned CSR matrix
 * @param x     Dense operand: device pointer to A.num_cols() values readable
 *              from every shard's place
 * @param y     Row-partitioned output, layout matching A
 *              (`A.make_row_partitioned()`, contiguous backing supported)
 */
template <typename _Tp>
void spmv(
  place_group& group, sharded_csr<_Tp>& A, const _Tp* x, sharded_array<_Tp>& y, _Tp alpha = _Tp{1}, _Tp beta = _Tp{0})
{
  if (A.num_shards() != group.size())
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::spmv: matrix was not partitioned over this group's places");
  }
  auto& st          = reserved::get_spmv_state(A);
  const auto shards = reserved::matched_output_shards(A, y, 1, "sharded::spmv");
  for (const auto& [i, y_ptr] : shards)
  {
    auto& sh = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    cusparseHandle_t handle = reserved::get_place_cusparse_handle(group, i);
    st.plans[i].run(handle, sh, A.num_cols(), x, y_ptr, alpha, beta, sh.stream);
  }
}

/**
 * @brief Localized cuSPARSE SpMM: C = alpha * A * B + beta * C
 *        (B and C row-major with leading dimension n_cols).
 *
 * One confined cuSPARSE call per shard of A on the shard's stream (CSR_ALG3,
 * deterministic); each shard's C row block is disjoint, so there is no
 * combine. Same lazy plan reuse, asynchrony and thread-safety contract as
 * `spmv`.
 *
 * @param group  Place group the matrix was built over
 * @param A      Row-partitioned CSR matrix
 * @param B      Dense operand: device pointer to A.num_cols() * n_cols
 *               values (row-major) readable from every shard's place
 * @param C      Row-partitioned output, `A.make_row_partitioned(n_cols)`
 *               (contiguous backing supported)
 * @param n_cols Number of dense columns
 */
template <typename _Tp>
void spmm(place_group& group,
          sharded_csr<_Tp>& A,
          const _Tp* B,
          sharded_array<_Tp>& C,
          ::std::int64_t n_cols,
          _Tp alpha = _Tp{1},
          _Tp beta  = _Tp{0})
{
  if (A.num_shards() != group.size())
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::spmm: matrix was not partitioned over this group's places");
  }
  auto& st          = reserved::get_spmm_state(A, n_cols);
  const auto shards = reserved::matched_output_shards(A, C, n_cols, "sharded::spmm");
  for (const auto& [i, C_ptr] : shards)
  {
    auto& sh = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    cusparseHandle_t handle = reserved::get_place_cusparse_handle(group, i);
    st.plans[i].run(handle, sh, A.num_cols(), n_cols, B, C_ptr, alpha, beta, sh.stream);
  }
}

/**
 * @brief Measure each shard's solo (confined) SpMV time, through the exact
 *        call path `spmv` uses (same plans, streams and places).
 *
 * Runs each shard alone (warmup + iters iterations, event-bracketed on the
 * shard's stream) and returns per-shard mean milliseconds. Feed the result to
 * `sharded_csr::time_balanced_boundaries` to rebalance a time-skewed split:
 * one calibration round is amortized over every subsequent call on the
 * rebuilt matrix. Row-less shards report 0.
 */
template <typename _Tp>
::std::vector<double> spmv_shard_times(
  place_group& group,
  sharded_csr<_Tp>& A,
  const _Tp* x,
  sharded_array<_Tp>& y,
  _Tp alpha  = _Tp{1},
  _Tp beta   = _Tp{0},
  int warmup = 3,
  int iters  = 10)
{
  if (A.num_shards() != group.size())
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::spmv_shard_times: matrix was not partitioned over this group's places");
  }
  auto& st          = reserved::get_spmv_state(A);
  const auto shards = reserved::matched_output_shards(A, y, 1, "sharded::spmv_shard_times");
  ::std::vector<double> ms(A.num_shards(), 0.0);
  for (const auto& pair : shards)
  {
    const size_t i = pair.first;
    _Tp* y_ptr     = pair.second;
    auto& sh       = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    cusparseHandle_t handle = reserved::get_place_cusparse_handle(group, i);
    ms[i]                   = reserved::time_on_stream(sh.stream, warmup, iters, [&, i, y_ptr, handle] {
      st.plans[i].run(handle, sh, A.num_cols(), x, y_ptr, alpha, beta, sh.stream);
    });
  }
  return ms;
}

/**
 * @brief Measure each shard's solo (confined) SpMM time, through the exact
 *        call path `spmm` uses. See `spmv_shard_times`.
 */
template <typename _Tp>
::std::vector<double> spmm_shard_times(
  place_group& group,
  sharded_csr<_Tp>& A,
  const _Tp* B,
  sharded_array<_Tp>& C,
  ::std::int64_t n_cols,
  _Tp alpha  = _Tp{1},
  _Tp beta   = _Tp{0},
  int warmup = 3,
  int iters  = 10)
{
  if (A.num_shards() != group.size())
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::spmm_shard_times: matrix was not partitioned over this group's places");
  }
  auto& st          = reserved::get_spmm_state(A, n_cols);
  const auto shards = reserved::matched_output_shards(A, C, n_cols, "sharded::spmm_shard_times");
  ::std::vector<double> ms(A.num_shards(), 0.0);
  for (const auto& pair : shards)
  {
    const size_t i = pair.first;
    _Tp* C_ptr     = pair.second;
    auto& sh       = A.shard(i);
    places::exec_place_scope scope(sh.exec);
    cusparseHandle_t handle = reserved::get_place_cusparse_handle(group, i);
    ms[i]                   = reserved::time_on_stream(sh.stream, warmup, iters, [&, i, C_ptr, handle] {
      st.plans[i].run(handle, sh, A.num_cols(), n_cols, B, C_ptr, alpha, beta, sh.stream);
    });
  }
  return ms;
}
} // namespace cuda::experimental::sharded

#endif // __CUDAX_SHARDED_SPARSE_CUH
