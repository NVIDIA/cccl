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
 * @brief Per-shard timing of the SpMV / SpMM call paths
 *        (`spmv_shard_times`, `spmm_shard_times`): the measurements
 *        `sharded_csr::time_balanced_boundaries` consumes to rebalance a
 *        time-skewed row split.
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
#  error "<cuda/experimental/__sharded/sparse/rebalance.cuh> requires the cuSPARSE headers (cusparse.h) to be installed"
#endif // !__has_include(<cusparse.h>)

#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/container/sharded_array.cuh>
#include <cuda/experimental/__sharded/sparse/cusparse.cuh>
#include <cuda/experimental/__sharded/sparse/spmm.cuh>
#include <cuda/experimental/__sharded/sparse/spmv.cuh>

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
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
