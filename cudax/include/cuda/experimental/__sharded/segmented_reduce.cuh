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
 * @brief Segmented reduce over sharded views: per shard,
 *        `out[i] = reduce(in[seg_begin[i] .. seg_end[i]))` via
 *        `cub::DeviceSegmentedReduce` on the shard's environment.
 *
 * Despite the name, this is a member of the MAP family, not the combine
 * family: every segment lives inside one shard, so there is no cross-shard
 * combine stage — the call is per-shard stream-ordered work and is
 * capture-legal in its asynchronous form. This is the primitive that turns
 * segment-structured data (CSR rows, ragged batches) into per-segment
 * aggregates — the SpMV-shaped terminator of neighbor/row reductions.
 *
 * The segments description is two sharded views of offsets, co-partitioned
 * with the output: `seg_begin[i]` / `seg_end[i]` bound segment `i` within
 * the shard's own input piece (shard-local offsets). For CSR-shaped data
 * both views are typically shifted aliases of one (n+1)-entry row-offsets
 * buffer per shard — `begin = offsets[0..n)`, `end = offsets[1..n+1)` —
 * which the non-owning view tier expresses directly (`make_sharded_view`
 * over shifted spans).
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

#include <cub/device/device_segmented_reduce.cuh>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Segmented reduce over sharded views: per shard `g` and segment `i`,
 * `out.shard(g)[i] = op(init, in.shard(g)[seg_begin.shard(g)[i]] ... )` —
 * one `cub::DeviceSegmentedReduce` per shard on `envs[g]`, scratch from the
 * environment's memory resource (stream-ordered).
 *
 * Partitioning contract:
 * - @p out, @p seg_begin and @p seg_end are co-partitioned (same shard
 *   count, identical per-shard regions): one output element and one
 *   [begin, end) pair per segment.
 * - @p in lives in a DIFFERENT index space (values) and is only required to
 *   be shard-count ALIGNED with the output: shard `g`'s segments select
 *   from shard `g`'s input piece. Offsets are shard-local — they index into
 *   `[0, in.shard(g).size]`.
 * - Precondition (v1): segments do not cross shard boundaries. Data
 *   sharded by segment ranges (a vertex-partitioned CSR, ragged batches
 *   split on batch boundaries) satisfies this by construction. Offset
 *   values are read on device; out-of-range offsets are undefined behavior,
 *   as they are for the underlying device-scope primitive.
 *
 * Empty segments (`begin == end`) receive `init`. Empty output shards are
 * skipped.
 *
 * Contract per the call environment, as for the map family: stream present
 * (`async_call_env`) = asynchronous (lane-ordered by default: enqueue on
 * the environments' streams, no call-stream edges, no host synchronization;
 * `composition::bracketed` on the call environment seals the call against
 * the call stream instead; capture-legal — under capture the lanes must
 * already be capturing, or the call refuses at entry); no stream =
 * synchronous convenience (refused under `sync_policy::forbid` and under
 * capture).
 *
 * @throws std::invalid_argument on environment shortfall, on
 *         out/seg_begin/seg_end partition mismatch, or on in/out shard
 *         count mismatch.
 */
_CCCL_TEMPLATE(
  class _SIn,
  class _Envs,
  class _SBegin,
  class _SEnd,
  class _SOut,
  class _ReduceOp,
  class _Tp,
  class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>
    _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SBegin>> _CCCL_AND
      sharded_view<::cuda::std::remove_cvref_t<_SEnd>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>)
_CCCL_HOST_API void segmented_reduce(
  const _SIn& in,
  _Envs&& envs,
  const _SBegin& seg_begin,
  const _SEnd& seg_end,
  _SOut&& out,
  _ReduceOp op,
  _Tp init,
  const _CallEnv& call_env = {})
{
  const ::std::size_t num_shards = reserved::__shard_count(out);
  if (reserved::__shard_count(in) != num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::segmented_reduce: in/out shard count mismatch");
  }
  reserved::__check_copartitioned(out, seg_begin, "sharded::segmented_reduce (out/seg_begin)");
  reserved::__check_copartitioned(out, seg_end, "sharded::segmented_reduce (out/seg_end)");

  __detail::__generic_map(
    out, envs, call_env, "sharded::segmented_reduce", [&](::std::size_t g, const auto& o, cudaStream_t s) {
      const ::cuda::stream_ref stream{s};
      // Two-phase CUB: size query (host-only, no work recorded), then run
      // with stream-ordered scratch from the shard's environment.
      void* d_temp        = nullptr;
      ::std::size_t bytes = 0;
      cuda_safe_call(cub::DeviceSegmentedReduce::Reduce(
        d_temp,
        bytes,
        in.shard(g).data,
        o.data,
        static_cast<int>(o.size),
        seg_begin.shard(g).data,
        seg_end.shard(g).data,
        op,
        init,
        s));
      auto mr = ::cuda::mr::get_memory_resource(envs[g]);
      d_temp  = mr.allocate(stream, bytes, 256);
      SCOPE(fail)
      {
        mr.deallocate(stream, d_temp, bytes, 256);
      };
      cuda_safe_call(cub::DeviceSegmentedReduce::Reduce(
        d_temp,
        bytes,
        in.shard(g).data,
        o.data,
        static_cast<int>(o.size),
        seg_begin.shard(g).data,
        seg_end.shard(g).data,
        op,
        init,
        s));
      mr.deallocate(stream, d_temp, bytes, 256);
    });
}

/**
 * @brief Segmented reduce with environments derived from a self-bound
 * output (`default_envs(out)`): the container-materialized spelling.
 */
_CCCL_TEMPLATE(
  class _SIn, class _SBegin, class _SEnd, class _SOut, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SBegin>> _CCCL_AND
    sharded_view<::cuda::std::remove_cvref_t<_SEnd>> _CCCL_AND self_bound<::cuda::std::remove_cvref_t<_SOut>>)
_CCCL_HOST_API void segmented_reduce(
  const _SIn& in,
  const _SBegin& seg_begin,
  const _SEnd& seg_end,
  _SOut&& out,
  _ReduceOp op,
  _Tp init,
  const _CallEnv& call_env = {})
{
  const auto envs = default_envs(out);
  sharded::segmented_reduce(in, envs, seg_begin, seg_end, ::cuda::std::forward<_SOut>(out), op, init, call_env);
}
} // namespace cuda::experimental::sharded
