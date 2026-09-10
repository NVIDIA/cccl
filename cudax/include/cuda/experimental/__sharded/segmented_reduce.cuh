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
 * Two spellings of the segments description, same precondition (no segment
 * crosses a value-shard boundary), same output:
 *
 * - SHARDED SHARD-LOCAL BEGIN/END VIEWS (the container form): two sharded
 *   views of offsets co-partitioned with the output, `seg_begin[i]` /
 *   `seg_end[i]` bounding segment `i` within the shard's own input piece
 *   (offsets REBASED per shard). This is what `sharded_csr` stores; for
 *   CSR-shaped data both views are typically shifted aliases of one
 *   (n+1)-entry rebased row-offsets buffer per shard — `begin =
 *   offsets[0..n)`, `end = offsets[1..n+1)` — which the non-owning view
 *   tier expresses directly (`make_sharded_view` over shifted spans).
 * - ONE WHOLE GLOBAL OFFSETS ARRAY (the classic CSR / `reduce_by_key`-shaped
 *   form, zero-copy): a single `num_segments + 1` random-access sequence of
 *   GLOBAL value positions, readable from every place. Segment `s` of the
 *   output's global index space reduces `in` over global value positions
 *   `[offsets[s], offsets[s + 1])`. Each shard rebases on the fly through a
 *   transform iterator (`offsets[s] - in.shard(g).global_offset`): no
 *   per-shard offsets buffers, no copies.
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

#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace __detail
{
//! Rebases a GLOBAL value offset into a shard-local one: `v - base`, where
//! `base` is the value shard's `global_offset`. Stateless apart from the
//! base, so the transform iterator is a plain pair (pointer, base) that CUB
//! copies into its kernels.
template <class _Off>
struct __rebase_offset
{
  _Off __base;

  _CCCL_HOST_DEVICE_API _Off operator()(_Off __v) const noexcept
  {
    return static_cast<_Off>(__v - __base);
  }
};
} // namespace __detail

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
/**
 * @brief Segmented reduce with ONE whole global offsets array (the classic
 * CSR / `reduce_by_key`-shaped spelling, zero-copy): for shard `g` of @p out
 * covering global segments `[s0, s1) = [out.shard(g).global_offset,
 * out.shard(g).global_offset + out.shard(g).size)`, segment `s` reduces
 * @p in over GLOBAL value positions `[offsets[s], offsets[s + 1])` — one
 * `cub::DeviceSegmentedReduce` per shard on `envs[g]`, scratch from the
 * environment's memory resource (stream-ordered).
 *
 * @p offsets is a random-access iterator (typically a plain device pointer)
 * over `num_segments + 1` GLOBAL value positions in @p in's global index
 * space. It is NOT a sharded view: it must be readable from every place the
 * output's shards live on (a whole-device allocation is fine within one
 * device; on several devices, peer-accessible or managed memory). Shard `g`
 * reads `offsets[s0 .. s1]` through a transform iterator that subtracts the
 * value shard's base, `offsets[s] - in.shard(g).global_offset`, so CUB sees
 * shard-local offsets without any per-shard offsets buffer or copy.
 *
 * Partitioning contract:
 * - @p in lives in the values index space and is only required to be
 *   shard-count ALIGNED with @p out: shard `g`'s segments select from shard
 *   `g`'s input piece.
 * - Precondition: segments do not cross the value cut, i.e. for every shard
 *   `g`, `offsets[s0] == in.shard(g).global_offset` and
 *   `offsets[s1] == in.shard(g).global_offset + in.shard(g).size`. The
 *   SYNCHRONOUS form checks exactly these two values per shard (a small
 *   device-to-host read of 2P integers, after the entry refusals and before
 *   any work is enqueued) when @p offsets is a contiguous iterator, and
 *   refuses with `std::invalid_argument`. The ASYNCHRONOUS form cannot read
 *   the offsets back without a host synchronization (which would also break
 *   graph capture), so there the precondition is the caller's, exactly like
 *   the shard-local form's. Interior offset values are read on device;
 *   non-monotone or out-of-range offsets are undefined behavior, as they are
 *   for the underlying device-scope primitive.
 *
 * Empty segments (`offsets[s] == offsets[s + 1]`) receive `init`. Empty
 * output shards are skipped.
 *
 * Contract per the call environment, as for the map family: stream present
 * (`async_call_env`) = asynchronous (lane-ordered by default, capture-legal;
 * `composition::bracketed` seals the call against the call stream instead);
 * no stream = synchronous convenience (refused under `sync_policy::forbid`
 * and under capture).
 *
 * @throws std::invalid_argument on environment shortfall, on in/out shard
 *         count mismatch, or (synchronous form, contiguous @p offsets) when
 *         a shard's first/last offsets do not match the value cut.
 */
_CCCL_TEMPLATE(
  class _SIn, class _Envs, class _OffsetIt, class _SOut, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>
    _CCCL_AND ::cuda::std::random_access_iterator<_OffsetIt> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>)
_CCCL_HOST_API void segmented_reduce(
  const _SIn& in,
  _Envs&& envs,
  const _OffsetIt offsets,
  _SOut&& out,
  _ReduceOp op,
  _Tp init,
  const _CallEnv& call_env = {})
{
  using _Off  = ::cuda::std::iter_value_t<_OffsetIt>;
  using _Diff = ::cuda::std::iter_difference_t<_OffsetIt>;
  static_assert(::cuda::std::is_integral_v<_Off>, "sharded::segmented_reduce: offsets must be integral");

  const ::std::size_t num_shards = reserved::__shard_count(out);
  if (reserved::__shard_count(in) != num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::segmented_reduce: in/out shard count mismatch");
  }

  if constexpr (!async_call_env<_CallEnv> && ::cuda::std::contiguous_iterator<_OffsetIt>)
  {
    // Synchronous form: the value-cut precondition is cheap to verify — two
    // offsets per shard. Entry refusals first (nothing enqueued before they
    // are decided), then a small stream-ordered read-back on each lane.
    require_sync_allowed(call_env, "sharded::segmented_reduce");
    places::check_not_capturing(nullptr, "sharded::segmented_reduce");
    if (reserved::__env_count(envs) < num_shards)
    {
      _CCCL_THROW(::std::invalid_argument, "sharded::segmented_reduce: fewer environments than shards");
    }
    for (::std::size_t g = 0; g < num_shards; g++)
    {
      places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::segmented_reduce");
    }
    ::std::vector<_Off> cut(2 * num_shards);
    for (::std::size_t g = 0; g < num_shards; g++)
    {
      const auto& o = out.shard(g);
      if (o.size == 0)
      {
        continue;
      }
      const auto s          = ::cuda::get_stream(envs[g]);
      const _OffsetIt first = offsets + static_cast<_Diff>(o.global_offset);
      const _OffsetIt last  = first + static_cast<_Diff>(o.size);
      cuda_safe_call(
        cudaMemcpyAsync(&cut[2 * g], ::cuda::std::to_address(first), sizeof(_Off), cudaMemcpyDefault, s.get()));
      cuda_safe_call(
        cudaMemcpyAsync(&cut[2 * g + 1], ::cuda::std::to_address(last), sizeof(_Off), cudaMemcpyDefault, s.get()));
      cuda_safe_call(cudaStreamSynchronize(s.get()));
      const auto& i = in.shard(g);
      if (static_cast<::std::size_t>(cut[2 * g]) != static_cast<::std::size_t>(i.global_offset)
          || static_cast<::std::size_t>(cut[2 * g + 1]) != static_cast<::std::size_t>(i.global_offset) + i.size)
      {
        _CCCL_THROW(::std::invalid_argument,
                    "sharded::segmented_reduce: offsets cross the value cut at shard " + ::std::to_string(g)
                      + " (offsets[" + ::std::to_string(o.global_offset) + "] = " + ::std::to_string(cut[2 * g])
                      + ", offsets[" + ::std::to_string(o.global_offset + o.size)
                      + "] = " + ::std::to_string(cut[2 * g + 1]) + "; value shard covers ["
                      + ::std::to_string(i.global_offset) + ", " + ::std::to_string(i.global_offset + i.size) + "))");
      }
    }
  }

  __detail::__generic_map(
    out, envs, call_env, "sharded::segmented_reduce", [&](::std::size_t g, const auto& o, cudaStream_t s) {
      const ::cuda::stream_ref stream{s};
      const auto& i = in.shard(g);
      // Global -> shard-local offsets on the fly: offsets[s0 + k] - value base.
      const auto begin = ::cuda::make_transform_iterator(
        offsets + static_cast<_Diff>(o.global_offset),
        __detail::__rebase_offset<_Off>{static_cast<_Off>(i.global_offset)});
      const auto end = begin + 1;
      // Two-phase CUB: size query (host-only, no work recorded), then run
      // with stream-ordered scratch from the shard's environment.
      void* d_temp        = nullptr;
      ::std::size_t bytes = 0;
      cuda_safe_call(cub::DeviceSegmentedReduce::Reduce(
        d_temp, bytes, i.data, o.data, static_cast<int>(o.size), begin, end, op, init, s));
      auto mr = ::cuda::mr::get_memory_resource(envs[g]);
      d_temp  = mr.allocate(stream, bytes, 256);
      SCOPE(fail)
      {
        mr.deallocate(stream, d_temp, bytes, 256);
      };
      cuda_safe_call(cub::DeviceSegmentedReduce::Reduce(
        d_temp, bytes, i.data, o.data, static_cast<int>(o.size), begin, end, op, init, s));
      mr.deallocate(stream, d_temp, bytes, 256);
    });
}

/**
 * @brief Whole-offsets segmented reduce with environments derived from a
 * self-bound output (`default_envs(out)`): the container-materialized
 * spelling.
 */
_CCCL_TEMPLATE(class _SIn, class _OffsetIt, class _SOut, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND ::cuda::std::random_access_iterator<_OffsetIt>
                 _CCCL_AND self_bound<::cuda::std::remove_cvref_t<_SOut>>)
_CCCL_HOST_API void segmented_reduce(
  const _SIn& in, const _OffsetIt offsets, _SOut&& out, _ReduceOp op, _Tp init, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(out);
  sharded::segmented_reduce(in, envs, offsets, ::cuda::std::forward<_SOut>(out), op, init, call_env);
}
} // namespace cuda::experimental::sharded
