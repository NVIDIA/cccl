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
 * @brief Reduction over sharded arrays: each place runs the device-scope
 *        primitive (CUB `DeviceReduce`) on its shard, then the per-place
 *        partials are combined — the same local-primitive-plus-combine
 *        structure the device scope itself uses over blocks.
 *
 * Algorithm temporaries are drawn from each shard's own place through the
 * group's per-place memory resources, so scratch lands where the work runs.
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

#include <cub/device/device_reduce.cuh>

#include <cuda/functional>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
// ============================================================================
// Concept-generic tier (pilot): any sharded_view + allocating environments
// ============================================================================

/**
 * @brief Synchronous reduce over any `sharded_view`: per-shard
 * `cub::DeviceReduce` on the shard's environment (stream + memory resource),
 * per-shard partials staged to pinned host memory, host combine in shard
 * order (deterministic for a fixed shard count).
 *
 * This is the synchronous convenience form: it returns the value to the
 * caller and therefore synchronizes with the host. It refuses under CUDA
 * graph capture and under `sync_policy::forbid` (both before any work is
 * enqueued, leaving all state valid).
 *
 * Requirements: environments must answer `cuda::mr::get_memory_resource`
 * with a stream-ordered-capable resource (`cuda::mr::resource` shape) — the
 * per-shard scratch stays in the asynchronous pipeline.
 *
 * @throws std::invalid_argument when fewer environments than shards are
 *         supplied.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Tp, class _ReduceOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API _Tp
reduce(const _S& data, const _Envs& envs, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env = {})
{
  const ::std::size_t num_shards = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::reduce: fewer environments than shards");
  }
  if (num_shards == 0)
  {
    return init_value;
  }

  // Refusals first, before any CUDA call: this form synchronizes.
  require_sync_allowed(call_env, "sharded::reduce (synchronous form)");
  places::check_not_capturing(nullptr, "sharded::reduce");
  for (const auto g : each(num_shards))
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::reduce");
  }

  // Pinned host staging for the per-shard partials (host-accessible +
  // async-transfer-capable). A per-call cudaMallocHost/cudaFreeHost pair
  // costs close to a millisecond (page pinning), which would dominate the
  // whole combine — the default is a cached thread-local pinned arena,
  // overridable by a memory resource carried on the call environment.
  constexpr bool __env_has_mr = ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
                             || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  _Tp* h_partials             = nullptr;
  if constexpr (__env_has_mr)
  {
    auto __staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_partials        = static_cast<_Tp*>(__staging_mr.allocate_sync(num_shards * sizeof(_Tp), alignof(_Tp)));
  }
  else
  {
    h_partials = static_cast<_Tp*>(reserved::__pinned_staging(num_shards * sizeof(_Tp)));
  }
  ::std::fill(h_partials, h_partials + num_shards, init_value);

  // Phase 1: local reduce per shard on the shard's environment
  struct __scratch
  {
    void* ptr;
    ::std::size_t bytes;
  };
  ::std::vector<__scratch> d_outputs(num_shards, __scratch{nullptr, 0});

  for (const auto g : each(num_shards))
  {
    const auto& s = data.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    const auto& env                       = envs[g];
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(env);
    stream_scope scope(shard_stream.get());

    auto mr      = ::cuda::mr::get_memory_resource(env);
    _Tp* d_out   = static_cast<_Tp*>(mr.allocate(shard_stream, sizeof(_Tp), alignof(_Tp)));
    d_outputs[g] = __scratch{d_out, sizeof(_Tp)};

    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_out, s.size, reduce_op, init_value, env));
    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Tp), cudaMemcpyDeviceToHost, shard_stream.get()));
  }

  // Phase 2: synchronize and combine in shard order (deterministic)
  barrier(envs);
  _Tp result = init_value;
  for (const auto g : each(num_shards))
  {
    result = reduce_op(result, h_partials[g]);
  }

  // Release scratch (stream-ordered; safe after the syncs above)
  for (const auto g : each(num_shards))
  {
    if (d_outputs[g].ptr != nullptr)
    {
      auto mr = ::cuda::mr::get_memory_resource(envs[g]);
      mr.deallocate(::cuda::get_stream(envs[g]), d_outputs[g].ptr, d_outputs[g].bytes, alignof(_Tp));
    }
  }
  if constexpr (__env_has_mr)
  {
    auto __staging_mr = ::cuda::mr::get_memory_resource(call_env);
    __staging_mr.deallocate_sync(h_partials, num_shards * sizeof(_Tp), alignof(_Tp));
  }
  // (arena staging is cached; nothing to release)

  return result;
}

namespace reserved
{
//! @brief Deterministic cross-shard combine: one thread folds the per-shard
//! partials in shard order and writes the aggregate through @p out exactly
//! once. Shards absent from @p mask (empty shards) contribute @p init — the
//! same fold the synchronous form performs on the host, bit for bit.
//!
//! @p _OutIt is any device-writable output iterator; the write may be a
//! store, or an action (a sink functor, a graph-conditional predicate, ...).
template <typename _Tp, typename _ReduceOp, typename _OutIt>
__global__ void __fold_partials_kernel(
  const _Tp* __partials, ::cuda::std::uint64_t __mask, unsigned __n, _ReduceOp __op, _Tp __init, _OutIt __out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    _Tp __acc = __init;
    for (unsigned __i = 0; __i < __n; ++__i)
    {
      __acc = __op(__acc, ((__mask >> __i) & 1u) ? __partials[__i] : __init);
    }
    *__out = __acc;
  }
}
} // namespace reserved

/**
 * @brief Asynchronous reduce over any `sharded_view`, writing the aggregate
 * through an output iterator: the value-returning form's stream-ordered
 * sibling.
 *
 * Per-shard `cub::DeviceReduce` writes each shard's partial directly into a
 * P-element scratch buffer; a single deterministic fold kernel (fixed shard
 * order, identical to the synchronous form's host fold) then writes the
 * aggregate through @p out on the call environment's stream. This is a
 * combine-bearing TERMINATOR, so unlike the map family its call-stream
 * edges are definitional, not the composition bracket: the entry edge
 * orders the stream-ordered scratch allocation before the shards' writes,
 * and every lane joins the call stream before the fold (the fold consumes
 * all partials). The aggregate is therefore ready in stream order on the
 * OUTPUT's timeline — awaiting the result means awaiting the call stream,
 * while the lanes stay free to run past the call (their next lane-ordered
 * work needs no further edges). Returns after enqueue and performs **no
 * host synchronization** (compatible with `sync_policy::forbid` and with
 * CUDA graph capture; the scratch allocation/free are stream-ordered and
 * enclosed).
 *
 * @param out Device-writable output iterator; written exactly once with the
 *            aggregate. Point it at device memory, pinned host memory (read
 *            after synchronizing the call stream), or a sink.
 *
 * Requirements: the call environment carries the result stream
 * (`cuda::get_stream`); environments are allocating; at most 64 shards
 * (mask-width limit of this implementation).
 *
 * @throws std::invalid_argument on fewer environments than shards or more
 *         than 64 shards.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Tp, class _ReduceOp, class _OutIt, class _CallEnv)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>> _CCCL_AND async_call_env<_CallEnv>)
_CCCL_HOST_API void reduce_into(
  const _S& data, const _Envs& envs, _OutIt out, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env)
{
  const ::std::size_t num_shards = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::reduce_into: fewer environments than shards");
  }
  if (num_shards > 64)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::reduce_into: more than 64 shards not supported");
  }

  const ::cuda::stream_ref call_stream = ::cuda::get_stream(call_env);

  if (num_shards == 0)
  {
    stream_scope scope(call_stream.get());
    reserved::__fold_partials_kernel<<<1, 1, 0, call_stream.get()>>>(
      static_cast<const _Tp*>(nullptr), ::cuda::std::uint64_t{0}, 0u, reduce_op, init_value, out);
    cuda_safe_call(cudaGetLastError());
    return;
  }

  // P-element partials scratch, stream-ordered on the call stream (visible
  // to every shard's stream through unified addressing).
  auto scratch_mr = ::cuda::mr::get_memory_resource(envs[0]);
  _Tp* d_partials = static_cast<_Tp*>(scratch_mr.allocate(call_stream, num_shards * sizeof(_Tp), alignof(_Tp)));
  ::cuda::std::uint64_t mask = 0;

  // Fork + enqueue first, join second: all shards' work is ordered after the
  // caller's timeline but runs CONCURRENTLY across shards; only then does the
  // caller's timeline wait for all of them. (Interleaving join into the fork
  // loop would route each shard's start through the previous shard's
  // completion and serialize the shards.)
  for (const auto g : each(num_shards))
  {
    const auto& s = data.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    mask |= ::cuda::std::uint64_t{1} << g;
    const auto& env                       = envs[g];
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(env);
    // Fork: order the shard's work (and its view of the scratch) after the
    // caller's timeline
    __detail::__wait_stream_on(shard_stream.get(), call_stream.get());
    stream_scope scope(shard_stream.get());
    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_partials + g, s.size, reduce_op, init_value, env));
  }
  for (const auto g : each(num_shards))
  {
    if (((mask >> g) & 1u) != 0)
    {
      // Join: the caller's timeline waits for this shard's partial
      __detail::__wait_stream_on(call_stream.get(), ::cuda::get_stream(envs[g]).get());
    }
  }

  {
    stream_scope scope(call_stream.get());
    reserved::__fold_partials_kernel<<<1, 1, 0, call_stream.get()>>>(
      d_partials, mask, static_cast<unsigned>(num_shards), reduce_op, init_value, out);
    cuda_safe_call(cudaGetLastError());
  }
  scratch_mr.deallocate(call_stream, d_partials, num_shards * sizeof(_Tp), alignof(_Tp));
}

/**
 * @brief Asynchronous reduce over a self-bound sharded structure:
 * environments derived via `default_envs`.
 */
_CCCL_TEMPLATE(class _S, class _Tp, class _ReduceOp, class _OutIt, class _CallEnv)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND async_call_env<_CallEnv>)
_CCCL_HOST_API void
reduce_into(const _S& data, _OutIt out, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env)
{
  const auto envs = default_envs(data);
  sharded::reduce_into(data, envs, out, reduce_op, init_value, call_env);
}

/**
 * @brief Synchronous reduce over a self-bound sharded structure:
 * environments derived via `default_envs`.
 */
_CCCL_TEMPLATE(class _S, class _Tp, class _ReduceOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ReduceOp>>))
[[nodiscard]] _CCCL_HOST_API _Tp
reduce(const _S& data, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::reduce(data, envs, reduce_op, init_value, call_env);
}

// Reduction conveniences over the generic tier -------------------------------

/// @brief Sum of all elements (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API view_element_t<_S> sum(const _S& data, const _Envs& envs, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  return sharded::reduce(data, envs, ::cuda::std::plus<elem_t>{}, elem_t{0}, call_env);
}

/// @brief Sum of all elements (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_CallEnv>>))
[[nodiscard]] _CCCL_HOST_API view_element_t<_S> sum(const _S& data, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::sum(data, envs, call_env);
}

/// @brief Minimum element (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API view_element_t<_S> min(const _S& data, const _Envs& envs, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  return sharded::reduce(data, envs, ::cuda::minimum<elem_t>{}, ::cuda::std::numeric_limits<elem_t>::max(), call_env);
}

/// @brief Minimum element (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_CallEnv>>))
[[nodiscard]] _CCCL_HOST_API view_element_t<_S> min(const _S& data, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::min(data, envs, call_env);
}

/// @brief Maximum element (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API view_element_t<_S> max(const _S& data, const _Envs& envs, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  return sharded::reduce(data, envs, ::cuda::maximum<elem_t>{}, ::cuda::std::numeric_limits<elem_t>::lowest(), call_env);
}

/// @brief Maximum element (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_CallEnv>>))
[[nodiscard]] _CCCL_HOST_API view_element_t<_S> max(const _S& data, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::max(data, envs, call_env);
}
} // namespace cuda::experimental::sharded
