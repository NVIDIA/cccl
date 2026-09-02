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
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Reduce all elements with a custom operator.
 *
 * Phase 1 runs CUB `DeviceReduce` per shard on the shard's stream, with
 * temporaries allocated from the shard's place; phase 2 combines the
 * per-place partials. SYNCHRONOUS: returns the final value.
 *
 * @param group   the place group providing per-place memory resources
 * @param data    the sharded input (not modified)
 * @param reduce_op host- and device-callable binary operator
 * @param init_value initial (identity) value
 */
template <typename _Tp, typename _ReduceOp>
[[nodiscard]] _CCCL_HOST_API _Tp
reduce(place_group& group, const sharded_array<_Tp>& data, _ReduceOp reduce_op, _Tp init_value = _Tp{})
{
  if (data.empty())
  {
    return init_value;
  }

  // Host-side combine + synchronization: cannot be recorded into a CUDA graph
  reserved::check_not_capturing(data, "sharded::reduce");

  const size_t num_shards = data.num_shards();

  // Pinned host memory for the per-place partials (initialized so skipped
  // empty shards contribute the identity)
  places::place_memory_resource host_mr(data_place::host());
  _Tp* h_partials = static_cast<_Tp*>(host_mr.allocate_sync(num_shards * sizeof(_Tp), alignof(_Tp)));
  ::std::fill(h_partials, h_partials + num_shards, init_value);

  // Phase 1: local reduce on each shard; free the per-shard outputs only
  // after the final sync (places without stream-ordered deallocation)
  ::std::vector<::std::pair<places::place_memory_resource, _Tp*>> d_outputs;
  d_outputs.reserve(num_shards);

  data.each_shard->*[&](const size_t g, const auto& s) {
    places::place_memory_resource mr(s.place);
    _Tp* d_out = static_cast<_Tp*>(mr.allocate(::cuda::stream_ref{s.stream}, sizeof(_Tp), alignof(_Tp)));
    d_outputs.emplace_back(mr, d_out);

    // Temporaries come from the shard's place through the group's resources
    const auto env = group.env(s.place, s.stream);
    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_out, s.size, reduce_op, init_value, env));

    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  // Phase 2: combine the per-place partials
  _Tp result = init_value;
  for (size_t g = 0; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  for (auto& [mr, ptr] : d_outputs)
  {
    mr.deallocate_sync(ptr, sizeof(_Tp), alignof(_Tp));
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Tp), alignof(_Tp));

  return result;
}

/// @brief Sum of all elements.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp sum(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::std::plus<_Tp>{}, _Tp{0});
}

/// @brief Minimum element.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp min(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::minimum<_Tp>{}, ::cuda::std::numeric_limits<_Tp>::max());
}

/// @brief Maximum element.
template <typename _Tp>
[[nodiscard]] _CCCL_HOST_API _Tp max(place_group& group, const sharded_array<_Tp>& data)
{
  return reduce(group, data, ::cuda::maximum<_Tp>{}, ::cuda::std::numeric_limits<_Tp>::lowest());
}

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
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API _Tp
reduce(const _S& data, const _Envs& envs, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env = {})
{
  const ::std::size_t num_shards = static_cast<::std::size_t>(data.num_shards());
  if (static_cast<::std::size_t>(envs.size()) < num_shards)
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
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), "sharded::reduce");
  }

  // Pinned host staging for the per-shard partials (host-accessible +
  // async-transfer-capable). Initialized to the identity so empty shards
  // contribute nothing.
  places::place_memory_resource host_mr(data_place::host());
  _Tp* h_partials = static_cast<_Tp*>(host_mr.allocate_sync(num_shards * sizeof(_Tp), alignof(_Tp)));
  ::std::fill(h_partials, h_partials + num_shards, init_value);

  // Phase 1: local reduce per shard on the shard's environment
  struct __scratch
  {
    void* ptr;
    ::std::size_t bytes;
  };
  ::std::vector<__scratch> d_outputs(num_shards, __scratch{nullptr, 0});

  for (::std::size_t g = 0; g < num_shards; g++)
  {
    const auto& s = data.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    const auto& env                       = envs[g];
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(env);
    stream_scope scope(shard_stream.get());

    auto mr    = ::cuda::mr::get_memory_resource(env);
    _Tp* d_out = static_cast<_Tp*>(mr.allocate(shard_stream, sizeof(_Tp), alignof(_Tp)));
    d_outputs[g] = __scratch{d_out, sizeof(_Tp)};

    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_out, s.size, reduce_op, init_value, env));
    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Tp), cudaMemcpyDeviceToHost, shard_stream.get()));
  }

  // Phase 2: synchronize and combine in shard order (deterministic)
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (data.shard(g).size != 0)
    {
      cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(envs[g]).get()));
    }
  }
  _Tp result = init_value;
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    result = reduce_op(result, h_partials[g]);
  }

  // Release scratch (stream-ordered; safe after the syncs above)
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (d_outputs[g].ptr != nullptr)
    {
      auto mr = ::cuda::mr::get_memory_resource(envs[g]);
      mr.deallocate(::cuda::get_stream(envs[g]), d_outputs[g].ptr, d_outputs[g].bytes, alignof(_Tp));
    }
  }
  host_mr.deallocate_sync(h_partials, num_shards * sizeof(_Tp), alignof(_Tp));

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
 * aggregate through @p out on the call environment's stream. The whole call
 * is ordered against that stream — forked on entry, joined before the fold —
 * returns after enqueue, and performs **no host synchronization**
 * (compatible with `sync_policy::forbid` and with CUDA graph capture; the
 * scratch allocation/free are stream-ordered and enclosed).
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
_CCCL_HOST_API void
reduce_into(const _S& data, const _Envs& envs, _OutIt out, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env)
{
  const ::std::size_t num_shards = static_cast<::std::size_t>(data.num_shards());
  if (static_cast<::std::size_t>(envs.size()) < num_shards)
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
  auto scratch_mr  = ::cuda::mr::get_memory_resource(envs[0]);
  _Tp* d_partials  = static_cast<_Tp*>(scratch_mr.allocate(call_stream, num_shards * sizeof(_Tp), alignof(_Tp)));
  ::cuda::std::uint64_t mask = 0;

  for (::std::size_t g = 0; g < num_shards; g++)
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
    // Join: the caller's timeline waits for this shard's partial
    __detail::__wait_stream_on(call_stream.get(), shard_stream.get());
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
_CCCL_HOST_API void reduce_into(const _S& data, _OutIt out, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env)
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
[[nodiscard]] _CCCL_HOST_API _Tp reduce(const _S& data, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return sharded::reduce(data, envs, reduce_op, init_value, call_env);
}

} // namespace cuda::experimental::sharded
