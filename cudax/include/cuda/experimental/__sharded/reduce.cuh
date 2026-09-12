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
 *
 * Init contract (all forms): `result = init (+) fold(all elements)` — the
 * `std::reduce` contract, with the initial value incorporated EXACTLY ONCE.
 * Every shard's `cub::DeviceReduce` runs with `cub::detail::reduce::no_init`
 * (CUB seeds the shard's partial from its first element; an empty shard runs
 * nothing and writes nothing), and the single global fold starts from `init`
 * and applies the operator over the PRESENT partials only, in shard order.
 * An all-empty view yields `init`.
 *
 * Three delivery forms:
 * - `reduce`       — synchronous, returns the value (host fold);
 * - `reduce_into`  — asynchronous, ONE output on the CALL stream: the
 *                    combine-bearing terminator; pick it when the caller
 *                    consumes the scalar on its own stream (a solver loop's
 *                    residual copied to pinned memory, a graph-conditional);
 * - `reduce_into_lanes` — asynchronous, P outputs, one per LANE, each written
 *                    on that lane's own stream (the MGMN "broadcast" output):
 *                    no call stream, no call-stream edges; pick it when the
 *                    scalar is consumed BY THE LANES (a per-shard rescale by
 *                    a global norm, a convergence test feeding lane-ordered
 *                    work) — the pipeline stays lane-ordered end to end.
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

  // Phase 1: local reduce per shard on the shard's environment. Every shard
  // reduces with `no_init` (partial = fold of the shard's own elements); the
  // initial value enters once, in the host fold below.
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

    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_out, s.size, reduce_op, cub::detail::reduce::no_init, env));
    cuda_safe_call(cudaMemcpyAsync(&h_partials[g], d_out, sizeof(_Tp), cudaMemcpyDeviceToHost, shard_stream.get()));
  }

  // Phase 2: synchronize and combine in shard order (deterministic): init
  // first, then every PRESENT partial (empty shards contribute nothing).
  barrier(envs);
  _Tp result = init_value;
  for (const auto g : each(num_shards))
  {
    if (d_outputs[g].ptr != nullptr)
    {
      result = reduce_op(result, h_partials[g]);
    }
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
//! once. The fold starts from @p init and applies the operator over the
//! partials PRESENT in @p mask only (empty shards ran no reduce and wrote no
//! partial; they contribute nothing) — the same fold the synchronous form
//! performs on the host, bit for bit. All-empty writes @p init.
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
      if ((__mask >> __i) & 1u)
      {
        __acc = __op(__acc, __partials[__i]);
      }
    }
    *__out = __acc;
  }
}

//! @brief Maximum shard count of the mask-based folds (64-bit presence mask).
inline constexpr unsigned __max_fold_shards = 64;

//! @brief The per-lane partial slots of `reduce_into_lanes`, passed to the
//! broadcast fold by value (one pointer per shard; absent shards are null).
template <typename _Tp>
struct __partial_slots
{
  const _Tp* __p[__max_fold_shards];
};

//! @brief Broadcast fold for `reduce_into_lanes`: same fold as
//! `__fold_partials_kernel` (init, then the present partials in shard order),
//! reading one slot per shard through @p __slots; launched once PER LANE, on
//! that lane's stream, writing that lane's output.
template <typename _Tp, typename _ReduceOp, typename _OutIt>
__global__ void __fold_partial_slots_kernel(
  __partial_slots<_Tp> __slots, ::cuda::std::uint64_t __mask, unsigned __n, _ReduceOp __op, _Tp __init, _OutIt __out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    _Tp __acc = __init;
    for (unsigned __i = 0; __i < __n; ++__i)
    {
      if ((__mask >> __i) & 1u)
      {
        __acc = __op(__acc, *__slots.__p[__i]);
      }
    }
    *__out = __acc;
  }
}

//! @brief A transient event per lane: recorded once, waited on by any number
//! of streams, destroyed after the waits are enqueued (the driver defers the
//! release until completion; capture-legal — record/wait become graph edges).
//! Plain create/record/destroy per call (~1 us each; no shared event pool
//! exists at this tier — `fork_join_event_pool` is per container).
struct __lane_events
{
  ::std::vector<cudaEvent_t> __ev;

  explicit __lane_events(::std::size_t __n)
      : __ev(__n, nullptr)
  {}
  __lane_events(const __lane_events&)            = delete;
  __lane_events& operator=(const __lane_events&) = delete;
  ~__lane_events()
  {
    for (cudaEvent_t __e : __ev)
    {
      if (__e != nullptr)
      {
        (void) cudaEventDestroy(__e);
      }
    }
  }

  //! Record lane @p __g's event on @p __stream (created under the stream's
  //! device, which `stream_scope` must have made current).
  void __record(::std::size_t __g, cudaStream_t __stream)
  {
    cuda_safe_call(cudaEventCreateWithFlags(&__ev[__g], cudaEventDisableTiming));
    cuda_safe_call(cudaEventRecord(__ev[__g], __stream));
  }

  //! Make @p __stream wait for lane @p __g's event (no-op when never recorded).
  void __wait(::std::size_t __g, cudaStream_t __stream) const
  {
    if (__ev[__g] != nullptr)
    {
      cuda_safe_call(cudaStreamWaitEvent(__stream, __ev[__g], 0));
    }
  }
};
} // namespace reserved

/**
 * @brief Asynchronous reduce over any `sharded_view`, writing the aggregate
 * through an output iterator: the value-returning form's stream-ordered
 * sibling.
 *
 * Per-shard `cub::DeviceReduce` (with `no_init`) writes each shard's partial
 * directly into a P-element scratch buffer; a single deterministic fold
 * kernel (`init`, then the present partials in fixed shard order — identical
 * to the synchronous form's host fold) then writes the aggregate through
 * @p out on the call environment's stream. This is a
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
  if (num_shards > reserved::__max_fold_shards)
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
    // `no_init`: the partial is the fold of the shard's own elements; the
    // initial value enters exactly once, in the fold kernel.
    cuda_safe_call(
      cub::DeviceReduce::Reduce(s.data, d_partials + g, s.size, reduce_op, cub::detail::reduce::no_init, env));
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
 * @brief Asynchronous LANE-RESIDENT reduce over any `sharded_view`: the
 * aggregate is delivered P times, once per lane, each copy written on ITS
 * OWN lane's stream — the multi-GPU "broadcast" output shape. No call
 * stream, no call-stream edges, no host synchronization.
 *
 * After the call, `outs[g]` holds the full aggregate in stream order on lane
 * g's timeline: lane-ordered work enqueued next on `envs[g]` (a rescale of
 * shard g by a global norm, a per-lane convergence test) consumes it with
 * no further edges, and the lanes never join a foreign stream. Prefer
 * `reduce_into` when the CALLER needs the scalar on its own stream.
 *
 * Design (P lanes, P at most 64):
 * - per lane g, on `envs[g]`'s stream and from `envs[g]`'s memory
 *   resource: a 1-element partial slot is allocated and the shard's
 *   `cub::DeviceReduce` (with `no_init`) writes it; event E_g is recorded.
 *   Per-lane slots — rather than one P-slot scratch on lane 0 — keep the
 *   heavy phase INDEPENDENT across lanes: lane g's reduce starts as soon as
 *   lane g is ready, never behind lane 0's timeline (the lanes only meet at
 *   the fold, where they must). Empty shards allocate nothing and record no
 *   E_g (they contribute nothing to the fold).
 * - lane g waits on E_h for every other present lane h (P(P-1) waits), then
 *   launches the broadcast fold on its own stream: `init`, then the present
 *   partials in shard order (the P slot pointers travel by value), writing
 *   `outs[g]`; event F_g is recorded after the fold.
 * - lifetime: slot g is read by every lane's fold, so lane g waits on F_h
 *   for every other lane h (P(P-1) waits) before its stream-ordered
 *   deallocate. Edge count per call: P E-records + P F-records +
 *   2 P (P-1) waits; events are transient (created/destroyed per call, no
 *   pool at this tier).
 *
 * CUDA graph capture: legal in the same way as every lane-ordered call —
 * the cross-lane event waits require all lanes to be capturing into the
 * SAME graph, i.e. forked from the capture origin beforehand
 * (`sharded_array::fork_from(origin)` or entry edges of the caller's own)
 * and joined back before `cudaStreamEndCapture`; the slot allocation/free
 * are stream-ordered and enclosed. A mix of capturing and non-capturing
 * lanes is a CUDA error at the first cross-lane wait.
 *
 * @param outs Random-access iterator over P device-writable output
 *             positions (`outs[g]` written exactly once by lane g). Device
 *             memory, or pinned host memory read after synchronizing the
 *             lane of interest.
 *
 * Requirements: allocating environments (`sharded_alloc_env_range`), one
 * per shard; at most 64 shards (mask-width limit).
 *
 * @throws std::invalid_argument on fewer environments than shards or more
 *         than 64 shards.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Tp, class _ReduceOp, class _OutIt)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void
reduce_into_lanes(const _S& data, const _Envs& envs, _OutIt outs, _ReduceOp reduce_op, _Tp init_value)
{
  const ::std::size_t num_shards = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::reduce_into_lanes: fewer environments than shards");
  }
  if (num_shards > reserved::__max_fold_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::reduce_into_lanes: more than 64 shards not supported");
  }
  if (num_shards == 0)
  {
    return; // no lanes, no outputs
  }

  reserved::__partial_slots<_Tp> slots{};
  ::cuda::std::uint64_t mask = 0;
  reserved::__lane_events reduced(num_shards); // E_g: lane g's partial is written
  reserved::__lane_events folded(num_shards); // F_g: lane g's fold has read every slot

  // Phase 1 (independent across lanes): slot + per-shard reduce, on the lane
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
    stream_scope scope(shard_stream.get());
    auto mr      = ::cuda::mr::get_memory_resource(env);
    _Tp* slot    = static_cast<_Tp*>(mr.allocate(shard_stream, sizeof(_Tp), alignof(_Tp)));
    slots.__p[g] = slot;
    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, slot, s.size, reduce_op, cub::detail::reduce::no_init, env));
    reduced.__record(g, shard_stream.get());
  }

  // Phase 2 (per lane): wait for every other present partial, fold on the
  // lane, publish F_g. All-empty: every lane writes init with no waits.
  for (const auto g : each(num_shards))
  {
    const cudaStream_t lane_stream = ::cuda::get_stream(envs[g]).get();
    stream_scope scope(lane_stream);
    for (const auto h : each(num_shards))
    {
      if (h != g)
      {
        reduced.__wait(h, lane_stream);
      }
    }
    reserved::__fold_partial_slots_kernel<<<1, 1, 0, lane_stream>>>(
      slots, mask, static_cast<unsigned>(num_shards), reduce_op, init_value, outs + g);
    cuda_safe_call(cudaGetLastError());
    folded.__record(g, lane_stream);
  }

  // Phase 3 (per present lane): release slot g once every lane's fold has
  // read it (stream-ordered on lane g, after the F_h edges)
  for (const auto g : each(num_shards))
  {
    if (((mask >> g) & 1u) == 0)
    {
      continue;
    }
    const auto& env                       = envs[g];
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(env);
    stream_scope scope(shard_stream.get());
    for (const auto h : each(num_shards))
    {
      if (h != g)
      {
        folded.__wait(h, shard_stream.get());
      }
    }
    auto mr = ::cuda::mr::get_memory_resource(env);
    mr.deallocate(shard_stream, const_cast<_Tp*>(slots.__p[g]), sizeof(_Tp), alignof(_Tp));
  }
  // `reduced` / `folded` destroy their events here; the enqueued waits keep
  // the driver-side references alive until they complete.
}

/**
 * @brief Lane-resident reduce over a self-bound sharded structure:
 * environments derived via `default_envs`.
 */
_CCCL_TEMPLATE(class _S, class _Tp, class _ReduceOp, class _OutIt)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>>)
_CCCL_HOST_API void reduce_into_lanes(const _S& data, _OutIt outs, _ReduceOp reduce_op, _Tp init_value)
{
  const auto envs = default_envs(data);
  sharded::reduce_into_lanes(data, envs, outs, reduce_op, init_value);
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
