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
 * @brief Reduction over sharded views.
 *
 * The engine is the MGMN reduce (`cuda::experimental::mgmn::reduce`: one
 * `cub::DeviceReduce` per rank on the rank's environment, then an
 * `all_reduce` of the P partials folded in rank order), instantiated over
 * the in-process `places_communicator` of `engine/mgmn.cuh`, one rank per
 * shard. Algorithm temporaries are drawn from each shard's environment
 * resource, so scratch lands where the work runs. The sharded verbs keep
 * their signatures and their contract; the engine is not visible to the
 * caller.
 *
 * Init contract (all forms): `result = init (+) fold(all elements)` — the
 * `std::reduce` contract, with the initial value incorporated EXACTLY ONCE:
 * shard 0's partial is seeded with `init`, every other shard's with the
 * operator's identity, and the cross-shard fold runs in shard order. An
 * all-empty view yields `init`. Operators `cuda::identity_element` knows
 * (`plus`, `multiplies`, `minimum`, `maximum`, the bit and logical
 * operators) run the engine directly on the element type; any other operator
 * runs it on a lifted `{value, present}` pair whose "absent" is the identity
 * — the fold then applies the caller's operator to present values only, in
 * the same order, so no identity is ever required of the caller.
 *
 * Three delivery forms:
 * - `reduce`       — synchronous, returns the value (refuses under capture
 *                    and `sync_policy::forbid`);
 * - `reduce_into`  — asynchronous, ONE output on the CALL stream: the
 *                    combine-bearing terminator; pick it when the caller
 *                    consumes the scalar on its own stream (a solver loop's
 *                    residual copied to pinned memory, a graph-conditional);
 * - `reduce_into_lanes` — asynchronous, P outputs, one per LANE, each written
 *                    on that lane's own stream (the MGMN "broadcasted"
 *                    output): no call stream, no call-stream edges; pick it
 *                    when the scalar is consumed BY THE LANES (a per-shard
 *                    rescale by a global norm, a convergence test feeding
 *                    lane-ordered work) — the pipeline stays lane-ordered end
 *                    to end.
 * The asynchronous forms perform no host synchronization and capture into
 * CUDA graphs.
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

#include <cuda/__functional/operator_properties.h> // identity_element
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/composition/verbs.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/concepts/guards.cuh>
#include <cuda/experimental/__sharded/container/default_envs.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/engine/mgmn.cuh>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
//! @brief Does `cuda::identity_element` know the identity of `_Op` over `_Tp`?
template <class _Op, class _Tp>
inline constexpr bool __has_identity_element_v =
  !::cuda::std::is_same_v<::cuda::std::remove_cvref_t<decltype(::cuda::identity_element<_Op, _Tp>())>,
                          ::cuda::__no_identity_element>;

//! @brief A value lifted with a presence flag: the engine's element type
//! for operators without a known identity. `{_, false}` is the identity of
//! `__lifted_op`.
template <class _Tp>
struct __lifted
{
  _Tp __value;
  bool __present;
};

//! @brief The caller's operator lifted over `__lifted<_Tp>`: absent operands
//! are skipped, present ones folded in the given order. Associative whenever
//! the caller's operator is.
template <class _Op, class _Tp>
struct __lifted_op
{
  mutable _Op __op;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE_API __lifted<_Tp> operator()(const __lifted<_Tp>& __a, const __lifted<_Tp>& __b) const
  {
    if (!__a.__present)
    {
      return __b;
    }
    if (!__b.__present)
    {
      return __a;
    }
    return __lifted<_Tp>{static_cast<_Tp>(__op(__a.__value, __b.__value)), true};
  }
};

//! @brief Input adaptor of the lifted path: every element is present.
template <class _Tp>
struct __lift_fn
{
  template <class _Up>
  _CCCL_HOST_DEVICE_API __lifted<_Tp> operator()(const _Up& __v) const
  {
    return __lifted<_Tp>{static_cast<_Tp>(__v), true};
  }
};

template <class _Tp>
_CCCL_HOST_DEVICE_API const _Tp& __unlift(const _Tp& __v) noexcept
{
  return __v;
}

template <class _Tp>
_CCCL_HOST_DEVICE_API const _Tp& __unlift(const __lifted<_Tp>& __v) noexcept
{
  return __v.__value;
}

//! @brief One thread writes `*__out = value(*__slot)`: the delivery of an
//! engine result through an arbitrary device-writable output iterator (a
//! store, a sink functor, a graph-conditional predicate, ...).
template <class _Stored, class _OutIt>
__global__ void __mgmn_store_kernel(const _Stored* __slot, _OutIt __out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    *__out = __unlift(*__slot);
  }
}

//! @brief One thread writes `*__out = __value` (the all-empty result).
template <class _Tp, class _OutIt>
__global__ void __mgmn_store_value_kernel(_Tp __value, _OutIt __out)
{
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    *__out = __value;
  }
}

//! @brief The engine's view of a reduction of `_Tp` under `_Op`: the stored
//! (accumulator) type, the operator and the identity it runs with, and the
//! input adaptor — the element type itself when the identity is known, the
//! lifted pair otherwise.
template <class _Op, class _Tp>
struct __reduce_engine
{
  static constexpr bool __direct = __has_identity_element_v<_Op, _Tp>;
  using __stored_t               = ::cuda::std::conditional_t<__direct, _Tp, __lifted<_Tp>>;
  using __op_t                   = ::cuda::std::conditional_t<__direct, _Op, __lifted_op<_Op, _Tp>>;

  [[nodiscard]] static __op_t __lift_op(_Op __op)
  {
    if constexpr (__direct)
    {
      return __op;
    }
    else
    {
      return __lifted_op<_Op, _Tp>{__op};
    }
  }

  [[nodiscard]] static __stored_t __lift_init(const _Tp& __init)
  {
    if constexpr (__direct)
    {
      return __init;
    }
    else
    {
      return __lifted<_Tp>{__init, true};
    }
  }

  [[nodiscard]] static __stored_t __identity()
  {
    if constexpr (__direct)
    {
      return static_cast<_Tp>(::cuda::identity_element<_Op, _Tp>());
    }
    else
    {
      return __lifted<_Tp>{_Tp{}, false};
    }
  }

  template <class _Elem>
  [[nodiscard]] static auto __input(const _Elem* __data)
  {
    if constexpr (__direct)
    {
      return __data;
    }
    else
    {
      return ::cuda::make_transform_iterator(__data, __lift_fn<_Tp>{});
    }
  }
};

//! @brief The engine call shared by the three forms: the broadcasted MGMN
//! reduce of @p __data over every shard, writing `init (+) fold` to
//! `__outputs[g]` (a `__stored_t*`) on lane g's stream. Driven through
//! `__mgmn_drive` under @p __drive_env (the contract: `default_call_env`
//! for the synchronous form, `__lane_ordered_t` for the forms that own
//! their edges); @p __call_env is the caller's environment, read for its
//! requirements only.
template <class _S, class _Envs, class _DriveEnv, class _CallEnv, class _Stored, class _ReduceOp, class _Tp>
_CCCL_HOST_API void __mgmn_reduce_into_slots(
  const _S& __data,
  const _Envs& __envs,
  const _DriveEnv& __drive_env,
  const _CallEnv& __call_env,
  const char* __what,
  const ::std::vector<_Stored*>& __outputs,
  _ReduceOp __op,
  const _Tp& __init)
{
  using __engine = __reduce_engine<_ReduceOp, _Tp>;
  using __elem_t = view_element_t<_S>;
  static_assert(::cuda::std::is_same_v<_Stored, typename __engine::__stored_t>);

  // Reductions honor `run_to_run` for every operator (CUB's default).
  const auto __reqs = __mgmn_requirements<true>(__call_env);
  __mgmn_drive<true>(
    __data,
    __envs,
    __drive_env,
    __what,
    [&](const auto& __env) {
      return __mgmn_alloc_env(__env, __reqs);
    },
    [&](const auto& __comms, const auto& __menvs, const auto& __lanes) {
      const auto __inputs = __mgmn_per_lane(__lanes, [&](::std::size_t __g) {
        return __engine::__input(static_cast<const __elem_t*>(__data.shard(__g).data));
      });
      ::cuda::experimental::mgmn::reduce(
        ::cuda::experimental::broadcasted,
        __comms,
        __menvs,
        __inputs,
        __mgmn_sizes(__data, __lanes),
        __outputs,
        __engine::__lift_init(__init),
        __engine::__lift_op(__op),
        __engine::__identity());
    });
}

//! @brief One `_Stored` scratch slot per lane, from the lane's resource on
//! the lane's stream (stream-ordered; released the same way).
template <class _Stored, class _Envs>
class __lane_slots
{
public:
  __lane_slots(const _Envs& __envs, ::std::size_t __n)
      : __envs_(__envs)
      , __slots_(__n, nullptr)
  {
    for (const auto __g : each(__n))
    {
      auto __mr = ::cuda::mr::get_memory_resource(__envs_[__g]);
      __slots_[__g] =
        static_cast<_Stored*>(__mr.allocate(::cuda::get_stream(__envs_[__g]), sizeof(_Stored), alignof(_Stored)));
    }
  }

  __lane_slots(const __lane_slots&)            = delete;
  __lane_slots& operator=(const __lane_slots&) = delete;

  ~__lane_slots()
  {
    for (const auto __g : each(__slots_.size()))
    {
      auto __mr = ::cuda::mr::get_memory_resource(__envs_[__g]);
      __mr.deallocate(::cuda::get_stream(__envs_[__g]), __slots_[__g], sizeof(_Stored), alignof(_Stored));
    }
  }

  [[nodiscard]] const ::std::vector<_Stored*>& __pointers() const noexcept
  {
    return __slots_;
  }

private:
  const _Envs& __envs_;
  ::std::vector<_Stored*> __slots_;
};
} // namespace reserved

// ============================================================================
// Concept-generic tier: any sharded_view + allocating environments
// ============================================================================

/**
 * @brief Synchronous reduce over any `sharded_view`: the MGMN reduce over
 * the shards (per-shard `cub::DeviceReduce` on the shard's environment, the
 * P partials folded in shard order on device), lane 0's copy of the result
 * returned.
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
 * @throws std::invalid_argument when the environment count does not match
 *         the shard count, or on more than 64 shards.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Tp, class _ReduceOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API _Tp
reduce(const _S& data, const _Envs& envs, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env = {})
{
  using __stored_t               = typename reserved::__reduce_engine<_ReduceOp, _Tp>::__stored_t;
  const ::std::size_t num_shards = reserved::__shard_count(data);
  reserved::__check_env_count(envs, num_shards, "sharded::reduce");
  if (num_shards == 0)
  {
    return init_value;
  }

  // Refusals first, before any CUDA call: this form synchronizes. (The call
  // environment is read for its policy and requirements only; a stream it
  // may carry does not select the asynchronous contract of this form.)
  require_sync_allowed(call_env, "sharded::reduce (synchronous form)");
  reserved::__check_envs_not_capturing(envs, num_shards, "sharded::reduce");

  // The synchronous form of the driver: every lane synchronized before it
  // returns, lane 0's slot then holds the result.
  reserved::__lane_slots<__stored_t, _Envs> slots(envs, num_shards);
  reserved::__mgmn_reduce_into_slots(
    data, envs, default_call_env{}, call_env, "sharded::reduce", slots.__pointers(), reduce_op, init_value);

  __stored_t result{};
  cuda_safe_call(cudaMemcpy(&result, slots.__pointers()[0], sizeof(__stored_t), cudaMemcpyDefault));
  return reserved::__unlift(result);
}

/**
 * @brief Asynchronous reduce over any `sharded_view`, writing the aggregate
 * through an output iterator: the value-returning form's stream-ordered
 * sibling.
 *
 * The MGMN reduce runs over the shards into a P-slot scratch (stream-ordered
 * from `envs[0]`'s resource on the call stream); the aggregate is then
 * written through @p out on the call environment's stream. This is a
 * combine-bearing TERMINATOR, so unlike the map family its call-stream
 * edges are definitional, not the composition bracket: the entry edge
 * orders the stream-ordered scratch allocation before the shards' work, and
 * every lane joins the call stream before the delivery. The aggregate is
 * therefore ready in stream order on the OUTPUT's timeline — awaiting the
 * result means awaiting the call stream, while the lanes stay free to run
 * past the call (their next lane-ordered work needs no further edges).
 * Returns after enqueue and performs **no host synchronization** (compatible
 * with `sync_policy::forbid` and with CUDA graph capture; the scratch
 * allocation/free are stream-ordered and enclosed).
 *
 * @param out Device-writable output iterator; written exactly once with the
 *            aggregate. Point it at device memory, pinned host memory (read
 *            after synchronizing the call stream), or a sink.
 *
 * Requirements: the call environment carries the result stream
 * (`cuda::get_stream`); environments are allocating; at most 64 shards.
 *
 * @throws std::invalid_argument when the environment count does not match
 *         the shard count, or on more than 64 shards.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Tp, class _ReduceOp, class _OutIt, class _CallEnv)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>> _CCCL_AND async_call_env<_CallEnv>)
_CCCL_HOST_API void reduce_into(
  const _S& data, const _Envs& envs, _OutIt out, _ReduceOp reduce_op, _Tp init_value, const _CallEnv& call_env)
{
  using __stored_t               = typename reserved::__reduce_engine<_ReduceOp, _Tp>::__stored_t;
  const ::std::size_t num_shards = reserved::__shard_count(data);
  reserved::__check_env_count(envs, num_shards, "sharded::reduce_into");
  if (num_shards > reserved::__max_fold_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::reduce_into: more than 64 shards not supported");
  }

  const ::cuda::stream_ref call_stream = ::cuda::get_stream(call_env);

  if (num_shards == 0)
  {
    stream_scope scope(call_stream.get());
    reserved::__mgmn_store_value_kernel<<<1, 1, 0, call_stream.get()>>>(init_value, out);
    cuda_safe_call(cudaGetLastError());
    return;
  }

  // P-slot scratch, stream-ordered on the call stream (visible to every
  // shard's stream through unified addressing); the fork below orders the
  // lanes' writes after its allocation.
  auto scratch_mr = ::cuda::mr::get_memory_resource(envs[0]);
  __stored_t* d_slots =
    static_cast<__stored_t*>(scratch_mr.allocate(call_stream, num_shards * sizeof(__stored_t), alignof(__stored_t)));
  ::std::vector<__stored_t*> outputs;
  outputs.reserve(num_shards);
  for (const auto g : each(num_shards))
  {
    outputs.push_back(d_slots + g);
    // Fork: order the lane's work (and its view of the scratch) after the
    // caller's timeline
    __detail::__wait_stream_on(::cuda::get_stream(envs[g]).get(), call_stream.get());
  }

  reserved::__mgmn_reduce_into_slots(
    data, envs, reserved::__lane_ordered_t{}, call_env, "sharded::reduce_into", outputs, reduce_op, init_value);

  for (const auto g : each(num_shards))
  {
    // Join: the caller's timeline waits for every lane (slot 0 is complete
    // once lane 0 is; the join of the other lanes releases the scratch they
    // read, in the same stream order)
    __detail::__wait_stream_on(call_stream.get(), ::cuda::get_stream(envs[g]).get());
  }

  {
    stream_scope scope(call_stream.get());
    reserved::__mgmn_store_kernel<<<1, 1, 0, call_stream.get()>>>(static_cast<const __stored_t*>(d_slots), out);
    cuda_safe_call(cudaGetLastError());
  }
  scratch_mr.deallocate(call_stream, d_slots, num_shards * sizeof(__stored_t), alignof(__stored_t));
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
 * This is the broadcasted MGMN reduce itself: per lane, on `envs[g]`'s
 * stream and from `envs[g]`'s resource, the shard's `cub::DeviceReduce`
 * writes its partial (the identity for an empty shard); the communicator's
 * `all_reduce` then folds the P partials in shard order on every lane, into
 * `outs[g]` directly when @p outs is a pointer to `_Tp` (else through a
 * per-lane scratch slot and one delivery kernel). The lanes only meet at the
 * fold, where they must (the communicator's event edges).
 *
 * CUDA graph capture: legal in the same way as every lane-ordered call —
 * the cross-lane event waits require all lanes to be capturing into the
 * SAME graph, i.e. forked from the capture origin beforehand
 * (`sharded_array::fork_from(origin)` or entry edges of the caller's own)
 * and joined back before `cudaStreamEndCapture`; the temporaries are
 * stream-ordered and enclosed. A mix of capturing and non-capturing lanes is
 * a CUDA error at the first cross-lane wait.
 *
 * @param outs Random-access iterator over P device-writable output
 *             positions (`outs[g]` written exactly once by lane g). Device
 *             memory, or pinned host memory read after synchronizing the
 *             lane of interest.
 *
 * Requirements: allocating environments (`sharded_alloc_env_range`), one
 * per shard; at most 64 shards.
 *
 * @throws std::invalid_argument when the environment count does not match
 *         the shard count, or on more than 64 shards.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Tp, class _ReduceOp, class _OutIt)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void
reduce_into_lanes(const _S& data, const _Envs& envs, _OutIt outs, _ReduceOp reduce_op, _Tp init_value)
{
  using __engine                 = reserved::__reduce_engine<_ReduceOp, _Tp>;
  using __stored_t               = typename __engine::__stored_t;
  const ::std::size_t num_shards = reserved::__shard_count(data);
  reserved::__check_env_count(envs, num_shards, "sharded::reduce_into_lanes");
  if (num_shards > reserved::__max_fold_shards)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::reduce_into_lanes: more than 64 shards not supported");
  }
  if (num_shards == 0)
  {
    return; // no lanes, no outputs
  }

  constexpr bool __direct_outputs =
    __engine::__direct && ::cuda::std::contiguous_iterator<_OutIt>
    && ::cuda::std::is_same_v<::cuda::std::remove_cv_t<::cuda::std::iter_value_t<_OutIt>>, _Tp>;

  if constexpr (__direct_outputs)
  {
    ::std::vector<__stored_t*> outputs;
    outputs.reserve(num_shards);
    for (const auto g : each(num_shards))
    {
      outputs.push_back(::cuda::std::to_address(outs + static_cast<::cuda::std::iter_difference_t<_OutIt>>(g)));
    }
    reserved::__mgmn_reduce_into_slots(
      data,
      envs,
      reserved::__lane_ordered_t{},
      default_call_env{},
      "sharded::reduce_into_lanes",
      outputs,
      reduce_op,
      init_value);
  }
  else
  {
    // Per-lane scratch slot, then one delivery kernel per lane on the lane's
    // stream (a generic output iterator, or the lifted engine type).
    reserved::__lane_slots<__stored_t, _Envs> slots(envs, num_shards);
    reserved::__mgmn_reduce_into_slots(
      data,
      envs,
      reserved::__lane_ordered_t{},
      default_call_env{},
      "sharded::reduce_into_lanes",
      slots.__pointers(),
      reduce_op,
      init_value);
    for (const auto g : each(num_shards))
    {
      const cudaStream_t lane_stream = ::cuda::get_stream(envs[g]).get();
      stream_scope scope(lane_stream);
      reserved::__mgmn_store_kernel<<<1, 1, 0, lane_stream>>>(
        static_cast<const __stored_t*>(slots.__pointers()[g]),
        outs + static_cast<::cuda::std::iter_difference_t<_OutIt>>(g));
      cuda_safe_call(cudaGetLastError());
    }
    // `slots` releases lane g's slot on lane g's stream, after its delivery.
  }
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
