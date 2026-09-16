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
 * @brief The MGMN bridge: run the cross-shard multi-GPU multi-node
 *        algorithms of `cuda/experimental/__multi_gpu/algorithm/` directly
 *        on sharded arrays, over the shared address space.
 *
 * The communicator (`places_communicator`) and the adapter
 * (`make_communicators`, `mgmn_envs`, the `reserved` lockstep-range helpers)
 * live in `mgmn_adapter.cuh`; this header adds the `mgmn::` verbs
 * (`inclusive_scan`, `exclusive_scan`, `reduce_into_lanes`, `reduce`) that
 * wrap the cross-shard MGMN algorithms behind the sharded vocabulary. The
 * sharded map-family verbs whose engine is a rank-local MGMN algorithm
 * (`sharded::transform`, `sharded::zip_transform`) are in `transform.cuh`,
 * on the same adapter.
 *
 * Contract of the `mgmn::` verbs: asynchronous and lane-ordered (the
 * composition contract of `composition.cuh`) — each shard's work is enqueued
 * on its environment's stream, cross-shard steps are event edges between
 * those streams, results are ready in stream order, no host synchronization
 * ever happens (so they capture into CUDA graphs). A call environment
 * carrying `composition::bracketed` seals the call against the call stream.
 *
 * The existing `sharded::` scans and reductions are untouched: they remain
 * the reference the MGMN path is checked against.
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
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__multi_gpu/algorithm/reduce/reduce.h>
#include <cuda/experimental/__multi_gpu/algorithm/scan/scan.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/stream_pool.cuh>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/mgmn_adapter.cuh>
#include <cuda/experimental/__sharded/reduce.cuh> // __partial_slots, __max_fold_shards
#include <cuda/experimental/__sharded/stream_scope.cuh>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <exception>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
//! @brief Shared driver of the MGMN scans over sharded views.
template <bool _Inclusive, class _SIn, class _SOut, class _Envs, class _ScanOp, class _Tp, class _CallEnv>
_CCCL_HOST_API void __mgmn_scan(
  const _SIn& __in,
  _SOut&& __out,
  const _Envs& __envs,
  _ScanOp __op,
  _Tp __init,
  _Tp __identity,
  const _CallEnv& __call_env,
  const char* __what)
{
  const ::std::size_t __n = __shard_count(__in);
  if (__env_count(__envs) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": fewer environments than shards");
  }
  __check_copartitioned(__in, __out, __what);
  if (__n == 0)
  {
    return;
  }
  const __mgmn_bracket<_Envs, _CallEnv> __bracket(__envs, __n, __call_env);

  const auto __comms   = sharded::make_communicators(__envs, __n);
  const auto __menvs   = sharded::mgmn_envs(__envs, __n);
  const auto __inputs  = __mgmn_pointers<const _Tp*>(__in);
  const auto __sizes   = __mgmn_sizes(__in);
  const auto __outputs = __mgmn_pointers<_Tp*>(__out);

  if constexpr (_Inclusive)
  {
    ::cuda::experimental::mgmn::inclusive_scan(
      ::cuda::experimental::distributed, __comms, __menvs, __inputs, __sizes, __outputs, __init, __op, __identity);
  }
  else
  {
    ::cuda::experimental::mgmn::exclusive_scan(
      ::cuda::experimental::distributed, __comms, __menvs, __inputs, __sizes, __outputs, __init, __op, __identity);
  }
}
} // namespace reserved

// ============================================================================
// The mgmn:: verbs: MGMN algorithms behind the sharded vocabulary
// ============================================================================

namespace mgmn
{
/**
 * @brief Inclusive scan `out[i] = fold(in[0..i])` over the global index
 * space, through the MGMN scan (reduce, all-gather of the P partials, device
 * prefix, seeded per-shard `cub::DeviceScan`). Asynchronous, lane-ordered,
 * capturable. @p out must be co-partitioned with @p in; it may be @p in.
 *
 * @p identity is the operator's identity element, defaulted where
 * `cuda::identity_element` knows the operator; custom operators supply it.
 */
_CCCL_TEMPLATE(class _SIn, class _SOut, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND
    sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_scan(
  const _SIn& in,
  _SOut&& out,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  reserved::__mgmn_scan<true>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/// @brief In-place inclusive scan (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__mgmn_scan<true>(data, data, envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/// @brief Inclusive scan into @p out (self-bound: environments of @p in).
_CCCL_TEMPLATE(class _SIn, class _SOut, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>
                 _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void inclusive_scan(
  const _SIn& in,
  _SOut&& out,
  _ScanOp scan_op,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  const auto envs = default_envs(in);
  reserved::__mgmn_scan<true>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/// @brief In-place inclusive scan (self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>)
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__mgmn_scan<true>(data, data, envs, scan_op, identity, identity, call_env, "sharded::mgmn::inclusive_scan");
}

/**
 * @brief Exclusive scan `out[i] = fold(init, in[0..i-1])` over the global
 * index space — the global semantics, init entering the fold exactly once —
 * through the MGMN scan. Asynchronous, lane-ordered, capturable. @p out must
 * be co-partitioned with @p in; it may be @p in.
 */
_CCCL_TEMPLATE(class _SIn, class _SOut, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND
    sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void exclusive_scan(
  const _SIn& in,
  _SOut&& out,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_SIn> init_value,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  reserved::__mgmn_scan<false>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief In-place exclusive scan (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__mgmn_scan<false>(
    data, data, envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief Exclusive scan into @p out (self-bound: environments of @p in).
_CCCL_TEMPLATE(class _SIn, class _SOut, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND sharded_view<::cuda::std::remove_cvref_t<_SOut>>
                 _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void exclusive_scan(
  const _SIn& in,
  _SOut&& out,
  _ScanOp scan_op,
  view_element_t<_SIn> init_value,
  view_element_t<_SIn> identity = ::cuda::identity_element<_ScanOp, view_element_t<_SIn>>(),
  const _CallEnv& call_env      = {})
{
  const auto envs = default_envs(in);
  reserved::__mgmn_scan<false>(
    in, ::cuda::std::forward<_SOut>(out), envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief In-place exclusive scan (self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>)
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__mgmn_scan<false>(
    data, data, envs, scan_op, init_value, identity, call_env, "sharded::mgmn::exclusive_scan");
}

/// @brief In-place inclusive sum (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_sum(_S&& data, const _Envs& envs, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  mgmn::inclusive_scan(::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, elem_t{0}, call_env);
}

/// @brief In-place inclusive sum (self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_CallEnv>>))
_CCCL_HOST_API void inclusive_sum(_S&& data, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  using elem_t    = view_element_t<_S>;
  mgmn::inclusive_scan(::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, elem_t{0}, call_env);
}

/// @brief In-place exclusive sum (explicit environments).
_CCCL_TEMPLATE(class _S, class _Envs, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void
exclusive_sum(_S&& data, const _Envs& envs, view_element_t<_S> init_value = {}, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  mgmn::exclusive_scan(
    ::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, init_value, elem_t{0}, call_env);
}

/// @brief In-place exclusive sum (self-bound).
_CCCL_TEMPLATE(class _S, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>>)
_CCCL_HOST_API void exclusive_sum(_S&& data, view_element_t<_S> init_value = {}, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  using elem_t    = view_element_t<_S>;
  mgmn::exclusive_scan(
    ::cuda::std::forward<_S>(data), envs, ::cuda::std::plus<elem_t>{}, init_value, elem_t{0}, call_env);
}

/**
 * @brief Reduction through the MGMN reduce (per-shard `cub::DeviceReduce`,
 * then `all_reduce` of the P partials): `init (+) fold(all elements)`, the
 * init entering exactly once, written to EVERY lane's output — `outs[g]` on
 * lane g's stream (the MGMN "broadcasted" result). Asynchronous,
 * lane-ordered, capturable.
 *
 * @param outs Indexable range of P device pointers (or output iterators),
 *        one per shard.
 * @param identity The operator's identity element (the partial of an empty
 *        shard); defaulted where `cuda::identity_element` knows the operator.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Outs, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void reduce_into_lanes(
  const _S& data,
  const _Envs& envs,
  const _Outs& outs,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  const ::std::size_t __n = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::mgmn::reduce_into_lanes: fewer environments than shards");
  }
  if (static_cast<::std::size_t>(outs.size()) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::mgmn::reduce_into_lanes: fewer outputs than shards");
  }
  if (__n == 0)
  {
    return;
  }
  const reserved::__mgmn_bracket<_Envs, _CallEnv> __bracket(envs, __n, call_env);

  using __out_t       = ::cuda::std::remove_cvref_t<decltype(outs[::std::size_t{0}])>;
  const auto __comms  = sharded::make_communicators(envs, __n);
  const auto __menvs  = sharded::mgmn_envs(envs, __n);
  const auto __inputs = reserved::__mgmn_pointers<const _Tp*>(data);
  const auto __sizes  = reserved::__mgmn_sizes(data);
  ::std::vector<__out_t> __outputs;
  __outputs.reserve(__n);
  for (const auto __g : each(__n))
  {
    __outputs.push_back(outs[__g]);
  }

  ::cuda::experimental::mgmn::reduce(
    ::cuda::experimental::broadcasted, __comms, __menvs, __inputs, __sizes, __outputs, init_value, reduce_op, identity);
}

/// @brief As above, self-bound (environments of @p data).
_CCCL_TEMPLATE(class _S, class _Outs, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Outs>>))
_CCCL_HOST_API void reduce_into_lanes(
  const _S& data,
  const _Outs& outs,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  mgmn::reduce_into_lanes(data, envs, outs, reduce_op, init_value, identity, call_env);
}

/**
 * @brief Synchronous reduction through the MGMN reduce, returning the value:
 * `init (+) fold(all elements)`. Convenience over `reduce_into_lanes` (one
 * scratch scalar per lane from the lane's resource, lane 0's copied back);
 * refuses under `sync_policy::forbid` and under capture.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] _CCCL_HOST_API _Tp reduce(
  const _S& data,
  const _Envs& envs,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  constexpr const char* __what = "sharded::mgmn::reduce";
  const ::std::size_t __n      = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < __n)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(__what) + ": fewer environments than shards");
  }
  require_sync_allowed(call_env, __what);
  places::check_not_capturing(nullptr, __what);
  for (const auto __g : each(__n))
  {
    places::check_not_capturing(::cuda::get_stream(envs[__g]).get(), __what);
  }
  if (__n == 0)
  {
    return init_value;
  }

  ::std::vector<_Tp*> __outs(__n, nullptr);
  for (const auto __g : each(__n))
  {
    auto __mr   = ::cuda::mr::get_memory_resource(envs[__g]);
    __outs[__g] = static_cast<_Tp*>(__mr.allocate(::cuda::get_stream(envs[__g]), sizeof(_Tp), alignof(_Tp)));
  }
  mgmn::reduce_into_lanes(data, envs, __outs, reduce_op, init_value, identity);

  _Tp __result                  = init_value;
  const ::cuda::stream_ref __s0 = ::cuda::get_stream(envs[0]);
  cuda_safe_call(cudaMemcpyAsync(&__result, __outs[0], sizeof(_Tp), cudaMemcpyDefault, __s0.get()));
  cuda_safe_call(cudaStreamSynchronize(__s0.get()));
  for (const auto __g : each(__n))
  {
    auto __mr = ::cuda::mr::get_memory_resource(envs[__g]);
    __mr.deallocate(::cuda::get_stream(envs[__g]), __outs[__g], sizeof(_Tp), alignof(_Tp));
  }
  return __result;
}

/// @brief As above, self-bound (environments of @p data).
_CCCL_TEMPLATE(class _S, class _ReduceOp, class _Tp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ReduceOp>>))
[[nodiscard]] _CCCL_HOST_API _Tp reduce(
  const _S& data,
  _ReduceOp reduce_op,
  _Tp init_value,
  _Tp identity             = ::cuda::identity_element<_ReduceOp, _Tp>(),
  const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return mgmn::reduce(data, envs, reduce_op, init_value, identity, call_env);
}
} // namespace mgmn
} // namespace cuda::experimental::sharded
