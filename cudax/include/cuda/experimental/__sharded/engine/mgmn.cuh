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
 * @brief The MGMN adapter: what lets the multi-GPU multi-node algorithms of
 *        `cuda/experimental/__multi_gpu/algorithm/` run as the engines of
 *        the sharded verbs, in one process, over the shared address space.
 *
 * The communicator the engines run over is `places::places_communicator`
 * (`__places/places_communicator.cuh`): P ranks sharing one state, the
 * shared address space as the network, NCCL-style groups. A `place_group`
 * owns one communicator group per lane (`place_group::communicators`), so
 * its event pools live as long as the group.
 *
 * The adapter — `make_communicators`, `mgmn_envs`, and the `reserved`
 * helpers — turns a sharded view and its per-shard environments into the
 * five lockstep ranges the MGMN multi-local-rank overloads consume
 * (communicators, environments, input iterators, sizes, output iterators).
 * `reserved::__mgmn_drive` is the driver every sharded verb whose engine is
 * an MGMN algorithm goes through: it applies the sharded contract — the
 * strict environment-count guard, the synchronous no-stream form,
 * lane-ordered or `composition::bracketed` composition, the capture-time
 * refusal — around one MGMN call. Its communicators are the lane's owned
 * group whenever the environments name one (`places::get_place_group`,
 * `places::get_lane_id`, one rank per shard in place order); environments
 * from elsewhere (foreign streams, foreign environment types) get a group
 * created for the call. `__mgmn_map` is its map-family spelling (rank-local
 * engines — the `reserved::mgmn_engine` reference transforms of
 * `reference/mgmn_transform.cuh`; the shard environments themselves when the engine
 * accepts them, empty shards skipped); the combine family
 * (`reduce.cuh`, `scan.cuh`) drives it with allocating environments that
 * also carry the determinism requirement (`__mgmn_alloc_env`), one rank per
 * shard, empty shards included (their partial is the identity).
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

#include <cub/thread/thread_operators.cuh> // is_cuda_binary_operator, is_cuda_std_plus_v

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/require.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__memory_resource/properties.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places_communicator.cuh>
#include <cuda/experimental/__places/stream_pool.cuh>
#include <cuda/experimental/__sharded/composition/verbs.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/concepts/guards.cuh>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
//! @brief The shared-address-space communicator the sharded engines run
//! over (see `__places/places_communicator.cuh`).
using ::cuda::experimental::places::places_communicator;

namespace reserved
{
//! @brief Maximum rank count of one communicator group (at most 64 shards).
using ::cuda::experimental::places::reserved::__max_fold_shards;
} // namespace reserved

// ============================================================================
// The adapter: sharded environments -> MGMN environments and communicators
// ============================================================================

namespace reserved
{
//! @brief One MGMN environment from one sharded environment: its stream and
//! its memory resource, nothing else. The resource is passed through as is;
//! the MGMN algorithms size their `cuda::buffer` temporaries from its
//! `default_queries` (`place_memory_resource` declares `device_accessible`).
template <class _Env>
[[nodiscard]] auto __mgmn_env(const _Env& __env)
{
  using __mr_t = ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(__env))>;
  static_assert(::cuda::mr::__has_default_queries<__mr_t>,
                "sharded::mgmn: the environment's memory resource must declare `default_queries` "
                "(the property set the MGMN algorithms build their temporaries from)");
  return ::cuda::std::execution::env<::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>,
                                     ::cuda::std::execution::prop<::cuda::mr::get_memory_resource_t, __mr_t>>{
    ::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>{
      ::cuda::get_stream, ::cuda::stream_ref{::cuda::get_stream(__env)}},
    ::cuda::std::execution::prop<::cuda::mr::get_memory_resource_t, __mr_t>{
      ::cuda::mr::get_memory_resource, __mr_t{::cuda::mr::get_memory_resource(__env)}}};
}
} // namespace reserved

//! @brief The MGMN environment type derived from the environments of an
//! allocating environment range.
template <class _Envs>
using mgmn_env_t =
  decltype(reserved::__mgmn_env(::cuda::std::declval<const ::cuda::std::remove_cvref_t<_Envs>&>()[::std::size_t{0}]));

/**
 * @brief The first @p count environments of @p envs as MGMN environments
 * (stream + memory resource), the range the MGMN algorithms iterate in
 * lockstep with the communicators.
 */
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<mgmn_env_t<_Envs>> mgmn_envs(const _Envs& envs, ::std::size_t count)
{
  if (reserved::__env_count(envs) < count)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::mgmn_envs: fewer environments than requested");
  }
  ::std::vector<mgmn_env_t<_Envs>> __result;
  __result.reserve(count);
  for (const auto __i : each(count))
  {
    __result.push_back(reserved::__mgmn_env(envs[__i]));
  }
  return __result;
}

/// @brief All environments of @p envs as MGMN environments.
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<mgmn_env_t<_Envs>> mgmn_envs(const _Envs& envs)
{
  return sharded::mgmn_envs(envs, reserved::__env_count(envs));
}

/**
 * @brief One NEW communicator group over the first @p count environments of
 * @p envs: rank i is environment i (its stream's device). For environments
 * born from a `place_group` lane prefer the lane's owned group
 * (`group.lane(k).communicators()`), whose event pools persist.
 */
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<places_communicator> make_communicators(const _Envs& envs, ::std::size_t count)
{
  if (reserved::__env_count(envs) < count)
  {
    _CCCL_THROW(::std::invalid_argument, "sharded::make_communicators: fewer environments than requested");
  }
  ::std::vector<::cuda::stream_ref> __streams;
  __streams.reserve(count);
  for (const auto __i : each(count))
  {
    __streams.push_back(::cuda::stream_ref{::cuda::get_stream(envs[__i])});
  }
  return places_communicator::create(__streams);
}

/// @brief One new communicator group over all environments of @p envs.
_CCCL_TEMPLATE(class _Envs)
_CCCL_REQUIRES(sharded_env_range<::cuda::std::remove_cvref_t<_Envs>>)
[[nodiscard]] ::std::vector<places_communicator> make_communicators(const _Envs& envs)
{
  return sharded::make_communicators(envs, reserved::__env_count(envs));
}

namespace reserved
{
//! @brief The values of @p __make over the shard indices in @p __lanes, as
//! one of the MGMN lockstep ranges.
template <class _Make>
[[nodiscard]] auto __mgmn_per_lane(const ::std::vector<::std::size_t>& __lanes, _Make __make)
{
  ::std::vector<::cuda::std::remove_cvref_t<decltype(__make(::std::size_t{0}))>> __result;
  __result.reserve(__lanes.size());
  for (const ::std::size_t __g : __lanes)
  {
    __result.push_back(__make(__g));
  }
  return __result;
}

//! @brief The data pointers of the shards in @p __lanes, as `_Ptr`: the
//! MGMN iterator range of a view.
template <class _Ptr, class _S>
[[nodiscard]] ::std::vector<_Ptr> __mgmn_pointers(const _S& __s, const ::std::vector<::std::size_t>& __lanes)
{
  return __mgmn_per_lane(__lanes, [&](::std::size_t __g) -> _Ptr {
    return static_cast<_Ptr>(__s.shard(__g).data);
  });
}

//! @brief The sizes of the shards in @p __lanes: the MGMN size range of a view.
template <class _S>
[[nodiscard]] ::std::vector<::std::size_t> __mgmn_sizes(const _S& __s, const ::std::vector<::std::size_t>& __lanes)
{
  return __mgmn_per_lane(__lanes, [&](::std::size_t __g) {
    return static_cast<::std::size_t>(__s.shard(__g).size);
  });
}

//! @brief Does the engine accept a sharded environment as is? It needs
//! `cuda::get_stream`; if the environment also answers
//! `cuda::mr::get_memory_resource`, the resource must declare the
//! `default_queries` the engine sizes its temporaries from (a
//! `place_memory_resource` does; a hand-rolled foreign resource may not).
template <class _Env, class = void>
inline constexpr bool __engine_env_v = ::cuda::std::__is_callable_v<::cuda::get_stream_t, const _Env&>;

template <class _Env>
inline constexpr bool
  __engine_env_v<_Env,
                 ::cuda::std::void_t<decltype(::cuda::mr::get_memory_resource(::cuda::std::declval<const _Env&>()))>> =
    ::cuda::std::__is_callable_v<::cuda::get_stream_t, const _Env&>
    && ::cuda::mr::__has_default_queries<
      ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(::cuda::std::declval<const _Env&>()))>>;

//! @brief The MGMN environment of a sharded environment the engine does not
//! accept as is: its stream only. The rank-local MGMN algorithms
//! (`transform`) allocate nothing, so the map family accepts any
//! `sharded_env_range`.
using __mgmn_stream_env_t =
  ::cuda::std::execution::env<::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>>;

template <class _Env>
[[nodiscard]] __mgmn_stream_env_t __mgmn_stream_env(const _Env& __env)
{
  return __mgmn_stream_env_t{::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>{
    ::cuda::get_stream, ::cuda::stream_ref{::cuda::get_stream(__env)}}};
}

//! @brief Tag standing for "no environment adaptation" in `__mgmn_drive`:
//! the shard environments are handed to the engine as they are — the
//! caller's range itself when every shard takes part.
struct __pass_envs_t
{};

//! @brief Tag standing for "no call environment" in `__mgmn_drive`: the
//! purely lane-ordered asynchronous contract of a verb that has no call
//! stream at all (`reduce_into_lanes`) — no refusal, no edge, no tail.
struct __lane_ordered_t
{};

//! @brief The requirements the combine-family verbs hand to the MGMN
//! environments, as the `cuda::execution::__get_requirements` property the
//! CUB environment overloads read: the call environment's own requirements
//! when it carries any (`cuda::execution::require(...)`), else
//! `determinism::run_to_run` when @p _RunToRun (CUB's local scans default to
//! `not_guaranteed`; reductions to `run_to_run`), else no requirement.
template <bool _RunToRun, class _CallEnv>
[[nodiscard]] auto __mgmn_requirements(const _CallEnv& __call_env)
{
  if constexpr (::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::execution::__get_requirements_t>)
  {
    using __reqs_t = ::cuda::std::remove_cvref_t<decltype(::cuda::execution::__get_requirements(__call_env))>;
    return ::cuda::std::execution::prop<::cuda::execution::__get_requirements_t, __reqs_t>{
      ::cuda::execution::__get_requirements_t{}, ::cuda::execution::__get_requirements(__call_env)};
  }
  else if constexpr (_RunToRun)
  {
    (void) __call_env;
    return ::cuda::execution::require(::cuda::execution::determinism::run_to_run);
  }
  else
  {
    (void) __call_env;
    return ::cuda::execution::require();
  }
}

//! @brief Can CUB honor `determinism::run_to_run` for a scan of `_Tp` under
//! `_Op`? (Its static contract: a known CUB operator on an integral type,
//! or `plus` on a floating-point type.) Reductions honor it for every
//! operator.
template <class _Op, class _Tp>
inline constexpr bool __scan_run_to_run_v =
  (::cuda::std::is_integral_v<_Tp> && CUB_NS_QUALIFIER::detail::is_cuda_binary_operator<_Op>)
  || (::cuda::std::is_floating_point_v<_Tp> && CUB_NS_QUALIFIER::detail::is_cuda_std_plus_v<_Op, _Tp>);

//! @brief A `cuda::mr::resource` that declares no `default_queries`, given
//! the `device_accessible` property set the MGMN algorithms size their
//! `cuda::buffer` temporaries from. The sharded contract admits any
//! stream-ordered resource of `cuda::mr::resource` shape (it only ever
//! allocated and deallocated through it); this keeps that contract on the
//! engine, which needs the property set.
template <class _Mr>
class __device_resource_adaptor
{
public:
  using default_queries = ::cuda::mr::properties_list<::cuda::mr::device_accessible>;

  explicit __device_resource_adaptor(_Mr __mr)
      : __mr_(::std::move(__mr))
  {}

  void*
  allocate(::cuda::stream_ref __stream, ::std::size_t __bytes, ::std::size_t __alignment = alignof(::std::max_align_t))
  {
    return __mr_.allocate(__stream, __bytes, __alignment);
  }
  void deallocate(::cuda::stream_ref __stream,
                  void* __ptr,
                  ::std::size_t __bytes,
                  ::std::size_t __alignment = alignof(::std::max_align_t))
  {
    __mr_.deallocate(__stream, __ptr, __bytes, __alignment);
  }
  void* allocate_sync(::std::size_t __bytes, ::std::size_t __alignment = alignof(::std::max_align_t))
  {
    return __mr_.allocate_sync(__bytes, __alignment);
  }
  void deallocate_sync(void* __ptr, ::std::size_t __bytes, ::std::size_t __alignment = alignof(::std::max_align_t))
  {
    __mr_.deallocate_sync(__ptr, __bytes, __alignment);
  }

  [[nodiscard]] friend bool operator==(const __device_resource_adaptor& __a, const __device_resource_adaptor& __b)
  {
    return __a.__mr_ == __b.__mr_;
  }
  [[nodiscard]] friend bool operator!=(const __device_resource_adaptor& __a, const __device_resource_adaptor& __b)
  {
    return !(__a == __b);
  }
  friend constexpr void get_property(const __device_resource_adaptor&, ::cuda::mr::device_accessible) noexcept {}

private:
  _Mr __mr_;
};

//! @brief The environment's resource as the engine consumes it: as is when
//! it declares `default_queries`, adapted otherwise.
template <class _Env>
using __mgmn_resource_t = ::cuda::std::conditional_t<
  ::cuda::mr::__has_default_queries<
    ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(::cuda::std::declval<const _Env&>()))>>,
  ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(::cuda::std::declval<const _Env&>()))>,
  __device_resource_adaptor<
    ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(::cuda::std::declval<const _Env&>()))>>>;

//! @brief The MGMN environment of an allocating sharded environment: its
//! stream, its memory resource (the MGMN algorithms size their `cuda::buffer`
//! temporaries from its `default_queries`; a resource declaring none is
//! wrapped in `__device_resource_adaptor`), and the requirements of
//! `__mgmn_requirements`.
template <class _Env, class _Reqs>
[[nodiscard]] auto __mgmn_alloc_env(const _Env& __env, const _Reqs& __reqs)
{
  using __mr_t    = __mgmn_resource_t<_Env>;
  using __sprop_t = ::cuda::std::execution::prop<::cuda::get_stream_t, ::cuda::stream_ref>;
  using __mprop_t = ::cuda::std::execution::prop<::cuda::mr::get_memory_resource_t, __mr_t>;
  return ::cuda::std::execution::env<__sprop_t, __mprop_t, _Reqs>{
    __sprop_t{::cuda::get_stream, ::cuda::stream_ref{::cuda::get_stream(__env)}},
    __mprop_t{::cuda::mr::get_memory_resource, __mr_t{::cuda::mr::get_memory_resource(__env)}},
    __reqs};
}

//! @brief The communicator group a `place_group` owns for the lane the
//! environments were manufactured on, when they were: every participating
//! environment names the same group (`places::get_place_group`) and the
//! same lane (`places::get_lane_id`), every shard takes part, and shard g's
//! stream is place g's stream on that lane — so rank g == shard g. Otherwise
//! (foreign environments, foreign streams, a permuted or partial range)
//! `nullptr`: the caller creates a group for the call.
template <class _Envs>
[[nodiscard]] const ::std::vector<places_communicator>*
__lane_communicators(const _Envs& __envs, const ::std::vector<::std::size_t>& __lanes, ::std::size_t __num_shards)
{
  using __env_t = ::cuda::std::remove_cvref_t<decltype(__envs[::std::size_t{0}])>;
  if constexpr (!::cuda::std::execution::__queryable_with<__env_t, places::get_place_group_t>
                || !::cuda::std::execution::__queryable_with<__env_t, places::get_lane_id_t>)
  {
    (void) __envs;
    (void) __lanes;
    (void) __num_shards;
    return nullptr;
  }
  else
  {
    if (__lanes.size() != __num_shards)
    {
      return nullptr;
    }
    places::place_group* const __group = places::query_place_group(__envs[__lanes[0]]);
    const auto __lane                  = places::query_lane_id(__envs[__lanes[0]]);
    if (__group == nullptr || !__lane.has_value() || __group->size() != __num_shards)
    {
      return nullptr;
    }
    for (const ::std::size_t __g : __lanes)
    {
      if (places::query_place_group(__envs[__g]) != __group || places::query_lane_id(__envs[__g]) != __lane
          || ::cuda::get_stream(__envs[__g]).get() != __group->get_stream(__g, *__lane))
      {
        return nullptr;
      }
    }
    return &__group->communicators(*__lane);
  }
}

//! @brief Driver of every sharded verb whose engine is an MGMN algorithm:
//! the sharded contract around one MGMN call.
//!
//! The per-call environment selects the contract — stream present =
//! asynchronous (LANE-ORDERED by default: each shard's work is enqueued on
//! its environment's stream and nothing else is touched; a call environment
//! carrying `composition::bracketed` seals the call against the call stream
//! with fork/join edges), no stream = synchronous convenience (refused under
//! `sync_policy::forbid` and under capture; every lane synchronized before
//! returning), `__lane_ordered_t` = asynchronous with no call stream at all.
//! A lane-ordered call whose call stream is capturing while a participating
//! shard's stream is not is REFUSED before anything is enqueued.
//!
//! The communicators are the lane's owned group when the environments name
//! one (`__lane_communicators`), a group created for the call otherwise.
//!
//! @tparam _AllLanes Every shard is a rank (the combine family: an empty
//!         shard's partial is the operator's identity and its output is
//!         still written); otherwise empty shards take no part — no rank,
//!         no launch, no edge (the map family).
//! @param __make_env Host callable `(const shard_env&) -> MGMN env`, or
//!        `__pass_envs_t{}` to hand the shard environments to the engine as
//!        they are (the caller's range itself when every shard takes part).
//! @param __body Host callable `(comms, envs, lanes)` receiving the
//!        communicator group (one rank per participating shard, in shard
//!        order), the matching MGMN environments, and the shard indices they
//!        stand for; it builds the remaining lockstep ranges and issues the
//!        MGMN call. The host never synchronizes inside the asynchronous
//!        forms.
template <bool _AllLanes, class _S, class _Envs, class _CallEnv, class _MakeEnv, class _Body>
_CCCL_HOST_API void __mgmn_drive(
  const _S& __data,
  const _Envs& __envs,
  const _CallEnv& __call_env,
  const char* __what,
  _MakeEnv __make_env,
  _Body __body)
{
  const ::std::size_t __num_shards = __shard_count(__data);
  __check_env_count(__envs, __num_shards, __what);

  constexpr bool __no_call_env      = ::cuda::std::is_same_v<_CallEnv, __lane_ordered_t>;
  constexpr bool __is_async         = __no_call_env || async_call_env<_CallEnv>;
  [[maybe_unused]] bool __bracketed = false;

  ::std::vector<::std::size_t> __lanes;
  __lanes.reserve(__num_shards);
  for (const auto __g : each(__num_shards))
  {
    if (_AllLanes || __data.shard(__g).size != 0)
    {
      __lanes.push_back(__g);
    }
  }

  if constexpr (!__is_async)
  {
    // Refusals first, before any CUDA call (the entry-guard discipline).
    require_sync_allowed(__call_env, __what);
    __check_envs_not_capturing(__envs, __num_shards, __what);
  }
  else if constexpr (!__no_call_env)
  {
    __bracketed = query_composition(__call_env) == composition::bracketed;
    if (!__bracketed && places::stream_in_capture(::cuda::get_stream(__call_env).get()))
    {
      // Lane-ordered under capture: every participating lane must already
      // be part of the capture, or its work would silently escape the graph.
      for (const ::std::size_t __g : __lanes)
      {
        if (!places::stream_in_capture(::cuda::get_stream(__envs[__g]).get()))
        {
          _CCCL_THROW(::std::runtime_error,
                      ::std::string(__what)
                        + ": lane-ordered asynchronous call during CUDA graph capture requires the "
                          "shard environments' streams to be capturing (fork the lanes from the "
                          "capture stream once per pipeline, e.g. sharded_array::fork_from), or opt "
                          "into composition::bracketed on the call environment");
        }
      }
    }
  }

  if (__lanes.empty())
  {
    return;
  }

  if constexpr (__is_async && !__no_call_env)
  {
    if (__bracketed)
    {
      for (const ::std::size_t __g : __lanes)
      {
        __detail::__wait_stream_on(::cuda::get_stream(__envs[__g]).get(), ::cuda::get_stream(__call_env).get());
      }
    }
  }

  // The lane's owned communicators when the environments name a group lane;
  // a group for this call otherwise (foreign environments, partial ranges).
  const ::std::vector<places_communicator>* __comms = __lane_communicators(__envs, __lanes, __num_shards);
  ::std::vector<places_communicator> __call_comms;
  if (__comms == nullptr)
  {
    __call_comms = places_communicator::create(__mgmn_per_lane(__lanes, [&](::std::size_t __g) {
      return ::cuda::stream_ref{::cuda::get_stream(__envs[__g])};
    }));
    __comms      = &__call_comms;
  }

  if constexpr (::cuda::std::is_same_v<_MakeEnv, __pass_envs_t>)
  {
    if (__lanes.size() == __num_shards)
    {
      __body(*__comms, __envs, __lanes);
    }
    else
    {
      __body(*__comms,
             __mgmn_per_lane(__lanes,
                             [&](::std::size_t __g) {
                               return __envs[__g];
                             }),
             __lanes);
    }
  }
  else
  {
    __body(*__comms,
           __mgmn_per_lane(__lanes,
                           [&](::std::size_t __g) {
                             return __make_env(__envs[__g]);
                           }),
           __lanes);
  }

  if constexpr (__is_async && !__no_call_env)
  {
    if (__bracketed)
    {
      for (const ::std::size_t __g : __lanes)
      {
        __detail::__wait_stream_on(::cuda::get_stream(__call_env).get(), ::cuda::get_stream(__envs[__g]).get());
      }
    }
  }
  else if constexpr (!__is_async)
  {
    for (const ::std::size_t __g : __lanes)
    {
      cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(__envs[__g]).get()));
    }
  }
}

//! @brief The map-family spelling of `__mgmn_drive`: rank-local engines
//! (`transform`, `zip_transform`) that allocate nothing — the shard
//! environments as they are when the engine accepts them (`__engine_env_v`),
//! stream-only environments otherwise (any `sharded_env_range`), one rank
//! per NON-EMPTY shard.
template <class _S, class _Envs, class _CallEnv, class _Body>
_CCCL_HOST_API void
__mgmn_map(const _S& __data, const _Envs& __envs, const _CallEnv& __call_env, const char* __what, _Body __body)
{
  using __env_t = ::cuda::std::remove_cvref_t<decltype(__envs[::std::size_t{0}])>;
  if constexpr (__engine_env_v<__env_t>)
  {
    __mgmn_drive<false>(__data, __envs, __call_env, __what, __pass_envs_t{}, ::std::move(__body));
  }
  else
  {
    __mgmn_drive<false>(
      __data,
      __envs,
      __call_env,
      __what,
      [](const auto& __env) {
        return __mgmn_stream_env(__env);
      },
      ::std::move(__body));
  }
}
} // namespace reserved
} // namespace cuda::experimental::sharded
