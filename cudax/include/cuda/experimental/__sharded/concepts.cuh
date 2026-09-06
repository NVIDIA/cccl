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
 * @brief Concepts for sharded structures: `sharded_view`, `owning_sharded`,
 *        per-shard environments, self-binding, and per-call environments.
 *
 * The design separates three things with three different lifecycles:
 *
 * 1. **The view** (`sharded_view`): plain data describing what the structure
 *    is — for each shard, a contiguous element range, its *region* in the
 *    global index space (offset + size), and an equality-comparable *place*
 *    identity saying where the bytes live. A view owns no elements and holds
 *    no execution resources (span/mdspan semantics: no capacity, no
 *    allocation, no growth). Views are what interop, inspection and
 *    transport consume.
 * 2. **The per-shard environments** (`sharded_env`, `sharded_env_range`):
 *    standard queryable environments supplying, for shard `i`, the stream to
 *    order work on (`cuda::get_stream`, mandatory) and — for algorithms that
 *    need scratch — a memory resource (`cuda::mr::get_memory_resource`).
 *    Environments are either passed explicitly alongside a view, or derived
 *    from structures that carry their own binding (`self_bound` /
 *    `default_envs`, in the spirit of `std::execution`'s `get_env`).
 * 3. **The per-call environment** ("call env"): resources of the scope where
 *    any cross-shard step runs — a result/join stream (its presence selects
 *    the asynchronous contract), a host-accessible staging resource, and the
 *    synchronization policy (`get_sync_policy`).
 *
 * Semantic guarantees of `sharded_view` (checked by `validate()`, not
 * expressible in the concept): shard regions are pairwise disjoint, ordered
 * by global offset, and tile `[0, total extent)` exactly; empty shards are
 * permitted. Algorithms such as scan and adjacent_difference rely on these.
 *
 * Deliberate v1 simplifications (recorded for review): descriptor and
 * structure access are member/field-structural (`.data`, `.size`,
 * `.global_offset`, `.place` on descriptors; `.num_shards()`, `.shard(i)` on
 * structures) rather than customization-point objects. Foreign structures
 * participate by exposing this shape — `basic_shard_view` is the ready-made
 * portable descriptor value type — or through a thin wrapper. Lifting the
 * access layer to CPOs is a mechanical follow-up if a foreign type ever
 * cannot provide the shape.
 *
 * Descriptor `.data` is pointer-only in v1: views double as the storage and
 * transport currency, where addresses are load-bearing (aliasing validation,
 * contiguity, ABI). The planned relaxation is a wider sibling concept for
 * ALGORITHM ARGUMENTS whose `.data` may be any random-access iterator (both
 * input and output positions, constrained per parameter by readability /
 * writability), so per-shard CUB can consume fancy iterators — additive, and
 * the compute paths already use pure iterator arithmetic in anticipation.
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

#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__utility/declval.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::sharded
{
// ===========================================================================
// Helper unary concepts (the C++17 emulation needs named unary concepts for
// _Satisfies; see __multi_gpu/concepts.h for the precedent)
// ===========================================================================

template <class _Tp>
_CCCL_CONCEPT __convertible_to_size = ::cuda::std::convertible_to<_Tp, ::std::size_t>;

template <class _Tp>
_CCCL_CONCEPT __convertible_to_stream_ref = ::cuda::std::convertible_to<_Tp, ::cuda::stream_ref>;

template <class _Tp>
_CCCL_CONCEPT __shard_data_pointer = ::cuda::std::is_pointer_v<::cuda::std::remove_cvref_t<_Tp>>;

namespace reserved
{
// The concepts below only promise that counts are *convertible* to size_t
// (foreign size types may be signed or narrower), so these normalize the
// type once at function entry instead of a static_cast at every use.

//! @brief Number of environments in a `sharded_env_range`, as `size_t`.
template <class _Envs>
[[nodiscard]] ::std::size_t __env_count(const _Envs& __envs)
{
  return static_cast<::std::size_t>(__envs.size());
}

//! @brief Number of shards of a `sharded_view`, as `size_t`.
template <class _S>
[[nodiscard]] ::std::size_t __shard_count(const _S& __s)
{
  return static_cast<::std::size_t>(__s.num_shards());
}
} // namespace reserved

template <class _Tp>
_CCCL_CONCEPT __equality_comparable_place = ::cuda::std::equality_comparable<::cuda::std::remove_cvref_t<_Tp>>;

// ===========================================================================
// Shard descriptor
// ===========================================================================

//! @brief A shard descriptor: one contiguous, placed piece of a sharded
//! structure, as plain data.
//!
//! Requirements (field/member-structural): `data` (pointer to elements),
//! `size` (element count), `global_offset` (first global index covered),
//! `place` (equality-comparable place identity — any type; our containers
//! use `data_place`, foreign models bring their own).
template <class _Sd>
_CCCL_CONCEPT shard_descriptor = _CCCL_REQUIRES_EXPR((_Sd), const _Sd& __d)(
  _Satisfies(__shard_data_pointer) __d.data,
  _Satisfies(__convertible_to_size) __d.size,
  _Satisfies(__convertible_to_size) __d.global_offset,
  _Satisfies(__equality_comparable_place) __d.place);

//! @brief Element type of a shard descriptor.
template <class _Sd>
using shard_element_t = ::cuda::std::remove_pointer_t<::cuda::std::remove_cvref_t<decltype(_Sd::data)>>;

//! @brief Ready-made portable shard descriptor value type.
//!
//! Foreign structures that do not already expose descriptor-shaped shards can
//! return this from their `shard(i)` accessor. `_PlaceId` is any
//! equality-comparable identity (`int` device ordinal, a `{device, domain}`
//! pair, a rank, ...).
template <class _Tp, class _PlaceId = int>
struct basic_shard_view
{
  _Tp* data                   = nullptr; //!< pointer to the shard's elements
  ::std::size_t size          = 0; //!< number of elements
  ::std::size_t global_offset = 0; //!< first global index covered
  _PlaceId place{}; //!< equality-comparable place identity
};

//! @brief A minimal owned-descriptor sharded view: a vector of
//! `basic_shard_view` plus the structure accessors. The simplest possible
//! model of `sharded_view` — what `make_sharded_view` returns.
template <class _Tp, class _PlaceId = int>
struct basic_sharded_view
{
  ::std::vector<basic_shard_view<_Tp, _PlaceId>> shards;

  [[nodiscard]] ::std::size_t num_shards() const noexcept
  {
    return shards.size();
  }
  [[nodiscard]] const basic_shard_view<_Tp, _PlaceId>& shard(::std::size_t __i) const noexcept
  {
    return shards[__i];
  }
};

//! @brief Upgrade an ordered sequence of contiguous pieces — a
//! `vector<span<T>>` in spirit and in practice — into a sharded view.
//!
//! A `vector<span<T>>` is exactly the *data* of a sharded view; what it
//! lacks are the two facts the algorithms consume beyond the bytes: each
//! piece's position in the global index space and an equality-comparable
//! *place* identity (defaulted to the piece index here; pass real
//! identities through the second overload when locality matters).
//!
//! On offsets: under the view's ordered+tiling guarantees the offsets are
//! *redundant* data — `offset_i` is the prefix sum of the sizes — so a bare
//! `vector<span<T>>` carries enough information. The requirement is that
//! regions be *obtainable*, and there are two conformance routes: models
//! that already have offsets provide them (stored descriptor field, O(1)
//! region queries — the containers, and structures with native ranges);
//! models that don't go through this factory, which derives them once
//! (one O(num_pieces) running sum at adaptation time). Storing the result
//! in the descriptor keeps every region query O(1) on algorithm hot paths
//! and keeps a descriptor a self-contained value — a lone shard knows
//! where it belongs without its siblings, which is what lets descriptors
//! travel. A no-store lazy route (per-query inference) would be strictly
//! less efficient where it matters; if a consumer ever needs to model the
//! concept directly without storing offsets and without adapting, an
//! optional-offset query protocol is a compatible future extension.
//! Users never supply offsets by hand on any route. Anything with
//! `data()` and `size()` qualifies as a piece (`cuda::std::span`,
//! `std::span`, ...).
template <class _SpanLike, class _PlaceId = int>
[[nodiscard]] auto make_sharded_view(const ::std::vector<_SpanLike>& __pieces)
{
  using _Tp = ::cuda::std::remove_pointer_t<decltype(::cuda::std::declval<const _SpanLike&>().data())>;
  basic_sharded_view<_Tp, _PlaceId> __v;
  __v.shards.reserve(__pieces.size());
  ::std::size_t __offset = 0;
  ::std::size_t __idx    = 0;
  for (const auto& __p : __pieces)
  {
    __v.shards.push_back({__p.data(), static_cast<::std::size_t>(__p.size()), __offset, static_cast<_PlaceId>(__idx)});
    __offset += static_cast<::std::size_t>(__p.size());
    ++__idx;
  }
  return __v;
}

//! @brief As above, with caller-supplied place identities (one per piece).
template <class _SpanLike, class _PlaceId>
[[nodiscard]] auto make_sharded_view(const ::std::vector<_SpanLike>& __pieces, const ::std::vector<_PlaceId>& __places)
{
  using _Tp = ::cuda::std::remove_pointer_t<decltype(::cuda::std::declval<const _SpanLike&>().data())>;
  if (__places.size() != __pieces.size())
  {
    throw ::std::invalid_argument("make_sharded_view: one place identity per piece required");
  }
  basic_sharded_view<_Tp, _PlaceId> __v;
  __v.shards.reserve(__pieces.size());
  ::std::size_t __offset = 0;
  for (::std::size_t __i = 0; __i < __pieces.size(); ++__i)
  {
    __v.shards.push_back(
      {__pieces[__i].data(), static_cast<::std::size_t>(__pieces[__i].size()), __offset, __places[__i]});
    __offset += static_cast<::std::size_t>(__pieces[__i].size());
  }
  return __v;
}

// ===========================================================================
// Sharded view (the mapping tier)
// ===========================================================================

//! @brief A sharded view: an indexed collection of shard descriptors over a
//! 1-D global index space.
//!
//! Syntactic requirements: `num_shards()` and `shard(i)` yielding a
//! `shard_descriptor`. Semantic guarantees (see `validate()`): regions
//! pairwise disjoint, ordered by `global_offset`, tiling `[0, extent)`
//! exactly; view semantics (no element ownership through this interface).
template <class _S>
_CCCL_CONCEPT sharded_view = _CCCL_REQUIRES_EXPR((_S), const _S& __s)(
  _Satisfies(__convertible_to_size) __s.num_shards(),
  requires(shard_descriptor<::cuda::std::remove_cvref_t<decltype(__s.shard(::std::size_t{0}))>>));

//! @brief Descriptor type of a sharded view.
template <class _S>
using shard_descriptor_t =
  ::cuda::std::remove_cvref_t<decltype(::cuda::std::declval<const _S&>().shard(::std::size_t{0}))>;

//! @brief Element type of a sharded view.
template <class _S>
using view_element_t = shard_element_t<shard_descriptor_t<::cuda::std::remove_cvref_t<_S>>>;

template <class _Tp>
_CCCL_CONCEPT __has_capacity_field =
  _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __d)(_Satisfies(__convertible_to_size) __d.capacity);

//! @brief An owning sharded structure: a `sharded_view` whose shards
//! additionally expose `capacity` (allocated element count, >= size) and
//! which supports the atomic size-mutation verb `commit_sizes`.
//!
//! This is the home of the size-mutating algorithm family (`copy_if`,
//! `unique`, sort): shrinking shards' logical sizes and re-tiling the global
//! offsets are container-metadata operations that a non-owning view must not
//! (and cannot) express. `commit_sizes(new_sizes)` applies one size per
//! shard (each `<= capacity`) and restores the view invariants in a single
//! step — `validate()` holds before and after, with no observable
//! intermediate state (the consumers all compute every new size first and
//! then apply: batch-then-commit is the algorithm shape, the verb names it).
//! Capacities never change through this interface: no growth, no
//! reallocation — redistribution stays an explicit rebuild.
template <class _S>
_CCCL_CONCEPT owning_sharded = _CCCL_REQUIRES_EXPR((_S), _S& __s, const ::std::vector<::std::size_t>& __sizes)(
  requires(sharded_view<_S>),
  requires(__has_capacity_field<::cuda::std::remove_cvref_t<decltype(__s.shard(::std::size_t{0}))>>),
  _Same_as(void) __s.commit_sizes(__sizes));

//! @brief Check the `sharded_view` semantic guarantees at runtime (debug
//! aid; concepts cannot express semantics).
//!
//! Verifies: descriptors ordered by `global_offset`, regions disjoint and
//! exactly tiling `[0, extent)` where `extent` is the last region's end.
//! Empty shards are permitted anywhere.
_CCCL_TEMPLATE(class _S)
_CCCL_REQUIRES(sharded_view<_S>)
[[nodiscard]] bool validate(const _S& __s)
{
  const ::std::size_t __n = reserved::__shard_count(__s);
  ::std::size_t __next    = 0;
  for (::std::size_t __i = 0; __i < __n; ++__i)
  {
    const auto& __d = __s.shard(__i);
    if (static_cast<::std::size_t>(__d.global_offset) != __next)
    {
      return false; // gap, overlap, or out-of-order region
    }
    __next += static_cast<::std::size_t>(__d.size);
  }
  return true;
}

// ===========================================================================
// Per-shard environments (the binding tier)
// ===========================================================================

//! @brief A per-shard environment: anything the `cuda::get_stream`
//! customization point can extract a stream from (a `.stream()` /
//! `.get_stream()` member, a `query(get_stream_t)` env, or something
//! convertible to `stream_ref`).
template <class _Env>
_CCCL_CONCEPT sharded_env =
  _CCCL_REQUIRES_EXPR((_Env), const _Env& __e)(_Satisfies(__convertible_to_stream_ref)::cuda::get_stream(__e));

template <class _Tp>
_CCCL_CONCEPT __not_void = !::cuda::std::is_void_v<_Tp>;

//! @brief A per-shard environment that can also allocate: additionally
//! answers `cuda::mr::get_memory_resource`. Required by scratch-bearing
//! algorithms (reduce, scan, histogram, ...); the map family needs only
//! `sharded_env`.
template <class _Env>
_CCCL_CONCEPT sharded_alloc_env = _CCCL_REQUIRES_EXPR((_Env), const _Env& __e)(
  requires(sharded_env<_Env>), _Satisfies(__not_void)::cuda::mr::get_memory_resource(__e));

//! @brief An indexed family of per-shard environments: `size()` and
//! `operator[](i)` yielding a `sharded_env`. `envs[i]` binds shard `i`.
template <class _Range>
_CCCL_CONCEPT sharded_env_range = _CCCL_REQUIRES_EXPR((_Range), const _Range& __r)(
  _Satisfies(__convertible_to_size) __r.size(),
  requires(sharded_env<::cuda::std::remove_cvref_t<decltype(__r[::std::size_t{0}])>>));

//! @brief As `sharded_env_range`, with allocating environments.
template <class _Range>
_CCCL_CONCEPT sharded_alloc_env_range = _CCCL_REQUIRES_EXPR((_Range), const _Range& __r)(
  _Satisfies(__convertible_to_size) __r.size(),
  requires(sharded_alloc_env<::cuda::std::remove_cvref_t<decltype(__r[::std::size_t{0}])>>));

//! @brief A self-bound sharded structure: a `sharded_view` for which
//! `default_envs(s)` (found by argument-dependent lookup) yields a
//! `sharded_env_range` with one environment per shard.
//!
//! This is an *optional* capability in the spirit of `std::execution`'s
//! `get_env`: the view concept never stores environments; types built by a
//! provider (containers whose shards recorded their streams and places at
//! construction) can answer the query anyway. Pure transported views do not
//! model it and are used through the explicit-environment overloads.
template <class _S>
_CCCL_CONCEPT self_bound = _CCCL_REQUIRES_EXPR((_S), const _S& __s)(
  requires(sharded_view<_S>), requires(sharded_env_range<::cuda::std::remove_cvref_t<decltype(default_envs(__s))>>));

// ===========================================================================
// Per-call environment (the combine-scope tier)
// ===========================================================================

// ===========================================================================
// The composition property (per-call): lane-ordered (default) or bracketed
// ===========================================================================

//! @brief Per-call composition selector: how an asynchronous call orders
//! against the call environment's stream.
enum class composition
{
  lane_ordered, //!< default: enqueue on the lanes, no call-stream edges
  bracketed //!< fork-all/join-all against the call stream, per call
};

//! @brief Query object for the per-call composition property (defaults to
//! `composition::lane_ordered` when absent).
struct get_composition_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, get_composition_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
  {
    return __env.query(*this);
  }
};
_CCCL_GLOBAL_CONSTANT get_composition_t get_composition{};

//! @brief Read the composition property off a call environment
//! (`composition::lane_ordered` when the environment does not carry one).
template <class _CallEnv>
[[nodiscard]] constexpr composition query_composition(const _CallEnv& __env) noexcept
{
  if constexpr (::cuda::std::execution::__queryable_with<_CallEnv, get_composition_t>)
  {
    return __env.query(get_composition);
  }
  else
  {
    (void) __env;
    return composition::lane_ordered;
  }
}

//! @brief Synchronization policy carried by a per-call environment.
enum class sync_policy
{
  allow, //!< best effort: the call may synchronize with the host where the
         //!< algorithm's documented contract says so
  forbid //!< any would-be host synchronization throws `std::runtime_error`
         //!< *before* the blocking call (the capture-guard discipline)
};

//! @brief Query tag for the synchronization policy of a per-call
//! environment: `env.query(get_sync_policy_t{}) -> sync_policy`.
struct get_sync_policy_t
{
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(::cuda::std::execution::__queryable_with<_Env, get_sync_policy_t>)
  [[nodiscard]] constexpr sync_policy operator()(const _Env& __env) const noexcept
  {
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT get_sync_policy_t get_sync_policy{};

//! @brief The synchronization policy of a per-call environment;
//! `sync_policy::allow` when the environment does not carry one.
template <class _CallEnv>
[[nodiscard]] constexpr sync_policy query_sync_policy(const _CallEnv& __env) noexcept
{
  if constexpr (::cuda::std::execution::__queryable_with<_CallEnv, get_sync_policy_t>)
  {
    return __env.query(get_sync_policy_t{});
  }
  else
  {
    (void) __env;
    return sync_policy::allow;
  }
}

//! @brief Does this per-call environment select the asynchronous contract?
//!
//! Presence of a stream (via `cuda::get_stream`) selects it: the call
//! returns after enqueue and performs no host synchronization (for the
//! operations whose documented contract offers the asynchronous form).
//! Ordering follows the composition contract: lane-ordered by default
//! (`composition::lane_ordered`), sealed against the call stream under
//! `composition::bracketed`; combine-bearing terminators deliver their
//! result on the call stream regardless (their edges are definitional).
template <class _CallEnv>
_CCCL_CONCEPT async_call_env = sharded_env<_CallEnv>;

//! @brief Guard for the `sync_policy::forbid` contract: throw before a
//! would-be host synchronization, leaving all state valid.
//!
//! Every internal host-blocking site of the algorithm tier routes through
//! this (the `check_not_capturing` discipline, generalized). Amortized state
//! warm-up (handle/plan creation) is exempt by contract: warm up before
//! entering a no-sync region.
template <class _CallEnv>
void require_sync_allowed(const _CallEnv& __env, const char* __what)
{
  if (query_sync_policy(__env) == sync_policy::forbid)
  {
    throw ::std::runtime_error(
      ::std::string(__what)
      + ": operation would synchronize with the host, but the call "
        "environment carries sync_policy::forbid");
  }
}

//! @brief An empty per-call environment: synchronous contract, best-effort
//! policy. The default for the convenience overloads.
using default_call_env = ::cuda::std::execution::env<>;

namespace reserved
{
//! @brief Check that two sharded views are co-partitioned: same shard count
//! and, per shard, identical global regions.
template <class _SA, class _SB>
void __check_copartitioned(const _SA& __a, const _SB& __b, const char* __what)
{
  const ::std::size_t __n = __shard_count(__a);
  if (__n != __shard_count(__b))
  {
    throw ::std::invalid_argument(::std::string(__what) + ": shard count mismatch");
  }
  for (::std::size_t __g = 0; __g < __n; ++__g)
  {
    if (static_cast<::std::size_t>(__a.shard(__g).size) != static_cast<::std::size_t>(__b.shard(__g).size)
        || static_cast<::std::size_t>(__a.shard(__g).global_offset)
             != static_cast<::std::size_t>(__b.shard(__g).global_offset))
    {
      throw ::std::invalid_argument(::std::string(__what) + ": shard regions differ (not co-partitioned)");
    }
  }
}
} // namespace reserved
} // namespace cuda::experimental::sharded

// NOLINTEND(bugprone-reserved-identifier)
