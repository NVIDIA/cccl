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
 * @brief The view tier of the sharded concepts: `shard_descriptor`,
 *        `sharded_view`, `owning_sharded`, the portable descriptor value
 *        types (`basic_shard_view`, `basic_sharded_view`,
 *        `make_sharded_view`) and the runtime `validate()` check.
 *
 * See `<cuda/experimental/__sharded/concepts.cuh>` for the design overview
 * of the three concept tiers (view, per-shard environments, per-call
 * environment).
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

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/span>

#include <cstddef>
#include <stdexcept>
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
_CCCL_CONCEPT __shard_data_pointer = ::cuda::std::is_pointer_v<::cuda::std::remove_cvref_t<_Tp>>;

namespace reserved
{
// The concepts below only promise that counts are *convertible* to size_t
// (foreign size types may be signed or narrower), so these normalize the
// type once at function entry instead of a static_cast at every use.

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

  //! The elements as a span: the placeless, device-passable view. Not part of
  //! the `shard_descriptor` concept; a convenience every descriptor can offer.
  [[nodiscard]] _CCCL_HOST_API ::cuda::std::span<_Tp> span() const noexcept
  {
    return {data, size};
  }
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
//! This is the home of the size-mutating algorithm family (`select_if` /
//! `remove_if`, `unique`, sort): shrinking shards' logical sizes and re-tiling the global
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
} // namespace cuda::experimental::sharded

// NOLINTEND(bugprone-reserved-identifier)
