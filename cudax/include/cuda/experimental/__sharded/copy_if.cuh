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
 * @brief In-place selection over sharded arrays (`copy_if` / `filter` /
 *        `remove_if`): each place runs the device-scope primitive (CUB
 *        `DeviceSelect::If`, in place) on its shard, then the container's
 *        shard sizes and offsets are updated to the compacted result.
 *
 * These algorithms MUTATE shard sizes. That is exactly what the contiguous
 * backing cannot represent: shrinking a shard would leave a gap between its
 * valid elements and the next shard's, falsifying the read-as-one-array
 * contract of `contiguous_data()`, while compacting across the gap would
 * migrate elements onto other places than the caller asked for. They
 * therefore refuse contiguous (`allocate_contiguous`) arrays with
 * `std::invalid_argument`.
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

#include <cub/device/device_select.cuh>

#include <cuda/std/cstdint>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/composition.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>
#include <cuda/experimental/__stf/utility/exception_policy.cuh> // SCOPE

#include <algorithm>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
/// @brief Negation of a predicate (for `remove_if`).
template <typename _Tp, typename _Pred>
struct negate_pred_fn
{
  _Pred pred;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE bool operator()(_Tp val) const
  {
    return !pred(val);
  }
};
} // namespace reserved

// ============================================================================
// Concept-generic tier: compaction over any owning_sharded structure
// ============================================================================

namespace reserved
{
//! @brief Shared generic compaction: per-shard in-place `DeviceSelect::If`
//! on each shard's environment, counts staged to host, then ONE atomic
//! `commit_sizes`. Exception discipline as reviewed: every fallible resource
//! is acquired before any shard is mutated (release guard armed before the
//! acquisition loop), and once the first select has run the compaction is
//! irrevocable — on any failure the structure is left VALID and EMPTY.
//!
//! The size-mutation capability is probed at entry by committing the
//! CURRENT sizes: a no-op that throws exactly when the model does not
//! support mutation (e.g. contiguous backing), before anything changes —
//! the portable spelling of the container tier's contiguous refusal.
template <class _S, class _Envs, class _Pred, class _CallEnv>
[[nodiscard]] _CCCL_HOST_API size_t
__copy_if_generic(_S&& data, const _Envs& envs, _Pred pred, const _CallEnv& call_env, const char* what)
{
  using count_type               = ::cuda::std::int64_t;
  const ::std::size_t num_shards = reserved::__shard_count(data);
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(what) + ": fewer environments than shards");
  }
  if (num_shards == 0)
  {
    return 0;
  }

  // Refusals first, before any CUDA call: size write-back synchronizes.
  require_sync_allowed(call_env, what);
  places::check_not_capturing(nullptr, what);
  for (const auto g : each(num_shards))
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), what);
  }

  // Entry probe: committing the current sizes is a no-op that throws exactly
  // when this model cannot mutate sizes, before any element moves.
  ::std::vector<size_t> new_sizes(num_shards);
  for (const auto g : each(num_shards))
  {
    new_sizes[g] = static_cast<size_t>(data.shard(g).size);
  }
  data.commit_sizes(new_sizes);

  constexpr bool __env_has_mr = ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
                             || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  count_type* h_new_sizes     = nullptr;
  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_new_sizes =
      static_cast<count_type*>(staging_mr.allocate_sync(num_shards * sizeof(count_type), alignof(count_type)));
  }
  else
  {
    h_new_sizes = static_cast<count_type*>(reserved::__pinned_staging(num_shards * sizeof(count_type)));
  }
  ::std::fill(h_new_sizes, h_new_sizes + num_shards, count_type{0});

  // Phase 0: acquire every fallible resource before any shard is mutated;
  // the release guard is armed first so mid-acquisition failures free what
  // was acquired.
  using scratch_mr_t = ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(envs[::std::size_t{0}]))>;
  ::std::vector<::std::tuple<scratch_mr_t, count_type*, cudaStream_t>> d_counts;
  d_counts.reserve(num_shards);
  SCOPE(exit)
  {
    for (auto& [mr, ptr, stream] : d_counts)
    {
      mr.deallocate(::cuda::stream_ref{stream}, ptr, sizeof(count_type), alignof(count_type));
    }
  };
  for (const auto g : each(num_shards))
  {
    const auto& s = data.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    auto mr                               = ::cuda::mr::get_memory_resource(envs[g]);
    count_type* d_num = static_cast<count_type*>(mr.allocate(shard_stream, sizeof(count_type), alignof(count_type)));
    d_counts.emplace_back(mv(mr), d_num, shard_stream.get());
  }

  // Phase 1: in-place select per shard. From here through commit_sizes the
  // compaction is irrevocable: on any failure, leave the structure VALID and
  // EMPTY rather than compacted under stale sizes (the entry probe proved
  // commit_sizes cannot refuse on this model, and zero never exceeds
  // capacity).
  SCOPE(fail)
  {
    data.commit_sizes(::std::vector<size_t>(num_shards, 0));
  };
  ::std::size_t next = 0;
  for (const auto g : each(num_shards))
  {
    const auto& s = data.shard(g);
    if (s.size == 0)
    {
      continue;
    }
    count_type* d_num                     = ::std::get<1>(d_counts[next++]);
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    stream_scope scope(shard_stream.get());
    cuda_safe_call(cub::DeviceSelect::If(s.data, d_num, static_cast<::cuda::std::int64_t>(s.size), pred, envs[g]));
    cuda_safe_call(
      cudaMemcpyAsync(&h_new_sizes[g], d_num, sizeof(count_type), cudaMemcpyDeviceToHost, shard_stream.get()));
  }
  barrier(envs);

  size_t total = 0;
  for (const auto g : each(num_shards))
  {
    _CCCL_ASSERT(h_new_sizes[g] >= 0 && static_cast<size_t>(h_new_sizes[g]) <= data.shard(g).size,
                 "sharded compaction: select returned an out-of-range count");
    new_sizes[g] = static_cast<size_t>(h_new_sizes[g]);
    total += new_sizes[g];
  }
  data.commit_sizes(new_sizes);

  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    staging_mr.deallocate_sync(h_new_sizes, num_shards * sizeof(count_type), alignof(count_type));
  }
  // (counters freed by SCOPE(exit); arena staging is cached)

  return total;
}

//! @brief Out-of-place selection: per-shard `DeviceSelect::If` from a source
//! view into an owning destination, then the destination's sizes committed
//! atomically. The source is untouched; the destination ends RAGGED (its
//! per-shard sizes are the data-dependent selected counts, offsets
//! re-tiled).
//!
//! Cross-space contract (the source's index space and the destination's
//! compacted space differ, so no co-partition check applies): equal shard
//! counts, and per shard `dst.capacity >= src.size` — checked at entry from
//! the descriptors, before anything moves. The destination's prior contents
//! and sizes are irrelevant (overwritten up to capacity; commit installs the
//! new sizes). On failure past the point of first mutation the destination
//! is left VALID and EMPTY.
template <class _SIn, class _SOut, class _Envs, class _Pred, class _CallEnv>
[[nodiscard]] _CCCL_HOST_API size_t __copy_if_into_generic(
  const _SIn& src, const _Envs& envs, _SOut&& dst, _Pred pred, const _CallEnv& call_env, const char* what)
{
  using count_type               = ::cuda::std::int64_t;
  const ::std::size_t num_shards = reserved::__shard_count(dst);
  if (reserved::__shard_count(src) != num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(what) + ": src/dst shard count mismatch");
  }
  if (reserved::__env_count(envs) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(what) + ": fewer environments than shards");
  }
  if (num_shards == 0)
  {
    return 0;
  }
  for (const auto g : each(num_shards))
  {
    if (static_cast<::std::size_t>(src.shard(g).size) > static_cast<::std::size_t>(dst.shard(g).capacity))
    {
      _CCCL_THROW(::std::invalid_argument,
                  ::std::string(what) + ": destination shard capacity smaller than source shard size");
    }
  }

  // Refusals first, before any CUDA call: size write-back synchronizes.
  require_sync_allowed(call_env, what);
  places::check_not_capturing(nullptr, what);
  for (const auto g : each(num_shards))
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), what);
  }

  // Entry probe: committing the current sizes is a no-op that throws exactly
  // when this model cannot mutate sizes, before any element moves.
  ::std::vector<size_t> new_sizes(num_shards);
  for (const auto g : each(num_shards))
  {
    new_sizes[g] = static_cast<size_t>(dst.shard(g).size);
  }
  dst.commit_sizes(new_sizes);

  constexpr bool __env_has_mr = ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
                             || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  count_type* h_new_sizes     = nullptr;
  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_new_sizes =
      static_cast<count_type*>(staging_mr.allocate_sync(num_shards * sizeof(count_type), alignof(count_type)));
  }
  else
  {
    h_new_sizes = static_cast<count_type*>(reserved::__pinned_staging(num_shards * sizeof(count_type)));
  }
  ::std::fill(h_new_sizes, h_new_sizes + num_shards, count_type{0});

  // Phase 0: acquire every fallible resource before the destination is
  // mutated; release guard armed first.
  using scratch_mr_t = ::cuda::std::remove_cvref_t<decltype(::cuda::mr::get_memory_resource(envs[::std::size_t{0}]))>;
  ::std::vector<::std::tuple<scratch_mr_t, count_type*, cudaStream_t>> d_counts;
  d_counts.reserve(num_shards);
  SCOPE(exit)
  {
    for (auto& [mr, ptr, stream] : d_counts)
    {
      mr.deallocate(::cuda::stream_ref{stream}, ptr, sizeof(count_type), alignof(count_type));
    }
  };
  for (const auto g : each(num_shards))
  {
    if (src.shard(g).size == 0)
    {
      continue;
    }
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    auto mr                               = ::cuda::mr::get_memory_resource(envs[g]);
    count_type* d_num = static_cast<count_type*>(mr.allocate(shard_stream, sizeof(count_type), alignof(count_type)));
    d_counts.emplace_back(mv(mr), d_num, shard_stream.get());
  }

  // Phase 1: select per shard into the destination's capacity. From here
  // through commit_sizes the destination's contents are indeterminate: on
  // any failure, leave it VALID and EMPTY.
  SCOPE(fail)
  {
    dst.commit_sizes(::std::vector<size_t>(num_shards, 0));
  };
  ::std::size_t next = 0;
  for (const auto g : each(num_shards))
  {
    const auto& si = src.shard(g);
    if (si.size == 0)
    {
      continue;
    }
    count_type* d_num                     = ::std::get<1>(d_counts[next++]);
    const ::cuda::stream_ref shard_stream = ::cuda::get_stream(envs[g]);
    stream_scope scope(shard_stream.get());
    cuda_safe_call(cub::DeviceSelect::If(
      si.data, dst.shard(g).data, d_num, static_cast<::cuda::std::int64_t>(si.size), pred, envs[g]));
    cuda_safe_call(
      cudaMemcpyAsync(&h_new_sizes[g], d_num, sizeof(count_type), cudaMemcpyDeviceToHost, shard_stream.get()));
  }
  barrier(envs);

  size_t total = 0;
  for (const auto g : each(num_shards))
  {
    _CCCL_ASSERT(h_new_sizes[g] >= 0 && static_cast<size_t>(h_new_sizes[g]) <= static_cast<size_t>(src.shard(g).size),
                 "sharded compaction: select returned an out-of-range count");
    new_sizes[g] = static_cast<size_t>(h_new_sizes[g]);
    total += new_sizes[g];
  }
  dst.commit_sizes(new_sizes);

  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    staging_mr.deallocate_sync(h_new_sizes, num_shards * sizeof(count_type), alignof(count_type));
  }
  return total;
}
} // namespace reserved

/**
 * @brief Keep only the elements satisfying @p pred, in place, over any
 * `owning_sharded` structure (kept elements compact to the front of each
 * shard, order preserved; shard sizes shrink and offsets re-tile through one
 * atomic `commit_sizes`). Returns the new total element count.
 * SYNCHRONOUS-only; refuses at entry under `sync_policy::forbid`, capture,
 * and on models whose sizes cannot be mutated (probed by committing the
 * current sizes before anything changes).
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_Pred>>))
[[nodiscard]] _CCCL_HOST_API size_t copy_if(_S&& data, const _Envs& envs, _Pred pred, const _CallEnv& call_env = {})
{
  return reserved::__copy_if_generic(::cuda::std::forward<_S>(data), envs, pred, call_env, "sharded::copy_if");
}

/// @brief Keep only the elements satisfying @p pred (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND self_bound<::cuda::std::remove_cvref_t<_S>>
                 _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Pred>>)
                   _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_Pred>>))
[[nodiscard]] _CCCL_HOST_API size_t copy_if(_S&& data, _Pred pred, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return reserved::__copy_if_generic(::cuda::std::forward<_S>(data), envs, pred, call_env, "sharded::copy_if");
}

/// @brief Alias of `copy_if` (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_Pred>>))
[[nodiscard]] _CCCL_HOST_API size_t filter(_S&& data, const _Envs& envs, _Pred pred, const _CallEnv& call_env = {})
{
  return reserved::__copy_if_generic(::cuda::std::forward<_S>(data), envs, pred, call_env, "sharded::filter");
}

/// @brief Alias of `copy_if` (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND self_bound<::cuda::std::remove_cvref_t<_S>>
                 _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Pred>>)
                   _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_Pred>>))
[[nodiscard]] _CCCL_HOST_API size_t filter(_S&& data, _Pred pred, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(data);
  return reserved::__copy_if_generic(::cuda::std::forward<_S>(data), envs, pred, call_env, "sharded::filter");
}

/// @brief Remove the elements satisfying @p pred (generic).
_CCCL_TEMPLATE(class _S, class _Envs, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>
    _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_Pred>>))
[[nodiscard]] _CCCL_HOST_API size_t remove_if(_S&& data, const _Envs& envs, _Pred pred, const _CallEnv& call_env = {})
{
  using elem_t = view_element_t<_S>;
  return reserved::__copy_if_generic(
    ::cuda::std::forward<_S>(data), envs, reserved::negate_pred_fn<elem_t, _Pred>{pred}, call_env, "sharded::remove_if");
}

/// @brief Remove the elements satisfying @p pred (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(owning_sharded<::cuda::std::remove_cvref_t<_S>> _CCCL_AND self_bound<::cuda::std::remove_cvref_t<_S>>
                 _CCCL_AND(!sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Pred>>)
                   _CCCL_AND(!sharded_view<::cuda::std::remove_cvref_t<_Pred>>))
[[nodiscard]] _CCCL_HOST_API size_t remove_if(_S&& data, _Pred pred, const _CallEnv& call_env = {})
{
  using elem_t    = view_element_t<_S>;
  const auto envs = default_envs(data);
  return reserved::__copy_if_generic(
    ::cuda::std::forward<_S>(data), envs, reserved::negate_pred_fn<elem_t, _Pred>{pred}, call_env, "sharded::remove_if");
}

/**
 * @brief Out-of-place selection: copy the elements of @p src satisfying
 * @p pred into @p dst, per shard (order preserved within each shard). The
 * source is untouched; the destination ends RAGGED — its per-shard sizes
 * become the data-dependent selected counts, committed through one atomic
 * `commit_sizes` (offsets re-tile, `validate()` holds). Returns the total
 * selected count.
 *
 * This is the frontier shape: derive a new, data-dependent-size structure
 * from a read-only view (e.g. the vertex ids whose property passes a
 * predicate) without destroying the source.
 *
 * Cross-space contract: @p src and @p dst have equal shard counts, and per
 * shard `dst.capacity >= src.size` (checked at entry; the destination's
 * prior sizes and contents are irrelevant). No co-partition requirement:
 * the destination's compacted index space is by nature different from the
 * source's. SYNCHRONOUS-only, like the whole compaction family; refuses at
 * entry under `sync_policy::forbid`, capture, and on destinations whose
 * sizes cannot be mutated.
 */
_CCCL_TEMPLATE(class _SIn, class _Envs, class _SOut, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>> _CCCL_AND
                   owning_sharded<::cuda::std::remove_cvref_t<_SOut>>)
[[nodiscard]] _CCCL_HOST_API size_t
copy_if(const _SIn& src, const _Envs& envs, _SOut&& dst, _Pred pred, const _CallEnv& call_env = {})
{
  return reserved::__copy_if_into_generic(
    src, envs, ::cuda::std::forward<_SOut>(dst), pred, call_env, "sharded::copy_if (out-of-place)");
}

/// @brief Out-of-place selection with environments derived from the
/// self-bound destination.
_CCCL_TEMPLATE(class _SIn, class _SOut, class _Pred, class _CallEnv = default_call_env)
_CCCL_REQUIRES(
  sharded_view<::cuda::std::remove_cvref_t<_SIn>> _CCCL_AND owning_sharded<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND
    self_bound<::cuda::std::remove_cvref_t<_SOut>> _CCCL_AND(
      !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_SOut>>))
[[nodiscard]] _CCCL_HOST_API size_t copy_if(const _SIn& src, _SOut&& dst, _Pred pred, const _CallEnv& call_env = {})
{
  const auto envs = default_envs(dst);
  return reserved::__copy_if_into_generic(
    src, envs, ::cuda::std::forward<_SOut>(dst), pred, call_env, "sharded::copy_if (out-of-place)");
}
} // namespace cuda::experimental::sharded
