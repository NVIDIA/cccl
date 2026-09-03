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
 * @brief In-place scans over sharded arrays: each place runs the device-scope
 *        primitive (CUB `DeviceScan`) on its shard, then per-place totals are
 *        prefix-combined and folded back into the shards in place over the
 *        shared address space.
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

#include <cub/device/device_scan.cuh>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>

#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <cub/device/device_reduce.cuh>

#include <cuda/__functional/operator_properties.h> // identity_element

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/default_envs.cuh>
#include <cuda/experimental/__sharded/pinned_staging.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace cuda::experimental::sharded
{
/// @brief Scan flavor.
enum class scan_type
{
  inclusive, //!< output[i] = op(input[0], ..., input[i])
  exclusive //!< output[i] = op(init, input[0], ..., input[i-1])
};

namespace reserved
{
template <typename _Tp, typename _ScanOp>
struct apply_prefix_fn
{
  _Tp prefix;
  _ScanOp op;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE _Tp operator()(_Tp val) const
  {
    return op(prefix, val);
  }
};

template <typename _Tp, typename _ScanOp>
_CCCL_HOST_API void
scan_impl(place_group&, sharded_array<_Tp>& data, scan_type type, _ScanOp scan_op, _Tp init_value, _Tp identity)
{
  if (data.empty())
  {
    return;
  }

  // Host-side prefix combine + synchronization: cannot be recorded into a graph
  reserved::check_not_capturing(
    data, type == scan_type::inclusive ? "sharded::inclusive_scan" : "sharded::exclusive_scan");

  const size_t num_shards = data.num_shards();

  // Pinned host staging for shard totals / prefixes
  // Cached pinned arena: a per-call pinned allocation costs ~1 ms and would
  // dominate the O(P) cross-shard stage.
  const size_t host_bytes = 3 * num_shards * sizeof(_Tp);
  _Tp* h_shard_totals     = static_cast<_Tp*>(reserved::__pinned_staging(host_bytes));
  _Tp* h_prefixes         = h_shard_totals + num_shards;
  _Tp* h_last_elements    = h_prefixes + num_shards; // for exclusive scans

  // Empty shards contribute nothing to the cross-shard chain: track presence
  // explicitly instead of pre-filling with a value that would have to be the
  // operator's identity.
  ::std::vector<bool> has_total(num_shards, false);

  // Per-shard CUB temp storage, freed after the final sync
  ::std::vector<::std::tuple<places::place_memory_resource, void*, size_t>> temp_storage;
  temp_storage.reserve(num_shards);

  // ==========================================================================
  // Phase 1: local scan on each shard (empty shards are skipped by the
  // visitation). Exclusive scans run locally with the IDENTITY, never with
  // `init_value`: the init belongs to the global sequence exactly once and is
  // folded through the prefix chain in phase 3.
  // ==========================================================================

  data.each_shard->*[&](const size_t g, auto& s) {
    has_total[g] = true;

    // For exclusive scans, save the last element BEFORE it is overwritten:
    // the shard total is op(exclusive_last, original_last)
    if (type == scan_type::exclusive)
    {
      cuda_safe_call(
        cudaMemcpyAsync(&h_last_elements[g], s.data + s.size - 1, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
    }

    // Query CUB temp storage requirements
    size_t bytes = 0;
    if (type == scan_type::inclusive)
    {
      cuda_safe_call(cub::DeviceScan::InclusiveScan(nullptr, bytes, s.data, s.data, scan_op, s.size, s.stream));
    }
    else
    {
      cuda_safe_call(
        cub::DeviceScan::ExclusiveScan(nullptr, bytes, s.data, s.data, scan_op, identity, s.size, s.stream));
    }

    places::place_memory_resource mr(s.place);
    void* d_temp = mr.allocate(::cuda::stream_ref{s.stream}, bytes);
    temp_storage.emplace_back(mr, d_temp, bytes);

    // Run the local scan in place
    if (type == scan_type::inclusive)
    {
      cuda_safe_call(cub::DeviceScan::InclusiveScan(d_temp, bytes, s.data, s.data, scan_op, s.size, s.stream));
    }
    else
    {
      cuda_safe_call(
        cub::DeviceScan::ExclusiveScan(d_temp, bytes, s.data, s.data, scan_op, identity, s.size, s.stream));
    }

    // The last scanned element: for inclusive scans this IS the shard total;
    // for exclusive scans the total also needs the saved original last element
    cuda_safe_call(
      cudaMemcpyAsync(&h_shard_totals[g], s.data + s.size - 1, sizeof(_Tp), cudaMemcpyDeviceToHost, s.stream));
  };

  data.sync();

  if (type == scan_type::exclusive)
  {
    for (size_t g = 0; g < num_shards; g++)
    {
      if (has_total[g])
      {
        h_shard_totals[g] = scan_op(h_shard_totals[g], h_last_elements[g]);
      }
    }
  }

  // ==========================================================================
  // Phase 2: prefix-combine the shard totals on the host (P values).
  //
  // Inclusive: shard g's prefix is the scan_op-fold of the totals of the
  // preceding NON-EMPTY shards; the first non-empty shard has none. No
  // identity is needed.
  //
  // Exclusive: the chain is additionally seeded with `init_value`, folding
  // the init into the global sequence exactly once (the local scans used the
  // identity).
  // ==========================================================================

  ::std::vector<bool> has_prefix(num_shards, false);
  {
    bool running_defined = (type == scan_type::exclusive);
    _Tp running          = (type == scan_type::exclusive) ? init_value : _Tp{};
    for (size_t g = 0; g < num_shards; g++)
    {
      has_prefix[g] = running_defined;
      if (running_defined)
      {
        h_prefixes[g] = running;
      }
      if (has_total[g])
      {
        running         = running_defined ? scan_op(running, h_shard_totals[g]) : h_shard_totals[g];
        running_defined = true;
      }
    }
  }

  // ==========================================================================
  // Phase 3: fold each shard's prefix into the shard, in place. Exclusive
  // scans may skip an identity prefix (comparison against the caller-supplied
  // identity, not against the init).
  // ==========================================================================

  data.each_shard->*[&](const size_t g, auto& s) {
    if (!has_prefix[g])
    {
      return;
    }
    // No identity-prefix shortcut: it would impose an undeclared operator==
    // requirement on _Tp. The fold is a cheap elementwise pass; skipping it
    // is a pure optimization the value-type contract should not pay for.
    const _Tp prefix = h_prefixes[g];
    thrust::transform(
      thrust::cuda::par_nosync.on(s.stream),
      s.data,
      s.data + s.size,
      s.data,
      apply_prefix_fn<_Tp, _ScanOp>{prefix, scan_op});
    cuda_safe_call(cudaGetLastError());
  };

  data.sync();

  for (auto& [mr, ptr, bytes] : temp_storage)
  {
    mr.deallocate_sync(ptr, bytes);
  }
  // (arena staging is cached; nothing to release)
}
} // namespace reserved

/// @brief In-place inclusive scan with a custom operator.
///
/// No identity is required: the cross-shard prefixes are folds of the
/// preceding non-empty shards' totals, and the first non-empty shard has
/// none.
template <typename _Tp, typename _ScanOp>
_CCCL_HOST_API void inclusive_scan(place_group& group, sharded_array<_Tp>& data, _ScanOp scan_op)
{
  reserved::scan_impl<_Tp>(group, data, scan_type::inclusive, scan_op, _Tp{}, _Tp{});
}

/// @brief In-place inclusive sum.
template <typename _Tp>
_CCCL_HOST_API void inclusive_scan(place_group& group, sharded_array<_Tp>& data)
{
  reserved::scan_impl<_Tp>(group, data, scan_type::inclusive, ::cuda::std::plus<_Tp>{}, _Tp{0}, _Tp{0});
}

/// @brief In-place exclusive scan with a custom operator.
///
/// `init_value` and `identity` are DIFFERENT parameters: the init seeds the
/// global sequence exactly once (element 0 of the whole array scans to it);
/// the identity is `scan_op`'s neutral element, used to seed the per-shard
/// local scans. For `plus` they coincide only when the init is zero.
template <typename _Tp,
          typename _ScanOp,
          typename = ::cuda::std::enable_if_t<::cuda::std::is_invocable_v<_ScanOp, _Tp, _Tp>>>
_CCCL_HOST_API void
exclusive_scan(place_group& group, sharded_array<_Tp>& data, _ScanOp scan_op, _Tp init_value, _Tp identity)
{
  reserved::scan_impl<_Tp>(group, data, scan_type::exclusive, scan_op, init_value, identity);
}

/// @brief In-place exclusive sum (identity 0; `init_value` seeds the global
/// sequence exactly once).
template <typename _Tp>
_CCCL_HOST_API void exclusive_scan(place_group& group, sharded_array<_Tp>& data, _Tp init_value = _Tp{0})
{
  reserved::scan_impl<_Tp>(group, data, scan_type::exclusive, ::cuda::std::plus<_Tp>{}, init_value, _Tp{0});
}

/// @brief Alias for the inclusive sum.
template <typename _Tp>
_CCCL_HOST_API void inclusive_sum(place_group& group, sharded_array<_Tp>& data)
{
  inclusive_scan(group, data);
}

/// @brief Alias for the exclusive sum.
template <typename _Tp>
_CCCL_HOST_API void exclusive_sum(place_group& group, sharded_array<_Tp>& data, _Tp init_value = _Tp{0})
{
  exclusive_scan(group, data, init_value);
}

// ============================================================================
// Concept-generic tier: scans over any sharded_view (reduce-then-scan)
// ============================================================================

namespace reserved
{
//! @brief Shared generic scan implementation, REDUCE-THEN-SCAN skeleton
//! (measured 23% faster at GiB scale than scan-then-propagate, and the
//! shape whose cross-shard stage is a pure prefix over P totals):
//!
//! 1. per-shard totals via `cub::DeviceReduce` on each shard's environment
//!    (collected BEFORE any mutation — which is what allows the in-place
//!    seeded scans of phase 3);
//! 2. host prefix over the P totals (the cross-shard stage; staged through
//!    the call environment's resource or the pinned arena);
//! 3. per-shard seeded scans (`InclusiveScan[Init]` / `ExclusiveScan` with
//!    the shard's seed), temp storage stream-ordered from each shard's
//!    environment, launched through the shared map driver whose synchronous
//!    tail provides the final join.
//!
//! SYNCHRONOUS-ONLY in this form (the host prefix synchronizes mid-flight):
//! refuses at entry under `sync_policy::forbid` and under capture. The
//! asynchronous variant (device prefix over the P totals, seeds delivered
//! as `cub::FutureValue`) is the recorded follow-up.
//!
//! Exclusive semantics are the global ones: `out[i] = fold(init,
//! x_0..x_{i-1})` — init enters the fold exactly once.
template <bool _Inclusive, class _S, class _Envs, class _ScanOp, class _Tp, class _CallEnv>
_CCCL_HOST_API void __scan_generic(
  _S&& data, const _Envs& envs, _ScanOp scan_op, _Tp init_value, _Tp identity, const _CallEnv& call_env, const char* what)
{
  const ::std::size_t num_shards = static_cast<::std::size_t>(data.num_shards());
  if (static_cast<::std::size_t>(envs.size()) < num_shards)
  {
    _CCCL_THROW(::std::invalid_argument, ::std::string(what) + ": fewer environments than shards");
  }
  if (num_shards == 0)
  {
    return;
  }

  // Refusals first, before any CUDA call: the host prefix synchronizes.
  require_sync_allowed(call_env, what);
  places::check_not_capturing(nullptr, what);
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    places::check_not_capturing(::cuda::get_stream(envs[g]).get(), what);
  }

  // Per-shard totals staging (host-accessible; call-env resource override,
  // pinned arena default). Prefilled with the identity so empty shards
  // contribute nothing to the prefix.
  constexpr bool __env_has_mr =
    ::cuda::std::execution::__queryable_with<_CallEnv, ::cuda::mr::get_memory_resource_t>
    || ::cuda::mr::__has_member_get_resource<_CallEnv>;
  _Tp* h_totals = nullptr;
  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    h_totals        = static_cast<_Tp*>(staging_mr.allocate_sync(num_shards * sizeof(_Tp), alignof(_Tp)));
  }
  else
  {
    h_totals = static_cast<_Tp*>(reserved::__pinned_staging(num_shards * sizeof(_Tp)));
  }
  ::std::fill(h_totals, h_totals + num_shards, identity);

  // Phase 1: per-shard totals, collected before any element is mutated.
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
    auto mr      = ::cuda::mr::get_memory_resource(env);
    _Tp* d_total = static_cast<_Tp*>(mr.allocate(shard_stream, sizeof(_Tp), alignof(_Tp)));
    cuda_safe_call(cub::DeviceReduce::Reduce(s.data, d_total, s.size, scan_op, identity, env));
    cuda_safe_call(cudaMemcpyAsync(&h_totals[g], d_total, sizeof(_Tp), cudaMemcpyDeviceToHost, shard_stream.get()));
    mr.deallocate(shard_stream, d_total, sizeof(_Tp), alignof(_Tp)); // stream-ordered, after the copy
  }
  for (::std::size_t g = 0; g < num_shards; g++)
  {
    if (data.shard(g).size != 0)
    {
      cuda_safe_call(cudaStreamSynchronize(::cuda::get_stream(envs[g]).get()));
    }
  }

  // Phase 2: host prefix — the seed of shard g is the fold of everything
  // before it (plus init, exactly once, for the exclusive form).
  ::std::vector<_Tp> seed(num_shards, identity);
  ::std::vector<bool> has_seed(num_shards, false);
  {
    _Tp running    = init_value; // meaningful for the exclusive form only
    bool have_prev = false;
    for (::std::size_t g = 0; g < num_shards; g++)
    {
      if constexpr (_Inclusive)
      {
        seed[g]     = running;
        has_seed[g] = have_prev;
      }
      else
      {
        seed[g]     = running;
        has_seed[g] = true; // exclusive always seeds (init on the first shard)
      }
      if (data.shard(g).size != 0)
      {
        running   = have_prev || !_Inclusive ? scan_op(running, h_totals[g]) : h_totals[g];
        have_prev = true;
      }
    }
  }

  // Phase 3: per-shard in-place seeded scans through the shared driver
  // (its synchronous tail provides this form's final join).
  __detail::__generic_map(
    data, envs, call_env, what, [&](::std::size_t g, const auto& d, cudaStream_t s) {
      const auto& env = envs[g];
      auto mr         = ::cuda::mr::get_memory_resource(env);

      auto run_two_call = [&](auto&& launch) {
        ::std::size_t temp_bytes = 0;
        launch(nullptr, temp_bytes);
        void* d_temp = mr.allocate(::cuda::stream_ref{s}, temp_bytes, alignof(::std::max_align_t));
        launch(d_temp, temp_bytes);
        mr.deallocate(::cuda::stream_ref{s}, d_temp, temp_bytes, alignof(::std::max_align_t));
      };

      if constexpr (_Inclusive)
      {
        if (has_seed[g])
        {
          run_two_call([&](void* t, ::std::size_t& b) {
            cuda_safe_call(cub::DeviceScan::InclusiveScanInit(
              t, b, d.data, d.data, scan_op, seed[g], static_cast<::cuda::std::int64_t>(d.size), s));
          });
        }
        else
        {
          run_two_call([&](void* t, ::std::size_t& b) {
            cuda_safe_call(cub::DeviceScan::InclusiveScan(
              t, b, d.data, d.data, scan_op, static_cast<::cuda::std::int64_t>(d.size), s));
          });
        }
      }
      else
      {
        run_two_call([&](void* t, ::std::size_t& b) {
          cuda_safe_call(cub::DeviceScan::ExclusiveScan(
            t, b, d.data, d.data, scan_op, seed[g], static_cast<::cuda::std::int64_t>(d.size), s));
        });
      }
    });

  if constexpr (__env_has_mr)
  {
    auto staging_mr = ::cuda::mr::get_memory_resource(call_env);
    staging_mr.deallocate_sync(h_totals, num_shards * sizeof(_Tp), alignof(_Tp));
  }
  // (arena staging is cached; nothing to release)
}
} // namespace reserved

/**
 * @brief In-place inclusive scan over any `sharded_view`:
 * `data[i] = fold(data[0..i])` across the global index space.
 *
 * Reduce-then-scan skeleton; synchronous-only (see the implementation note).
 * @p identity is the operator's identity element, defaulted where
 * `cuda::identity_element` knows the operator; custom operators supply it.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__scan_generic<true>(
    ::cuda::std::forward<_S>(data), envs, scan_op, identity, identity, call_env, "sharded::inclusive_scan");
}

/// @brief In-place inclusive scan (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void inclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__scan_generic<true>(
    ::cuda::std::forward<_S>(data), envs, scan_op, identity, identity, call_env, "sharded::inclusive_scan");
}

/**
 * @brief In-place exclusive scan over any `sharded_view`:
 * `data[i] = fold(init, data[0..i-1])` across the global index space — the
 * global semantics, init entering the fold exactly once.
 */
_CCCL_TEMPLATE(class _S, class _Envs, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(sharded_view<::cuda::std::remove_cvref_t<_S>> _CCCL_AND
                 sharded_alloc_env_range<::cuda::std::remove_cvref_t<_Envs>>)
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  const _Envs& envs,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  reserved::__scan_generic<false>(
    ::cuda::std::forward<_S>(data), envs, scan_op, init_value, identity, call_env, "sharded::exclusive_scan");
}

/// @brief In-place exclusive scan (generic, self-bound).
_CCCL_TEMPLATE(class _S, class _ScanOp, class _CallEnv = default_call_env)
_CCCL_REQUIRES(self_bound<::cuda::std::remove_cvref_t<_S>> _CCCL_AND(
  !sharded_alloc_env_range<::cuda::std::remove_cvref_t<_ScanOp>>))
_CCCL_HOST_API void exclusive_scan(
  _S&& data,
  _ScanOp scan_op,
  view_element_t<_S> init_value,
  view_element_t<_S> identity = ::cuda::identity_element<_ScanOp, view_element_t<_S>>(),
  const _CallEnv& call_env    = {})
{
  const auto envs = default_envs(data);
  reserved::__scan_generic<false>(
    ::cuda::std::forward<_S>(data), envs, scan_op, init_value, identity, call_env, "sharded::exclusive_scan");
}

} // namespace cuda::experimental::sharded
