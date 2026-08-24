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

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>

#include <algorithm>
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
  places::place_memory_resource host_mr(data_place::host());
  const size_t host_bytes = 3 * num_shards * sizeof(_Tp);
  _Tp* h_shard_totals     = static_cast<_Tp*>(host_mr.allocate_sync(host_bytes, alignof(_Tp)));
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
  host_mr.deallocate_sync(h_shard_totals, host_bytes, alignof(_Tp));
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
} // namespace cuda::experimental::sharded
