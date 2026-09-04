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
 * @brief Global sort over a sharded array, shared-address-space engine.
 *
 * Each shard ends holding the slice of the globally sorted sequence at its
 * ORIGINAL boundaries — sizes, offsets and capacities unchanged by
 * construction, so a contiguous (`allocate_contiguous`) array reads as ONE
 * globally sorted array afterwards, and any co-partitioned sibling stays
 * co-partitioned. The engine does local per-shard sorts, finds exact
 * splitters by multi-sequence selection, and fuses a gather-merge that
 * writes each destination's slice straight into its own shard storage,
 * loading across shard boundaries through the one address space the places
 * share (locality domains of one device, or the device itself).
 *
 * Scope. This is the PLACES-rung engine: it requires every shard on
 * device-backed places of ONE device (checked by `one_shared_address_space`;
 * otherwise it refuses with a diagnostic). The cross-address-space
 * (communicator/MGMN) engine is a separate change that arrives with the
 * multi-node seam and is deliberately not wired here — this header pulls in
 * nothing from the `__multi_gpu` layer.
 *
 * On the `place_group` parameter. `sort` is the one algorithm where the
 * group is genuinely load-bearing (the engine drives per-place streams for
 * the local sorts and the gather-merge), unlike the map/reduce family whose
 * bodies used no group state — so it keeps the group parameter by design,
 * not by legacy.
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

#include <cuda/std/functional>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__sharded/sharded_array.cuh>
#include <cuda/experimental/__sharded/sort_shared_va.cuh>

#include <stdexcept>

namespace cuda::experimental::sharded
{
/**
 * @brief Sort a sharded array in place into globally ascending order (by
 * @p comp), each shard keeping its original boundaries.
 *
 * SYNCHRONOUS (drains the shard streams before returning, like the other
 * combine-bearing sharded algorithms). Requires every shard on
 * device-backed places of one device; throws `std::invalid_argument`
 * otherwise (the cross-address-space engine is a separate change).
 *
 * @throws std::invalid_argument when the array does not have one shard per
 *         group place, or when the shards do not share one device's address
 *         space.
 */
template <typename _Tp, typename _Compare = ::cuda::std::less<_Tp>>
void sort(place_group& group, sharded_array<_Tp>& data, _Compare comp = {})
{
  if (data.size() == 0)
  {
    return;
  }

  check_places(data, group, "sharded::sort");

  if (!reserved::one_shared_address_space(data))
  {
    _CCCL_THROW(::std::invalid_argument,
                "sharded::sort: this engine requires every shard on device-backed places of one device "
                "(a single device's locality domains, green contexts, or the device itself); sorting across "
                "separate address spaces is a distinct engine that arrives with the multi-node seam");
  }

  reserved::sort_shared_va(group, data, comp);
}
} // namespace cuda::experimental::sharded
