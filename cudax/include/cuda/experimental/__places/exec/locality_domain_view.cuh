//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/**
 * @file
 * @brief View type identifying a locality domain of a CUDA device, and the
 * SM split methods available when building an execution place for one
 */

#include <cuda/experimental/__stf/utility/hash.cuh>

namespace cuda::experimental::places
{
template <typename T>
struct hash;

/**
 * @brief SM split methods for locality-domain execution places
 *
 * Selects how the SM partition backing `exec_place::locality_domain` is
 * carved out of the device (via `cuDevSmResourceSplit`). The methods trade
 * SM coverage against SM/memory affinity and thread-block cluster support:
 *
 * - `backfill` (the default): every domain place is sized to an even share
 *   of the device total (`CU_DEV_SM_RESOURCE_GROUP_BACKFILL`). The driver
 *   fills each group with the target domain's SMs first, then SMs not
 *   assigned to any domain, then SMs from other domains, so together the
 *   domain places cover the whole device. The backfilled SMs may sit
 *   outside the place's domain (no memory affinity with it), and the
 *   groups use the finest co-scheduling granularity, which does not
 *   support launching thread-block clusters.
 * - `aligned`: only SMs of the domain that form complete co-scheduled
 *   groups at the device's default alignment (`smCoscheduledAlignment`).
 *   Every SM of the place is affine to the place's domain and thread-block
 *   cluster launches remain available, but domain SMs that do not fill a
 *   complete aligned group -- and SMs outside any domain -- are left out of
 *   the partition.
 * - `fine`: every SM attributed to the domain, grouped at the finest
 *   co-scheduling granularity (groups of 2). Every SM of the place is
 *   affine to the place's domain, at the cost of thread-block cluster
 *   launches; SMs outside any domain are left out.
 *
 * `backfill` is the least surprising default: work spread over the domain
 * places uses the whole device. When per-place SM/memory affinity matters
 * more than whole-device coverage (e.g. work partitioned by data
 * affinity), prefer the strictly per-domain `aligned` or `fine` methods.
 *
 * The split method only affects the execution side. Data places have no
 * split method, and places built with different methods for the same
 * (device, domain) share the same (equal) affine data place. On backends
 * without native locality-domain support (pre-13.4 toolkits, the
 * whole-device degrade, or the fake-topology override) the method is
 * accepted and ignored.
 *
 * Toolkit requirements: every method splits by
 * `CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID`, the CUDA 13.4 flag the
 * native backend is already gated on (`_CCCL_CTK_AT_LEAST(13, 4)`), and
 * the pieces individual methods add on top of it are older
 * (`CU_DEV_SM_RESOURCE_GROUP_BACKFILL` and `coscheduledSmCount` date back
 * to the CUDA 13.1 `cuDevSmResourceSplit`), so today the single 13.4 gate
 * covers every value. A method added in the future may sit behind a higher
 * toolkit gate of its own.
 */
enum class locality_domain_sm_split : unsigned int
{
  backfill = 0, ///< even share of the device, backfilled to full coverage (default)
  aligned  = 1, ///< only complete co-scheduled groups of the domain
  fine     = 2, ///< all of the domain's SMs, finest co-scheduling granularity
};

/**
 * @brief Identifies one locality domain of a CUDA device
 *
 * Some devices partition their multiprocessors and memory into locality
 * domains (see `CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT`). A
 * `locality_domain_view` is a plain (device ordinal, domain ordinal) identity
 * token: constructing one performs no existence check, mirroring
 * `data_place::device(i)` which likewise does not validate the ordinal. The
 * ordinals are validated lazily, when the view is turned into a place that is
 * actually used.
 */
class locality_domain_view
{
public:
  locality_domain_view(int devid, int domain_id)
      : devid(devid)
      , domain_id(domain_id)
  {}

  int devid;
  int domain_id;

  bool operator==(const locality_domain_view& other) const
  {
    return (devid == other.devid) && (domain_id == other.domain_id);
  }

  bool operator!=(const locality_domain_view& other) const
  {
    return !(*this == other);
  }

  bool operator<(const locality_domain_view& other) const
  {
    if (devid != other.devid)
    {
      return devid < other.devid;
    }
    return domain_id < other.domain_id;
  }
};

/**
 * @brief Specialization of `places::hash` for `locality_domain_view`
 */
template <>
struct hash<locality_domain_view>
{
  ::std::size_t operator()(const locality_domain_view& k) const
  {
    return ::cuda::experimental::stf::hash_all(k.devid, k.domain_id);
  }
};
} // end namespace cuda::experimental::places
