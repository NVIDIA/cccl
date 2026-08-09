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
 * @brief View type identifying a locality domain of a CUDA device
 */

#include <cuda/experimental/__stf/utility/hash.cuh>

namespace cuda::experimental::places
{
template <typename T>
struct hash;

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
