// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_scan.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN
namespace detail::set_ops
{
//! The tuning policy for all algorithms in DeviceSetOps.
struct SetOpsPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory
  BlockScanAlgorithm scan_algorithm; //!< The @ref BlockScanAlgorithm used to compact the per-tile output

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const SetOpsPolicy& lhs, const SetOpsPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier && lhs.scan_algorithm == rhs.scan_algorithm;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const SetOpsPolicy& lhs, const SetOpsPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const SetOpsPolicy& p)
  {
    return os << "SetOpsPolicy { .threads_per_block = " << p.threads_per_block
              << ", .items_per_thread = " << p.items_per_thread << ", .load_modifier = " << p.load_modifier
              << ", .scan_algorithm = " << p.scan_algorithm << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept set_ops_policy_selector = policy_selector<T, SetOpsPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> SetOpsPolicy
  {
    // We don't use cub::detail::nominal_4B_items_to_items here because Thrust scaled the nominal item count with
    // a ceiling division, whereas nominal_4B_items_to_items rounds down.
    if (cc >= ::cuda::compute_capability{6, 0})
    {
      constexpr int nominal_items_per_thread = 19;
      return SetOpsPolicy{
        512,
        ::cuda::std::clamp(::cuda::ceil_div(nominal_items_per_thread * 4, key_size), 1, nominal_items_per_thread),
        LOAD_DEFAULT,
        BLOCK_SCAN_WARP_SCANS};
    }

    // default is SM52
    constexpr int nominal_items_per_thread = 15;
    return SetOpsPolicy{
      256,
      ::cuda::std::clamp(::cuda::ceil_div(nominal_items_per_thread * 4, key_size), 1, nominal_items_per_thread),
      LOAD_DEFAULT,
      BLOCK_SCAN_WARP_SCANS};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(set_ops_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeysIt1, typename ValuesIt1, typename KeysIt2, typename ValuesIt2, typename OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> SetOpsPolicy
  {
    using key_t = it_value_t<KeysIt1>;
    return policy_selector{int{sizeof(key_t)}}(cc);
  }
};
} // namespace detail::set_ops

CUB_NAMESPACE_END
