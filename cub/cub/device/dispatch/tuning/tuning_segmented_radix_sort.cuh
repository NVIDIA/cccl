// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/tuning/tuning_radix_sort.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN
namespace detail::segmented_radix_sort
{
using radix_sort::make_reg_scaled_radix_sort_downsweep_policy;
using radix_sort::radix_sort_downsweep_policy;

struct segmented_radix_sort_policy
{
  radix_sort_downsweep_policy segmented;
  radix_sort_downsweep_policy alt_segmented;

  _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const segmented_radix_sort_policy& lhs, const segmented_radix_sort_policy& rhs)
  {
    return lhs.segmented == rhs.segmented && lhs.alt_segmented == rhs.alt_segmented;
  }

  _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const segmented_radix_sort_policy& lhs, const segmented_radix_sort_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_radix_sort_policy& p)
  {
    return os << "segmented_radix_sort_policy { .segmented = " << p.segmented
              << ", .alt_segmented = " << p.alt_segmented << " }";
  }
#endif // _CCCL_HOSTED()
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_radix_sort_policy_selector = detail::policy_selector<T, segmented_radix_sort_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int key_size;
  int value_size; // when 0, indicates keys-only

  // Dominant-sized key/value type
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int __dominant_size() const
  {
    return ::cuda::std::max(value_size, key_size);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> segmented_radix_sort_policy
  {
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      const int segmented_radix_bits = (key_size > 1) ? 6 : 5;

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return segmented_radix_sort_policy{segmented, alt_segmented};
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      const int segmented_radix_bits = (key_size > 1) ? 6 : 5;

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return segmented_radix_sort_policy{segmented, alt_segmented};
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      const int segmented_radix_bits = (key_size > 1) ? 6 : 5;

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return segmented_radix_sort_policy{segmented, alt_segmented};
    }

    if (cc >= ::cuda::compute_capability{7, 0})
    {
      const int segmented_radix_bits = (key_size > 1) ? 6 : 5;

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return segmented_radix_sort_policy{segmented, alt_segmented};
    }

    if (cc >= ::cuda::compute_capability{6, 2})
    {
      // SM62: segmented policies match the downsweep policies
      const int primary_radix_bits = 5;
      const int alt_radix_bits     = primary_radix_bits - 1;

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        16,
        __dominant_size(),
        primary_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        256,
        16,
        __dominant_size(),
        alt_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_RAKING_MEMOIZE);

      return segmented_radix_sort_policy{segmented, alt_segmented};
    }

    if (cc >= ::cuda::compute_capability{6, 1})
    {
      const int segmented_radix_bits = (key_size > 1) ? 6 : 5;

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return segmented_radix_sort_policy{segmented, alt_segmented};
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      const int segmented_radix_bits = (key_size > 1) ? 6 : 5;

      const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
        192,
        39,
        __dominant_size(),
        segmented_radix_bits,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
        384,
        11,
        __dominant_size(),
        segmented_radix_bits - 1,
        BLOCK_LOAD_TRANSPOSE,
        LOAD_DEFAULT,
        RADIX_RANK_MEMOIZE,
        BLOCK_SCAN_WARP_SCANS);

      return segmented_radix_sort_policy{segmented, alt_segmented};
    }

    // SM50
    const int segmented_radix_bits = (key_size > 1) ? 6 : 5;

    const auto segmented = make_reg_scaled_radix_sort_downsweep_policy(
      192,
      31,
      __dominant_size(),
      segmented_radix_bits,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    const auto alt_segmented = make_reg_scaled_radix_sort_downsweep_policy(
      256,
      11,
      __dominant_size(),
      segmented_radix_bits - 1,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE,
      BLOCK_SCAN_WARP_SCANS);

    return segmented_radix_sort_policy{segmented, alt_segmented};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_radix_sort_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <typename KeyT, typename ValueT, typename OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(cuda::compute_capability cc) const
    -> segmented_radix_sort_policy
  {
    constexpr auto policies =
      policy_selector{int{sizeof(KeyT)}, ::cuda::std::is_same_v<ValueT, NullType> ? 0 : int{sizeof(ValueT)}};
    return policies(cc);
  }
};
} // namespace detail::segmented_radix_sort

CUB_NAMESPACE_END
