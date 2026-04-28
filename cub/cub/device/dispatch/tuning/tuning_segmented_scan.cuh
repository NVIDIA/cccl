// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/round_up.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/void_t.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
struct block_segmented_scan_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int max_segments;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const block_segmented_scan_policy& lhs, const block_segmented_scan_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.max_segments == rhs.max_segments;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const block_segmented_scan_policy& lhs, const block_segmented_scan_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const block_segmented_scan_policy& policy)
  {
    return os
        << "block_segmented_scan_policy { .block_threads = " << policy.block_threads
        << ", .items_per_thread = " << policy.items_per_thread << ", .load_algorithm = " << policy.load_algorithm
        << ", .load_modifier = " << policy.load_modifier << ", .store_algorithm = " << policy.store_algorithm
        << ", .scan_algorithm = " << policy.scan_algorithm << ", .max_segments_per_block = " << policy.max_segments
        << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct segmented_scan_policy
{
  block_segmented_scan_policy block;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const segmented_scan_policy& lhs, const segmented_scan_policy& rhs)
  {
    return lhs.block == rhs.block;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const segmented_scan_policy& lhs, const segmented_scan_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_scan_policy& policy)
  {
    return os << "segmented_scan_policy { .block = " << policy.block << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_scan_policy_selector = policy_selector<T, segmented_scan_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  // size and alignment of accumulator type, that would be used by non-segmented scan
  int accum_size;
  int accum_align;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id) const -> segmented_scan_policy
  {
    constexpr int nominal_block_threads    = 128;
    constexpr int nominal_items_per_thread = 9;
    constexpr int max_segments_per_block   = 512;

    _CCCL_ASSERT(accum_size > 0, "Accumulator size must be positive");
    _CCCL_ASSERT(accum_align > 0, "Accumulator alignment must be positive");
    _CCCL_ASSERT((accum_size % accum_align) == 0, "Size and alignment are not consistent");

    // multi-segment block- granularity agents use tuple<AccumT, bool>, single segment use AccumT
    // deduce its size here.
    const int augmented_size_block =
      ::cuda::round_up(accum_size + ((max_segments_per_block == 1) ? 0 : 1), accum_align);

    const auto block_scaled = scale_mem_bound(nominal_block_threads, nominal_items_per_thread, augmented_size_block);

    const bool large_values = augmented_size_block > 128;
    const auto scan_transposed_blockload =
      large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
    const auto scan_transposed_blockstore =
      large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

    return segmented_scan_policy{block_segmented_scan_policy{
      block_scaled.block_threads,
      block_scaled.items_per_thread,
      scan_transposed_blockload,
      LOAD_DEFAULT,
      scan_transposed_blockstore,
      BLOCK_SCAN_WARP_SCANS,
      max_segments_per_block}};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_scan_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename AccumT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> segmented_scan_policy
  {
    constexpr auto accum_size  = static_cast<int>(sizeof(AccumT));
    constexpr auto accum_align = static_cast<int>(alignof(AccumT));
    return policy_selector{accum_size, accum_align}(arch);
  };
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
