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

#include <cub/agent/agent_segmented_scan.cuh>
#include <cub/agent/agent_thread_segmented_scan.cuh>
#include <cub/agent/agent_warp_segmented_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/void_t.h>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
struct block_segmented_scan_policy
{
  int block_threads;
  int items_per_thread;
  CacheLoadModifier load_modifier;
  BlockLoadAlgorithm load_algorithm;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int max_segments_per_block;

  CUB_RUNTIME_FUNCTION constexpr int BlockThreads() const
  {
    return block_threads;
  }

  CUB_RUNTIME_FUNCTION constexpr int ItemsPerThread() const
  {
    return items_per_thread;
  }

  CUB_RUNTIME_FUNCTION constexpr int WorkersPerBlock() const
  {
    return 1;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const block_segmented_scan_policy& lhs, const block_segmented_scan_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier && lhs.load_algorithm == rhs.load_algorithm
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.max_segments_per_block == rhs.max_segments_per_block;
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
        << ", .items_per_thread = " << policy.items_per_thread << ", .load_modifier = " << policy.load_modifier
        << ", .load_algorithm = " << policy.load_algorithm << ", .store_algorithm = " << policy.store_algorithm
        << ", .scan_algorithm = " << policy.scan_algorithm
        << ", .max_segments_per_block = " << policy.max_segments_per_block << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct warp_segmented_scan_policy
{
  int block_threads;
  int items_per_thread;
  CacheLoadModifier load_modifier;
  WarpLoadAlgorithm load_algorithm;
  WarpStoreAlgorithm store_algorithm;
  int max_segments_per_warp;

  CUB_RUNTIME_FUNCTION constexpr int BlockThreads() const
  {
    return block_threads;
  }

  CUB_RUNTIME_FUNCTION constexpr int ItemsPerThreads() const
  {
    return items_per_thread;
  }

  CUB_RUNTIME_FUNCTION constexpr int WorkersPerBlock() const
  {
    _CCCL_ASSERT(0 == (int(block_threads) % cub::detail::warp_threads), "Block size must be divisible by warp size");
    return (int(block_threads) >> cub::detail::log2_warp_threads);
  }

  CUB_RUNTIME_FUNCTION constexpr int MaxSegmentsPerWarp() const
  {
    return max_segments_per_warp;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const warp_segmented_scan_policy& lhs, const warp_segmented_scan_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier && lhs.load_algorithm == rhs.load_algorithm
        && lhs.store_algorithm == rhs.store_algorithm && lhs.max_segments_per_warp == rhs.max_segments_per_warp;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const warp_segmented_scan_policy& lhs, const warp_segmented_scan_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const warp_segmented_scan_policy& policy)
  {
    return os
        << "warp_segmented_scan_policy { .block_threads = " << policy.block_threads
        << ", .items_per_thread = " << policy.items_per_thread << ", .load_modifier = " << policy.load_modifier
        << ", .load_algorithm = " << policy.load_algorithm << ", .store_algorithm = " << policy.store_algorithm
        << ", .max_segments_per_warp = " << policy.max_segments_per_warp << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct thread_segmented_scan_policy
{
  int block_threads;
  int items_per_thread;
  CacheLoadModifier load_modifier;

  CUB_RUNTIME_FUNCTION constexpr int BlockThreads() const
  {
    return block_threads;
  }

  CUB_RUNTIME_FUNCTION constexpr int ItemsPerThread() const
  {
    return items_per_thread;
  }

  CUB_RUNTIME_FUNCTION constexpr int WorkersPerBlock() const
  {
    return block_threads;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const thread_segmented_scan_policy& lhs, const thread_segmented_scan_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_modifier == rhs.load_modifier;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const thread_segmented_scan_policy& lhs, const thread_segmented_scan_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const thread_segmented_scan_policy& policy)
  {
    return os
        << "thread_segmented_scan_policy { .block_threads = " << policy.block_threads
        << ", .items_per_thread = " << policy.items_per_thread << ", .load_modifier = " << policy.load_modifier << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct segmented_scan_policy
{
  block_segmented_scan_policy block;
  warp_segmented_scan_policy warp;
  thread_segmented_scan_policy thread;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const segmented_scan_policy& lhs, const segmented_scan_policy& rhs)
  {
    return lhs.block == rhs.block && lhs.warp == rhs.warp && lhs.thread == rhs.thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const segmented_scan_policy& lhs, const segmented_scan_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const segmented_scan_policy& policy)
  {
    return os << "segmented_scan_policy { .block = " << policy.block << ", .warp = " << policy.warp
              << ", .thread = " << policy.thread << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)

  _CCCL_API constexpr void CheckLoadModifier() const
  {
    _CCCL_ASSERT(
      (block.load_modifier != CacheLoadModifier::LOAD_LDG) && (warp.load_modifier != CacheLoadModifier::LOAD_LDG)
        && (thread.load_modifier != CacheLoadModifier::LOAD_LDG),
      "The memory consistency model does not apply to texture accesses");
  }
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept segmented_scan_policy_selector = policy_selector<T, segmented_scan_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  int accum_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id) const -> segmented_scan_policy
  {
    const bool large_values = accum_size > 128;
    const auto scan_transposed_blockload =
      large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
    const auto scan_transposed_blockstore =
      large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

    constexpr int nominal_block_threads    = 128;
    constexpr int nominal_items_per_thread = 9;
    constexpr int max_segments_per_block   = 512;
    constexpr int max_segments_per_warp    = 128;

    const int align          = accum_size;
    const int augmented_size = ((accum_size + 1 + align - 1) / align) * align;

    const auto block_scaled  = detail::scale_mem_bound(nominal_block_threads, nominal_items_per_thread, augmented_size);
    const auto warp_scaled   = detail::scale_mem_bound(nominal_block_threads, nominal_items_per_thread, augmented_size);
    const auto thread_scaled = detail::scale_mem_bound(nominal_block_threads, nominal_items_per_thread, accum_size);

    return segmented_scan_policy{
      block_segmented_scan_policy{
        block_scaled.block_threads,
        block_scaled.items_per_thread,
        LOAD_DEFAULT,
        scan_transposed_blockload,
        scan_transposed_blockstore,
        BLOCK_SCAN_WARP_SCANS,
        max_segments_per_block},
      warp_segmented_scan_policy{
        warp_scaled.block_threads,
        warp_scaled.items_per_thread,
        LOAD_DEFAULT,
        WARP_LOAD_TRANSPOSE,
        WARP_STORE_TRANSPOSE,
        max_segments_per_warp},
      thread_segmented_scan_policy{thread_scaled.block_threads, thread_scaled.items_per_thread, LOAD_DEFAULT}};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(segmented_scan_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

// stateless version which can be passed to kernels
template <typename AccumT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id /* arch */) const -> segmented_scan_policy
  {
    constexpr bool large_values = sizeof(AccumT) > 128;
    constexpr auto scan_transposed_blockload =
      large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
    constexpr auto scan_transposed_blockstore =
      large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

    constexpr int nominal_block_threads    = 128;
    constexpr int nominal_items_per_thread = 9;
    constexpr int max_segments_per_block   = 512;
    constexpr int max_segments_per_warp    = 128;

    // Actual ComputeT is AccumT for single segment per worker,
    // but becomes tuple<AccumT, bool> for multi-segment per worker
    using block_compute_t = agent_block_segmented_scan_compute_t<AccumT, max_segments_per_block>;
    using warp_compute_t  = agent_warp_segmented_scan_compute_t<AccumT, max_segments_per_warp>;

    constexpr auto block_scaled =
      detail::scale_mem_bound(nominal_block_threads, nominal_items_per_thread, int{sizeof(block_compute_t)});
    constexpr auto warp_scaled =
      detail::scale_mem_bound(nominal_block_threads, nominal_items_per_thread, int{sizeof(warp_compute_t)});
    constexpr auto thread_scaled =
      detail::scale_mem_bound(nominal_block_threads, nominal_items_per_thread, int{sizeof(AccumT)});

    return segmented_scan_policy{
      block_segmented_scan_policy{
        block_scaled.block_threads,
        block_scaled.items_per_thread,
        LOAD_DEFAULT,
        scan_transposed_blockload,
        scan_transposed_blockstore,
        BLOCK_SCAN_WARP_SCANS,
        max_segments_per_block},
      warp_segmented_scan_policy{
        warp_scaled.block_threads,
        warp_scaled.items_per_thread,
        LOAD_DEFAULT,
        WARP_LOAD_TRANSPOSE,
        WARP_STORE_TRANSPOSE,
        max_segments_per_warp},
      thread_segmented_scan_policy{thread_scaled.block_threads, thread_scaled.items_per_thread, LOAD_DEFAULT}};
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
