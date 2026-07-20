// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_batch_memcpy.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/common.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

//! The small buffer sub-policy for @ref BatchedCopyPolicy.
struct BatchedCopySmallBufferPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int buffers_per_thread; //!< Number of buffers processed per thread
  int bytes_per_thread; //!< The number of bytes that each thread will work on with each iteration of reading in
                        //!< bytes from one or more source-buffers and writing them out to the respective
                        //!< destination-buffers.
  bool prefer_pow2_bits; //!< Whether the bit_packed_counter should prefer allocating a power-of-2 number of bits per
                         //!< counter
  int block_level_tile_size; //!< Tile size granularity for block-level buffers
  int warp_level_threshold; //!< Buffer size threshold above which warp-level collaboration is used
  int block_level_threshold; //!< Buffer size threshold above which block-level collaboration is used
  LookbackDelayPolicy buffer_lookback_delay; //!< The @ref LookbackDelayPolicy for the buffer offset scan
  LookbackDelayPolicy block_lookback_delay; //!< The @ref LookbackDelayPolicy for the block offset scan

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const BatchedCopySmallBufferPolicy& lhs, const BatchedCopySmallBufferPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.buffers_per_thread == rhs.buffers_per_thread
        && lhs.bytes_per_thread == rhs.bytes_per_thread && lhs.prefer_pow2_bits == rhs.prefer_pow2_bits
        && lhs.block_level_tile_size == rhs.block_level_tile_size
        && lhs.warp_level_threshold == rhs.warp_level_threshold
        && lhs.block_level_threshold == rhs.block_level_threshold
        && lhs.buffer_lookback_delay == rhs.buffer_lookback_delay
        && lhs.block_lookback_delay == rhs.block_lookback_delay;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const BatchedCopySmallBufferPolicy& lhs, const BatchedCopySmallBufferPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const BatchedCopySmallBufferPolicy& policy)
  {
    return os
        << "BatchedCopySmallBufferPolicy { .threads_per_block = " << policy.threads_per_block
        << ", .buffers_per_thread = " << policy.buffers_per_thread
        << ", .bytes_per_thread = " << policy.bytes_per_thread << ", .prefer_pow2_bits = " << policy.prefer_pow2_bits
        << ", .block_level_tile_size = " << policy.block_level_tile_size << ", .warp_level_threshold = "
        << policy.warp_level_threshold << ", .block_level_threshold = " << policy.block_level_threshold
        << ", .buffer_lookback_delay = " << policy.buffer_lookback_delay
        << ", .block_lookback_delay = " << policy.block_lookback_delay << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The large buffer sub-policy for @ref BatchedCopyPolicy.
struct BatchedCopyLargeBufferPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int bytes_per_thread; //!< Number of bytes processed per thread

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const BatchedCopyLargeBufferPolicy& lhs, const BatchedCopyLargeBufferPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.bytes_per_thread == rhs.bytes_per_thread;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const BatchedCopyLargeBufferPolicy& lhs, const BatchedCopyLargeBufferPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const BatchedCopyLargeBufferPolicy& policy)
  {
    return os << "BatchedCopyLargeBufferPolicy { .threads_per_block = " << policy.threads_per_block
              << ", .bytes_per_thread = " << policy.bytes_per_thread << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for all algorithms in @ref DeviceMemcpy.
struct BatchedCopyPolicy
{
  BatchedCopySmallBufferPolicy small_buffer; //!< Sub-policy for small buffers copied by a single thread block
  BatchedCopyLargeBufferPolicy large_buffer; //!< Sub-policy for large buffers requiring multi-block collaboration

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator==(const BatchedCopyPolicy& lhs, const BatchedCopyPolicy& rhs) noexcept
  {
    return lhs.small_buffer == rhs.small_buffer && lhs.large_buffer == rhs.large_buffer;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr bool
  operator!=(const BatchedCopyPolicy& lhs, const BatchedCopyPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const BatchedCopyPolicy& policy)
  {
    return os << "BatchedCopyPolicy { .small_buffer = " << policy.small_buffer
              << ", .large_buffer = " << policy.large_buffer << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::batch_memcpy
{
#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept batch_memcpy_policy_selector = policy_selector<T, BatchedCopyPolicy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> BatchedCopyPolicy
  {
    const auto large = BatchedCopyLargeBufferPolicy{
      256,
      32,
    };
    return BatchedCopyPolicy{
      BatchedCopySmallBufferPolicy{
        /* .threads_per_block = */ 128,
        /* .buffers_per_thread = */ 4,
        /* .bytes_per_thread = */ 8,
        /* .prefer_pow2_bits = */ cc < ::cuda::compute_capability{7, 0},
        /* .block_level_tile_size = */ large.threads_per_block * large.bytes_per_thread,
        /* .warp_level_threshold = */ 128,
        /* .block_level_threshold = */ 8 * 1024,
        // BufferOffsetT and BlockOffsetT are primitive/trivially copyable
        /* .buffer_lookback_delay = */ default_delay_constructor_policy(true),
        /* .block_lookback_delay = */ default_delay_constructor_policy(true)},
      large,
    };
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(batch_memcpy_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()
} // namespace detail::batch_memcpy

CUB_NAMESPACE_END
