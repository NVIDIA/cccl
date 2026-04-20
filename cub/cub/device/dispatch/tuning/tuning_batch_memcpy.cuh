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

#include <cuda/__device/arch_id.h>
#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

namespace detail::batch_memcpy
{
struct small_buffer_policy
{
  int block_threads;
  int buffers_per_thread;
  int tlev_bytes_per_thread;
  bool prefer_pow2_bits;
  int block_level_tile_size;
  int warp_level_threshold;
  int block_level_threshold;
  delay_constructor_policy buff_delay_constructor;
  delay_constructor_policy block_delay_constructor;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const small_buffer_policy& lhs, const small_buffer_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.buffers_per_thread == rhs.buffers_per_thread
        && lhs.tlev_bytes_per_thread == rhs.tlev_bytes_per_thread && lhs.prefer_pow2_bits == rhs.prefer_pow2_bits
        && lhs.block_level_tile_size == rhs.block_level_tile_size
        && lhs.warp_level_threshold == rhs.warp_level_threshold
        && lhs.block_level_threshold == rhs.block_level_threshold
        && lhs.buff_delay_constructor == rhs.buff_delay_constructor
        && lhs.block_delay_constructor == rhs.block_delay_constructor;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const small_buffer_policy& lhs, const small_buffer_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const small_buffer_policy& policy)
  {
    return os
        << "small_buffer_policy { .block_threads = " << policy.block_threads << ", .buffers_per_thread = "
        << policy.buffers_per_thread << ", .tlev_bytes_per_thread = " << policy.tlev_bytes_per_thread
        << ", .prefer_pow2_bits = " << policy.prefer_pow2_bits << ", .block_level_tile_size = "
        << policy.block_level_tile_size << ", .warp_level_threshold = " << policy.warp_level_threshold
        << ", .block_level_threshold = " << policy.block_level_threshold << ", .buff_delay_constructor = "
        << policy.buff_delay_constructor << ", .block_delay_constructor = " << policy.block_delay_constructor << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct large_buffer_policy
{
  int block_threads;
  int bytes_per_thread;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const large_buffer_policy& lhs, const large_buffer_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.bytes_per_thread == rhs.bytes_per_thread;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const large_buffer_policy& lhs, const large_buffer_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const large_buffer_policy& policy)
  {
    return os << "large_buffer_policy { .block_threads = " << policy.block_threads
              << ", .bytes_per_thread = " << policy.bytes_per_thread << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

struct batch_memcpy_policy
{
  small_buffer_policy small_buffer;
  large_buffer_policy large_buffer;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const batch_memcpy_policy& lhs, const batch_memcpy_policy& rhs)
  {
    return lhs.small_buffer == rhs.small_buffer && lhs.large_buffer == rhs.large_buffer;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const batch_memcpy_policy& lhs, const batch_memcpy_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const batch_memcpy_policy& policy)
  {
    return os << "batch_memcpy_policy { .small_buffer = " << policy.small_buffer
              << ", .large_buffer = " << policy.large_buffer << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept batch_memcpy_policy_selector = policy_selector<T, batch_memcpy_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> batch_memcpy_policy
  {
    const auto large = large_buffer_policy{
      256,
      32,
    };
    return batch_memcpy_policy{
      small_buffer_policy{
        /* .block_threads = */ 128,
        /* .buffers_per_thread = */ 4,
        /* .tlev_bytes_per_thread = */ 8,
        /* .prefer_pow2_bits = */ arch < ::cuda::arch_id::sm_70,
        /* .block_level_tile_size = */ large.block_threads * large.bytes_per_thread,
        /* .warp_level_threshold = */ 128,
        /* .block_level_threshold = */ 8 * 1024,
        // BufferOffsetT and BlockOffsetT are primitive/trivially copyable
        /* .buff_delay_constructor = */ default_delay_constructor_policy(true),
        /* .block_delay_constructor = */ default_delay_constructor_policy(true)},
      large,
    };
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(batch_memcpy_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()
} // namespace detail::batch_memcpy

CUB_NAMESPACE_END
