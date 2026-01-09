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
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/util_device.cuh>

CUB_NAMESPACE_BEGIN

namespace detail::batch_memcpy
{
/**
 * Parameterizable tuning policy type for AgentBatchMemcpy
 */
template <uint32_t _BLOCK_THREADS, uint32_t _BYTES_PER_THREAD>
struct agent_large_buffer_policy
{
  /// Threads per thread block
  static constexpr uint32_t BLOCK_THREADS = _BLOCK_THREADS;
  /// The number of bytes each thread copies
  static constexpr uint32_t BYTES_PER_THREAD = _BYTES_PER_THREAD;
};

template <class BufferOffsetT, class BlockOffsetT>
struct policy_hub
{
  static constexpr uint32_t BLOCK_THREADS         = 128U;
  static constexpr uint32_t BUFFERS_PER_THREAD    = 4U;
  static constexpr uint32_t TLEV_BYTES_PER_THREAD = 8U;

  static constexpr uint32_t LARGE_BUFFER_BLOCK_THREADS    = 256U;
  static constexpr uint32_t LARGE_BUFFER_BYTES_PER_THREAD = 32U;

  static constexpr uint32_t WARP_LEVEL_THRESHOLD  = 128;
  static constexpr uint32_t BLOCK_LEVEL_THRESHOLD = 8 * 1024;

  using buff_delay_constructor_t  = detail::default_delay_constructor_t<BufferOffsetT>;
  using block_delay_constructor_t = detail::default_delay_constructor_t<BlockOffsetT>;

  /// SM50
  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr bool PREFER_POW2_BITS = true;
    using AgentSmallBufferPolicyT          = AgentBatchMemcpyPolicy<
               BLOCK_THREADS,
               BUFFERS_PER_THREAD,
               TLEV_BYTES_PER_THREAD,
               PREFER_POW2_BITS,
               LARGE_BUFFER_BLOCK_THREADS * LARGE_BUFFER_BYTES_PER_THREAD,
               WARP_LEVEL_THRESHOLD,
               BLOCK_LEVEL_THRESHOLD,
               buff_delay_constructor_t,
               block_delay_constructor_t>;

    using AgentLargeBufferPolicyT =
      agent_large_buffer_policy<LARGE_BUFFER_BLOCK_THREADS, LARGE_BUFFER_BYTES_PER_THREAD>;
  };

  /// SM70
  struct Policy700 : ChainedPolicy<700, Policy700, Policy500>
  {
    static constexpr bool PREFER_POW2_BITS = false;
    using AgentSmallBufferPolicyT          = AgentBatchMemcpyPolicy<
               BLOCK_THREADS,
               BUFFERS_PER_THREAD,
               TLEV_BYTES_PER_THREAD,
               PREFER_POW2_BITS,
               LARGE_BUFFER_BLOCK_THREADS * LARGE_BUFFER_BYTES_PER_THREAD,
               WARP_LEVEL_THRESHOLD,
               BLOCK_LEVEL_THRESHOLD,
               buff_delay_constructor_t,
               block_delay_constructor_t>;

    using AgentLargeBufferPolicyT =
      agent_large_buffer_policy<LARGE_BUFFER_BLOCK_THREADS, LARGE_BUFFER_BYTES_PER_THREAD>;
  };

  using MaxPolicy = Policy700;
};
} // namespace detail::batch_memcpy

CUB_NAMESPACE_END
