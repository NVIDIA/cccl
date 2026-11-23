// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_load.cuh>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
namespace detail
{
// Synthesizing iterators produce their values without accessing memory, so exchanging them through shared memory to
// coalesce memory accesses only costs shared memory and synchronization. Loading them directly yields the same blocked
// arrangement.
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
load_algorithm_for_input(BlockLoadAlgorithm load_algorithm, bool input_synthesizing) noexcept -> BlockLoadAlgorithm
{
  if (input_synthesizing
      && (load_algorithm == BLOCK_LOAD_TRANSPOSE || load_algorithm == BLOCK_LOAD_WARP_TRANSPOSE
          || load_algorithm == BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED))
  {
    return BLOCK_LOAD_DIRECT;
  }
  return load_algorithm;
}
} // namespace detail
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
