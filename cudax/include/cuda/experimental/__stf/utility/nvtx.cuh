//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Wrappers for NVTX
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017) \
                      && !defined(CCCL_DISABLE_NVTX)                                             \
                      && !defined(NVTX_DISABLE)
#  include <nvtx3/nvToolsExt.h>
#endif

#include <type_traits>
#include <utility>

namespace cuda::experimental::stf
{

/**
 * @brief A RAII style wrapper for NVTX ranges
 */
class nvtx_range
{
public:
  explicit nvtx_range(const char* message)
  {
#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017) \
                      && !defined(CCCL_DISABLE_NVTX)                                             \
                      && !defined(NVTX_DISABLE)
    nvtxRangePushA(message);
#endif
    static_assert(::std::is_move_constructible_v<nvtx_range>, "nvtx_range must be move constructible");
    static_assert(::std::is_move_assignable_v<nvtx_range>, "nvtx_range must be move assignable");
  }

  // Noncopyable to avoid multiple pops
  nvtx_range(const nvtx_range&)            = delete;
  nvtx_range& operator=(const nvtx_range&) = delete;

  // Move constructor
  nvtx_range(nvtx_range&& other) noexcept
      : active(::std::exchange(other.active, false))
  {}

  // Move assignment
  nvtx_range& operator=(nvtx_range&& other) noexcept
  {
    if (this != &other)
    {
      end(); // Ensure the current range is properly closed
      active = std::exchange(other.active, false);
    }
    return *this;
  }

  // Explicitly end the NVTX range
  void end()
  {
    if (!active)
    {
      return; // Prevent double pop
    }
    active = false;

#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017) \
                      && !defined(CCCL_DISABLE_NVTX)                                             \
                      && !defined(NVTX_DISABLE)
    nvtxRangePop();
#endif
  }

  ~nvtx_range()
  {
    end();
  }

private:
  bool active = true; // Tracks if nvtxRangePop() should be called
};

} // end namespace cuda::experimental::stf
