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

#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017)
#  include <nvtx3/nvToolsExt.h>
#endif

namespace cuda::experimental::stf
{

/**
 * @brief A RAII style wrapper for NVTX ranges
 */
class nvtx_range
{
public:
  nvtx_range(const char* message)
  {
#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017)
    nvtxRangePushA(message);
#endif
  }

  // Noncopyable and nonassignable to avoid multiple pops
  nvtx_range(const nvtx_range&)            = delete;
  nvtx_range(nvtx_range&&)                 = delete;
  nvtx_range& operator=(const nvtx_range&) = delete;

  ~nvtx_range()
  {
#if _CCCL_HAS_INCLUDE(<nvtx3/nvToolsExt.h>) && (!_CCCL_COMPILER(NVHPC) || _CCCL_STD_VER <= 2017)
    nvtxRangePop();
#endif
  }
};

} // end namespace cuda::experimental::stf
