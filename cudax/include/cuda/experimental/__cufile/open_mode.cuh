//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/to_underlying.h>

namespace cuda::experimental
{
//! @brief Open mode for cufile.
enum class cufile_open_mode : unsigned
{
  in        = (1u << 0),
  out       = (1u << 1),
  trunc     = (1u << 2),
  noreplace = (1u << 3),
  direct    = (1u << 4),
};

[[nodiscard]] _CCCL_HOST_API constexpr cufile_open_mode
operator|(cufile_open_mode __lhs, cufile_open_mode __rhs) noexcept
{
  return static_cast<cufile_open_mode>(::cuda::std::to_underlying(__lhs) | ::cuda::std::to_underlying(__rhs));
}

_CCCL_HOST_API constexpr cufile_open_mode& operator|=(cufile_open_mode& __lhs, cufile_open_mode __rhs) noexcept
{
  return __lhs = __lhs | __rhs;
}

[[nodiscard]] _CCCL_HOST_API constexpr cufile_open_mode
operator&(cufile_open_mode __lhs, cufile_open_mode __rhs) noexcept
{
  return static_cast<cufile_open_mode>(::cuda::std::to_underlying(__lhs) & ::cuda::std::to_underlying(__rhs));
}

_CCCL_HOST_API constexpr cufile_open_mode& operator&=(cufile_open_mode& __lhs, cufile_open_mode __rhs) noexcept
{
  return __lhs = __lhs & __rhs;
}

[[nodiscard]] _CCCL_HOST_API constexpr cufile_open_mode
operator^(cufile_open_mode __lhs, cufile_open_mode __rhs) noexcept
{
  return static_cast<cufile_open_mode>(::cuda::std::to_underlying(__lhs) ^ ::cuda::std::to_underlying(__rhs));
}

_CCCL_HOST_API constexpr cufile_open_mode& operator^=(cufile_open_mode& __lhs, cufile_open_mode __rhs) noexcept
{
  return __lhs = __lhs ^ __rhs;
}

[[nodiscard]] _CCCL_HOST_API constexpr cufile_open_mode operator~(cufile_open_mode __b) noexcept
{
  return static_cast<cufile_open_mode>(~::cuda::std::to_underlying(__b));
}
} // namespace cuda::experimental
