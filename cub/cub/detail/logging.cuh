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

#include <cuda/std/cstdlib>
#include <cuda/std/string_view>

CUB_NAMESPACE_BEGIN
namespace detail
{
//! Returns if logging is enabled
_CCCL_HOST_API inline bool logging_enabled()
{
  static bool enabled = ::std::getenv("CCCL_EXPERIMENTAL_LOGGING") != nullptr;
  return enabled;
}

// TODO(bgruber): switch to an interface like std::print once davebayer has implemented <cuda/std/format>
//! Logs the message, independently of whether logging is enabled
_CCCL_HOST_API inline void log_always([[maybe_unused]] ::cuda::std::string_view message)
{
#if _CCCL_HOSTED()
  ::printf("%.*s", static_cast<int>(message.size()), message.data());
#endif // _CCCL_HOSTED()
}

// TODO(bgruber): switch to an interface like std::print once davebayer has implemented <cuda/std/format>
//! Logs the message when logging is enabled
_CCCL_HOST_API inline void log(::cuda::std::string_view message)
{
  if (logging_enabled())
  {
    log_always(message);
  }
}
} // namespace detail
CUB_NAMESPACE_END
