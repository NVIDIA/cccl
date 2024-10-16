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
 * @brief Source location
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

// GCC11 provides non constexpr builtins
#if defined(__cpp_lib_source_location) && (!defined(_CCCL_COMPILER_GCC) || _CCCL_GCC_VERSION >= 120000)
// C++20 version using std::source_location
#  include <source_location>

namespace cuda::experimental::stf
{
using source_location = ::std::source_location;
}
#  define RESERVED_STF_SOURCE_LOCATION() ::cuda::experimental::stf::source_location::current()

#elif __has_include(<experimental/source_location>)
#  include <experimental/source_location>
namespace cuda::experimental::stf
{
using source_location = ::std::experimental::source_location;
}

#  define RESERVED_STF_SOURCE_LOCATION() ::cuda::experimental::stf::source_location::current()

#else
// Custom implementation for C++17 or earlier
namespace cuda::experimental::stf
{
class source_location
{
public:
  constexpr source_location(
    const char* file = "unknown file", int line = 0, const char* func = "unknown function") noexcept
      : file_(file)
      , line_(line)
      , func_(func)
  {}

  constexpr const char* file_name() const noexcept
  {
    return file_;
  }
  constexpr int line() const noexcept
  {
    return line_;
  }
  constexpr const char* function_name() const noexcept
  {
    return func_;
  }

  static constexpr source_location
  current(const char* file = __FILE__, int line = __LINE__, const char* func = __func__) noexcept
  {
    return source_location(file, line, func);
  }

private:
  const char* file_;
  int line_;
  const char* func_;
};
} // end namespace cuda::experimental::stf

#  define RESERVED_STF_SOURCE_LOCATION() \
    ::cuda::experimental::stf::source_location::current(__FILE__, __LINE__, __func__)
#endif
