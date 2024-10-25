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

#elif __has_include(<experimental/source_location>)
#  include <experimental/source_location>
namespace cuda::experimental::stf
{
using source_location = ::std::experimental::source_location;
}

#else
// Custom implementation for C++17 or earlier
namespace cuda::experimental::stf
{
class source_location
{
public:
  constexpr source_location() noexcept = default;

  constexpr const char* file_name() const noexcept
  {
    return file_;
  }
  constexpr ::std::uint_least32_t line() const noexcept
  {
    return line_;
  }
  constexpr ::std::uint_least32_t column() const noexcept
  {
    return column_;
  }
  constexpr const char* function_name() const noexcept
  {
    return func_;
  }

  static constexpr source_location
  current(const char* file = __builtin_FILE(), ::std::uint_least32_t line = __builtin_LINE(), ::std::uint_least32_t column = __builtin_COLUMN(), const char* func = __builtin_FUNCTION()) noexcept
  {
    return source_location(file, line, column, func);
  }

private:
  constexpr source_location(const char* file, ::std::uint_least32_t line, ::std::uint_least32_t column, const char* func) noexcept
      : file_(file)
      , line_(line)
      , column(_column)
      , func_(func)
  {}

  const char* file_ = "";
  ::std::uint_least32_t line_ = 0;
  ::std::uint_least32_t column_ = 0;
  const char* func_ = "";
};

} // end namespace cuda::experimental::stf

#endif
