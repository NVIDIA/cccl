//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Utilities for source_location
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

#include <cuda/std/source_location>

namespace cuda::experimental::stf::reserved
{

struct source_location_hash
{
  /* We use const char * and not string because these are string literals,
   * and it is safe to assume they are not going to change. We also take the
   * function name into account because the same callsite could be used in
   * different instantiation of the same templated class, the name will reflect
   * the template parameters. */
  ::std::size_t operator()(const ::cuda::std::source_location& loc) const noexcept
  {
    return ::std::hash<const char*>{}(loc.file_name()) ^ (::std::hash<uint_least32_t>{}(loc.line()) << 1)
         ^ (::std::hash<uint_least32_t>{}(loc.column()) << 2) ^ (::std::hash<const char*>{}(loc.function_name()) << 3);
  }
};

struct source_location_equal
{
  bool operator()(const ::cuda::std::source_location& lhs, const ::cuda::std::source_location& rhs) const noexcept
  {
    // Comparing const char * is legit here because these are string literal constants
    return lhs.file_name() == rhs.file_name() && lhs.line() == rhs.line() && lhs.column() == rhs.column()
        && lhs.function_name() == rhs.function_name();
  }
};

} // end namespace cuda::experimental::stf::reserved
