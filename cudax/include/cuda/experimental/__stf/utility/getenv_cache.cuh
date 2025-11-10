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
 * @brief Cache getenv results
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

#include <iostream>
#include <string>
#include <unordered_map>

namespace cuda::experimental::stf::reserved
{
/**
 * @brief Retrieves the value of an environment variable, caching the result for subsequent calls.
 *
 * This function checks a thread-local cache to see if the environment variable
 * has already been retrieved. If the variable is found in the cache, its value is returned.
 * Otherwise, the function calls `std::getenv` to retrieve the value from the environment,
 * caches the result, and then returns it.
 *
 * @param name The name of the environment variable to retrieve.
 * @return The value of the environment variable as a C-style string, or `nullptr` if the variable is not set.
 *
 * @note The cache is thread-local, so each thread will have its own independent cache.
 */
inline const char* cached_getenv(const char* name)
{
  static thread_local ::std::unordered_map<::std::string, ::std::string> cache;

  if (auto it = cache.find(name); it != cache.end())
  {
    return it->second.c_str();
  }

  const char* value = ::std::getenv(name);
  if (value)
  {
    cache[name] = value;
  }

  return value;
}
} // end namespace cuda::experimental::stf::reserved
