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
 * @brief Partitioning granularity for execution places.
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

#include <cstdlib>
#include <string>

namespace cuda::experimental::stf
{

/**
 * @brief Partitioning granularity for execution places.
 */
enum class place_partition_scope
{
  cuda_device,
  green_context,
  cuda_stream,
};

/**
 * @brief Convert a place_partition_scope value to a string (for debugging).
 */
inline ::std::string place_partition_scope_to_string(place_partition_scope scope)
{
  switch (scope)
  {
    case place_partition_scope::cuda_device:
      return "cuda_device";
    case place_partition_scope::green_context:
      return "green_context";
    case place_partition_scope::cuda_stream:
      return "cuda_stream";
  }

  abort();
  return "unknown";
}

} // namespace cuda::experimental::stf
