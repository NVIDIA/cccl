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
 * @brief Different constant values and some related methods
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

#include <cuda/experimental/__stf/utility/unittest.cuh>

namespace cuda::experimental::stf
{
/**
 * @brief Encodes an access mode (read, write, redux, ...)
 */
enum class access_mode : unsigned int
{
  none    = 0,
  read    = 1,
  write   = 2,
  rw      = 3, // READ + WRITE
  relaxed = 4, /* operator ? */
  reduce  = 8, // overwrite the content of the logical data (if any) with the result of the reduction (equivalent to
              // write)
  reduce_no_init = 16, // special case where the reduction will accumulate into the existing content (equivalent to rw)
};

/**
 * @brief Combines two access mode into a compatible access mode for both
 *         accesses (eg. read+write = rw, read+read = read)
 */
inline access_mode operator|(access_mode lhs, access_mode rhs)
{
  assert(as_underlying(lhs) < 16);
  assert(as_underlying(rhs) < 16);
  EXPECT(lhs != access_mode::relaxed);
  EXPECT(rhs != access_mode::relaxed);
  return access_mode(as_underlying(lhs) | as_underlying(rhs));
}

//! @brief In-place version of `operator|`
inline access_mode& operator|=(access_mode& lhs, access_mode rhs)
{
  return lhs = lhs | rhs;
}

/**
 * @brief Convert an access mode into a C string
 */
inline const char* access_mode_string(access_mode mode)
{
  switch (mode)
  {
    case access_mode::none:
      return "none";
    case access_mode::read:
      return "read";
    case access_mode::rw:
      return "rw";
    case access_mode::write:
      return "write";
    case access_mode::relaxed:
      return "relaxed"; // op ?
    case access_mode::reduce_no_init:
      return "reduce (no init)"; // op ?
    case access_mode::reduce:
      return "reduce"; // op ?
    default:
      assert(false);
      abort();
  }
}

/**
 * @brief A tag type used in combination with the reduce access mode to
 * indicate that we should not overwrite a logical data, but instead accumulate
 * the result of the reduction with the existing content of the logical data
 * (thus making it a rw access, instead a of a write-only access)
 */
class no_init
{};
} // namespace cuda::experimental::stf
