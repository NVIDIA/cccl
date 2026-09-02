//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstddef>
#include <cstdint>

#include <cccl/c/device_copy.h>
#if __cplusplus >= 202002L && __has_include(<format>)
#  include <format>
#  if defined(__cpp_lib_format)
#    define CCCL_DEVICE_COPY_HAS_STD_FORMAT 1
#  endif // defined(__cpp_lib_format)
#endif // __cplusplus >= 202002L && __has_include(<format>)
#if !defined(CCCL_DEVICE_COPY_HAS_STD_FORMAT)
#  define CCCL_DEVICE_COPY_HAS_STD_FORMAT 0
#endif // !defined(CCCL_DEVICE_COPY_HAS_STD_FORMAT)
#include <string>

namespace cccl::detail::device_copy_codegen
{
inline std::string formatted_extent_argument(int64_t value)
{
#if CCCL_DEVICE_COPY_HAS_STD_FORMAT
  return std::format("{0}", value);
#else // ^^^ CCCL_DEVICE_COPY_HAS_STD_FORMAT ^^^ / vvv !CCCL_DEVICE_COPY_HAS_STD_FORMAT vvv
  return std::to_string(value);
#endif // !CCCL_DEVICE_COPY_HAS_STD_FORMAT
}

inline std::string formatted_runtime_extent_argument(const char* values, size_t axis, const char* type)
{
#if CCCL_DEVICE_COPY_HAS_STD_FORMAT
  return std::format("static_cast<{0}>({1}[{2}])", type, values, axis);
#else // ^^^ CCCL_DEVICE_COPY_HAS_STD_FORMAT ^^^ / vvv !CCCL_DEVICE_COPY_HAS_STD_FORMAT vvv
  std::string result;
  result += "static_cast<";
  result += type;
  result += ">(";
  result += values;
  result += "[";
  result += std::to_string(axis);
  result += "])";
  return result;
#endif // !CCCL_DEVICE_COPY_HAS_STD_FORMAT
}

inline std::string extents_template_arguments(const cccl_device_copy_axis_metadata_t* shape, size_t rank)
{
  std::string result;
  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (axis != 0)
    {
      result += ", ";
    }
    if (shape[axis].kind == CCCL_DEVICE_COPY_AXIS_STATIC)
    {
      result += formatted_extent_argument(shape[axis].value);
    }
    else
    {
      result += "::cuda::std::dynamic_extent";
    }
  }

  return result;
}

inline std::string dynamic_extent_constructor_arguments(
  const char* values, const cccl_device_copy_axis_metadata_t* shape, size_t rank, const char* type)
{
  std::string result;
  bool first_dynamic_axis = true;
  for (size_t axis = 0; axis < rank; ++axis)
  {
    if (shape[axis].kind != CCCL_DEVICE_COPY_AXIS_RUNTIME)
    {
      continue;
    }
    if (!first_dynamic_axis)
    {
      result += ", ";
    }
    first_dynamic_axis = false;

    result += formatted_runtime_extent_argument(values, axis, type);
  }

  return result;
}
} // namespace cccl::detail::device_copy_codegen

#undef CCCL_DEVICE_COPY_HAS_STD_FORMAT
