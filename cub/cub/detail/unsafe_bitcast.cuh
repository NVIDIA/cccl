// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN
namespace detail
{
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

// NOTE: bit_cast cannot be always used because __half, __nv_bfloat16, etc. are not trivially copyable
template <typename Output, typename Input>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE Output unsafe_bitcast(const Input& input)
{
  Output output;
  static_assert(sizeof(input) == sizeof(output), "wrong size");
  ::memcpy(&output, &input, sizeof(input));
  return output;
}

#endif // !_CCCL_DOXYGEN_INVOKED
} // namespace detail
CUB_NAMESPACE_END
