// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/**
 * \brief Statically determine if N is a power-of-two
 * deprecated [since 3.2]
 */
template <int N>
struct [[deprecated("Use cuda::is_power_of_two(N) instead")]] PowerOfTwo
{
  static constexpr bool VALUE = (N & (N - 1)) == 0;
};
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
