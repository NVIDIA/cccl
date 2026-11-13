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
 * \brief Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE   // 3
 *     Log2<3>::VALUE   // 2
 */
template <int N, int CurrentVal = N, int COUNT = 0>
struct Log2
{
  /// Static logarithm value
  static constexpr int VALUE = Log2<N, (CurrentVal >> 1), COUNT + 1>::VALUE;
};

#  ifndef _CCCL_DOXYGEN_INVOKED // Do not document

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
  static constexpr int VALUE = (1 << (COUNT - 1) < N) ? COUNT : COUNT - 1;
};

#  endif // _CCCL_DOXYGEN_INVOKED
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
