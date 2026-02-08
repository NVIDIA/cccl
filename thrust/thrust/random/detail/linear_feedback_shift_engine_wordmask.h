// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

THRUST_NAMESPACE_BEGIN

namespace random::detail
{
template <typename T, int w, int i = w - 1>
struct linear_feedback_shift_engine_wordmask
{
  static const T value = (T(1u) << i) | linear_feedback_shift_engine_wordmask<T, w, i - 1>::value;
}; // end linear_feedback_shift_engine_wordmask

template <typename T, int w>
struct linear_feedback_shift_engine_wordmask<T, w, 0>
{
  static const T value = 0;
}; // end linear_feedback_shift_engine_wordmask
} // namespace random::detail

THRUST_NAMESPACE_END
