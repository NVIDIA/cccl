// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
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

#include <cuda/std/__iterator/distance.h>
#include <cuda/std/__iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

//! deprecated [since 3.1]

template <class InputIter>
[[nodiscard]] CCCL_DEPRECATED_BECAUSE("Use cuda::std::distance instead")
_CCCL_API constexpr typename ::cuda::std::iterator_traits<InputIter>::difference_type
distance(InputIter first, InputIter last)
{
  return ::cuda::std::distance(first, last);
}

THRUST_NAMESPACE_END
