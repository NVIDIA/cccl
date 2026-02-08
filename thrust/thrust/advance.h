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

#include <cuda/std/__iterator/advance.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__iterator/prev.h>

THRUST_NAMESPACE_BEGIN

//! deprecated [since 3.1]
template <typename InputIterator, typename Distance>
CCCL_DEPRECATED_BECAUSE("Use ::cuda::std::advance instead") _CCCL_API constexpr void
advance(InputIterator& i, Distance n)
{
  ::cuda::std::advance(i, n);
}

//! deprecated [since 3.1]
template <typename InputIterator>
CCCL_DEPRECATED_BECAUSE("Use ::cuda::std::next instead") _CCCL_API constexpr InputIterator
next(InputIterator i, typename ::cuda::std::iterator_traits<InputIterator>::difference_type n = 1)
{
  return ::cuda::std::next(i, n);
}

//! deprecated [since 3.1]
template <typename InputIterator>
CCCL_DEPRECATED_BECAUSE("Use ::cuda::std::prev instead") _CCCL_API constexpr InputIterator
prev(InputIterator i, typename ::cuda::std::iterator_traits<InputIterator>::difference_type n = 1)
{
  return ::cuda::std::prev(i, n);
}

THRUST_NAMESPACE_END
