/*
 *  Copyright 2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
