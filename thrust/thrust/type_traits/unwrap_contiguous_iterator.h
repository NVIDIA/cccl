// SPDX-FileCopyrightText: Copyright (c) 2008-2021, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/std/__utility/declval.h>

THRUST_NAMESPACE_BEGIN

//! Converts a contiguous iterator to its underlying raw pointer.
_CCCL_EXEC_CHECK_DISABLE
template <typename ContiguousIterator>
_CCCL_HOST_DEVICE auto unwrap_contiguous_iterator(ContiguousIterator it)
{
  static_assert(thrust::is_contiguous_iterator_v<ContiguousIterator>,
                "unwrap_contiguous_iterator called with non-contiguous iterator.");
  return thrust::raw_pointer_cast(&*it);
}

//! Converts a contiguous iterator type to its underlying raw pointer type.
template <typename ContiguousIterator>
using unwrap_contiguous_iterator_t = decltype(unwrap_contiguous_iterator(::cuda::std::declval<ContiguousIterator>()));

//! Takes an iterator and, if it is contiguous, unwraps it to the raw pointer it represents. Otherwise returns the
//! iterator unmodified.
_CCCL_EXEC_CHECK_DISABLE
template <typename Iterator>
_CCCL_HOST_DEVICE auto try_unwrap_contiguous_iterator(Iterator it)
{
  if constexpr (thrust::is_contiguous_iterator_v<Iterator>)
  {
    return unwrap_contiguous_iterator(it);
  }
  else
  {
    return it;
  }
}

//! Takes an iterator type and, if it is contiguous, yields the raw pointer type it represents. Otherwise returns the
//! iterator type unmodified.
template <typename Iterator>
using try_unwrap_contiguous_iterator_t = decltype(try_unwrap_contiguous_iterator(::cuda::std::declval<Iterator>()));

THRUST_NAMESPACE_END
