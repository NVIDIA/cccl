/*
 *  Copyright 2008-2021 NVIDIA Corporation
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
